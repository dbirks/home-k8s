---
name: kiwix-library
description: Operate the home Kiwix offline library (apps/kiwix, issue 164) - kiwix-serve on the HDD at kiwix.beluga-wyvern.ts.net, and the kiwix-mcp broker Hermes uses to search the Kiwix catalog, queue user-approved ZIM downloads, watch progress and full-text search what is on disk. Use when adding or removing archives, debugging a stuck or failed download, upgrading kiwix-tools, or wiring another agent to the library.
---

# Kiwix offline library

## What runs (all in `default`, pinned to talos-210-73x because `local-path-hdd` only exists there)
| Piece | File | Notes |
|---|---|---|
| `kiwix` Deployment | `apps/kiwix/kiwix.yaml` | `kiwix-serve --monitorLibrary --library /library/library.xml`, library mounted **read-only**, uid 65534. Init `layout` (root, CHOWN only) creates `zim/`, `.queue/` and an empty `library.xml` on first boot. |
| `kiwix-ts` Ingress | same | tailnet only: https://kiwix.beluga-wyvern.ts.net/ |
| `kiwix-library` PVC | same | 500Gi request on the HDD (not enforced by local-path). |
| `kiwix-mcp` Deployment | `apps/kiwix/kiwix-mcp.yaml` | stdlib Python broker (`kiwix_mcp.py`, shipped as ConfigMap `kiwix-mcp-code.yaml`), MCP at `http://kiwix-mcp.default.svc.cluster.local:8080/mcp`. The init container copies the static `kiwix-manage` out of the kiwix-tools image. |
| `kiwix-mcp-token` | `apps/kiwix/kiwix-mcp-token.enc.yaml` | SOPS. Hermes gets it as env `KIWIX_MCP_TOKEN`. |

Both images are `ghcr.io/kiwix/kiwix-tools` pinned by digest. The `kiwix-serve` image is the same static binaries plus a `start.sh` we don't need.

Volume layout:
- `/library/library.xml`: the book index. Only `kiwix-manage` (in the broker) writes it.
- `/library/zim/*.zim`: the archives.
- `/library/.queue/queue.json`: broker state.

## How a download works
1. `search_catalog` reads the OPDS feed, e.g. `https://library.kiwix.org/catalog/v2/entries?q=&lang=eng&category=wikipedia&count=N`.
   - `lang` is ISO-639-3.
   - The acquisition link is `https://lb.download.kiwix.org/zim/<cat>/<file>.zim.meta4`, and its `length` is approximate.
2. `queue_download(url)`:
   - Accepts only `download.kiwix.org` (or `lb.`) `/zim/<cat>/<file>.zim`, with or without `.meta4`.
   - Reads the exact size and sha-256 from the `.meta4`.
   - Refuses if free space minus pending bytes minus this archive would drop below `MIN_FREE_GIB` (100).
   - Is idempotent: a file already on disk or already queued gives back `queued:false` with a reason.
3. The worker thread runs **one at a time**:
   - Downloads to `zim/<file>.part`, resuming with HTTP Range.
   - Checks the size, then the sha-256.
   - Renames atomically, then runs `kiwix-manage library.xml add`.
   - kiwix-serve notices the newer mtime within about 1 s and reloads. No restart is needed.
4. Failures are retried with backoff (30 s × tries, max 10 min). After 5 tries the job becomes `failed`, and calling `queue_download` again re-queues it.

**Nothing is ever deleted.** A bad hash keeps the file as `.part.badhash`; an oversize partial becomes `.part.oversize`.

Tools: `search_catalog`, `queue_download`, `list_queue`, `list_library`, `search_library`. The `queue_download` description tells the agent to show title, flavour, size and total, and to get explicit approval first.

## Hermes wiring (by hand on the PVC, like the other MCP servers)
```sh
kubectl exec deploy/hermes -c hermes -- hermes config set mcp_servers.kiwix '{url: "http://kiwix-mcp.default.svc.cluster.local:8080/mcp", headers: {Authorization: "Bearer ${KIWIX_MCP_TOKEN}"}, tools: {include: [search_catalog, queue_download, list_queue, list_library, search_library]}}'
kubectl exec deploy/hermes -c hermes -- hermes mcp test kiwix     # expect 5 tools
```

## Verify / operate
```sh
kubectl logs deploy/kiwix-mcp -c broker | tail      # "queued X", "download start/done X", attempt N errors
kubectl logs deploy/kiwix | tail                    # "The library was successfully loaded." after each add
kubectl exec deploy/kiwix-mcp -c broker -- ls -la /library/zim
kubectl exec deploy/kiwix-mcp -c broker -- cat /library/.queue/queue.json
curl -s "https://kiwix.beluga-wyvern.ts.net/search?pattern=piano&books.name=<book>" | head   # from the tailnet
```
To call a tool by hand without ever seeing the token, run curl inside the Hermes container, where `$KIWIX_MCP_TOKEN` is in env:
```sh
kubectl exec -i deploy/hermes -c hermes -- sh -c 'curl -s http://kiwix-mcp.default.svc.cluster.local:8080/mcp -H "Authorization: Bearer $KIWIX_MCP_TOKEN" -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"list_queue\",\"arguments\":{}}}"'
```

## Recovery
- **Stuck `downloading` after a crash.** On start the broker resets `downloading`/`verifying` jobs to `queued` and resumes from the `.part`. Just restart the pod: `kubectl rollout restart deploy/kiwix-mcp`.
- **`failed` job.** Read `error` in `list_queue`, then call `queue_download` again with the same URL.
- **sha mismatch.** Look at the `.part.badhash` file and remove it by hand (a human decision). Then re-queue.
- **Remove an archive (human only, no tool for it).**
  1. `kubectl exec deploy/kiwix-mcp -c broker -- /tools/kiwix-manage /library/library.xml remove <book-id>`. The book id is the `id` attribute in `library.xml`.
  2. Then `rm /library/zim/<file>.zim`.
- **Replace with a newer snapshot.** Queue the new dated file, then remove the old one as above once you're happy. The broker never replaces anything itself.
- **kiwix-serve won't start.** It exits if `library.xml` is missing or unparsable. The `layout` init only creates it when empty or missing; if it's corrupt, rebuild it: move it aside, then `kiwix-manage /library/library.xml add /library/zim/*.zim`.

## Gotchas
- kiwix-serve with an **empty** library just warns (`is empty (or has only remote books)`) and serves an empty catalog. That's fine.
- `library.xml` stores **relative** paths (`zim/<file>`), so any mount point works as long as `library.xml` sits next to `zim/`.
- The download mirror can be slow (~150 KB/s seen for release tarballs). Wikipedia `maxi` archives are 100+ GiB and take hours to days.
- The HDD is shared with Jellyfin, Frigate and others. The 100 GiB floor is about the whole disk, not this PVC.
- The broker's `queued` check is per filename, and every Kiwix snapshot has a dated name, so a newer snapshot is a new file.
- Upgrading: bump both `kiwix-tools` digests. kiwix-serve and kiwix-manage should match versions.
- Regenerate the ConfigMap after editing `kiwix_mcp.py` (command in `kiwix-mcp.yaml`'s header).
