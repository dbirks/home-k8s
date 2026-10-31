---
name: youtarr
description: Operate Youtarr (apps/youtarr.yaml, issue 166) - the YouTube channel/playlist downloader that feeds the Jellyfin library from the HDD, its MariaDB sidecar, login, cookies, Jellyfin hookup, and the youtarr-mcp adapter Hermes uses to queue videos and subscribe channels. Use when adding YouTube content, debugging a download, upgrading Youtarr or MariaDB, or wiring another agent to it.
---

# Youtarr

- UI: https://youtarr.beluga-wyvern.ts.net (tailnet only, Tailscale Ingress `youtarr-ts`). Swagger at `/swagger`.
- Manifests: `apps/youtarr.yaml` (PVC + Deployment + Service + Ingress), `apps/youtarr-secrets.enc.yaml`,
  `apps/youtarr-mcp/` (adapter). Upstream: https://github.com/DialmasterOrg/Youtarr (docs/ folder).
- Pod `youtarr` on talos-210-73x (the HDD only exists there): init `perms` (chowns the local-path dirs),
  native-sidecar `mariadb` (10.11 LTS, 127.0.0.1:3306, uid 999), main `youtarr` (uid 1000, port 3011).

## Storage (all HDD)
| What | Where |
|---|---|
| Videos | `jellyfin-media-hdd` subPath `youtarr` = `/usr/src/app/data` in Youtarr = `/media/youtarr` in Jellyfin |
| config.json, logs, setup-token | `youtarr-data` subPath `config` (`/app/config`) |
| jobs, thumbnails | `youtarr-data` subPaths `jobs`, `images` |
| MariaDB | `youtarr-data` subPath `mysql` |

Layout is `/media/youtarr/<channel>/<channel> - <title> - <id>/<file>.mp4` + `.jpg`, with `poster.jpg` per channel.
The older `youtube-media-capture` CronJob writes other folders under `/media` and keeps its own archive; the two
don't share state, so the same video could download twice if queued in both.

## Login
Single account `david`. The password is `AUTH_PRESET_PASSWORD` in the `youtarr-secrets` Secret and is
re-applied on **every restart**, so changing it in the UI only lasts until the next pod start. The user reads it
from the live cluster themselves (never paste it into chat or a skill):
```bash
kubectl get secret youtarr-secrets -o jsonpath='{.data.AUTH_PRESET_PASSWORD}' | base64 -d; echo
```
Rotate: generate a new value into a 0600 temp file, rebuild the Secret with `kubectl create secret ... --dry-run`
piped through `sops encrypt`, commit and reconcile. The MCP adapter reads the same key, so restart it too
(`kubectl rollout restart deploy/youtarr-mcp`; it re-logs in on 401 anyway). The age private key is not on the
workstation, so you can't `sops -d` the committed file: rewrite the whole Secret (keep DB_PASSWORD and
MARIADB_ROOT_PASSWORD by copying them from the live Secret into the temp files, never printing them).

## One-time UI steps (manual)
1. Jellyfin: Dashboard -> Libraries -> add a library (type "Shows" or "Mixed") for folder `/media/youtarr`.
2. Youtarr Settings -> Media server: Jellyfin URL `http://jellyfin.default.svc.cluster.local:8096` plus a Jellyfin
   API key (minted in Jellyfin Dashboard -> API Keys) for library refresh, playlist mirroring and watch status.
   Upstream guide: docs/media-servers/jellyfin.md.
3. Cookies (only if YouTube bot-checks or age-gates): `YOUTARR_COOKIES_FILE=/cookies/cookies.txt` is already set
   from the `ncc-youtube-cookies` Secret (read-only; Youtarr validates a private copy). Turn on
   Settings -> Cookies -> Enable Cookies.

## Agent access (youtarr-mcp)
Youtarr API keys (`x-api-key`) can ONLY call `POST /api/videos/download`. Everything else needs a 7-day session
token (`x-access-token` from `POST /auth/login`). So `apps/youtarr-mcp/youtarr_mcp.py` logs in with the preset
password and exposes an allowlist: `queue_video`, `list_channels`, `add_channel` (addchannelinfo, then
updatechannels `{add:[{url, channel_id}]}`), `run_channel_downloads`, `list_playlists`, `add_playlist`,
`list_videos`, `download_status`. No delete, unsubscribe, config, auth, key or maintenance routes.
- Hermes: env `YOUTARR_MCP_TOKEN` (Secret `youtarr-mcp-token`) in apps/hermes-agent.yaml, plus
  `mcp_servers.youtarr` in /opt/data/config.yaml (set by hand; see the hermes-agent skill for the pattern).
  Check with `kubectl exec deploy/hermes -c hermes -- hermes mcp test youtarr`.
- After editing the .py, regenerate the ConfigMap (command in the youtarr-mcp.yaml header) and roll the pod.
- Smoke test from the Hermes pod without printing the token:
```bash
kubectl exec deploy/hermes -c hermes -- sh -c 'curl -s http://youtarr-mcp.default.svc.cluster.local:8080/mcp \
  -H "Authorization: Bearer $YOUTARR_MCP_TOKEN" -H "Content-Type: application/json" \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"download_status\",\"arguments\":{}}}"'
```

## Verify / debug
```bash
kubectl get pod -l app=youtarr                     # 2/2 = youtarr + mariadb sidecar
kubectl logs deploy/youtarr -c youtarr --tail=50   # treat as sensitive before first setup (setup token)
kubectl logs deploy/youtarr -c mariadb --tail=20
kubectl exec deploy/jellyfin -- ls -la /media/youtarr
curl -s https://youtarr.beluga-wyvern.ts.net/api/health
```
- The first request to a new Tailscale Ingress can fail for a minute while its cert is issued.
- MariaDB listens on 127.0.0.1 only, so its probes use the image's `healthcheck.sh --connect` (a TCP probe to the
  pod IP always fails).
- Known upstream noise: "Error downloading channel thumbnail ... Unable to find a suitable output format for
  '...jpg.temp'" is an ffmpeg output-extension bug in Youtarr's thumbnail resize. The original thumbnail stays and
  the video downloads fine.
- The pod-security "restricted" warning on the `perms` init container (root + CHOWN) is expected; it only warns.

## Upgrades
- Bump the image tag AND digest in both the `perms` and `youtarr` containers (same image).
- **Never bump MariaDB and Youtarr in the same commit.** Upgrade MariaDB, wait for `ready for connections`
  and a healthy Youtarr, then bump Youtarr (its migrations run on start). Upstream: docs/platforms/external-db.md.
- Back up first: `kubectl exec deploy/youtarr -c mariadb -- sh -c 'mariadb-dump -uroot -p"$MARIADB_ROOT_PASSWORD" youtarr' > backup.sql`
  (keep it out of the repo), plus `config/config.json`.

## Recovery
- Locked out: impossible while AUTH_PRESET is set; restarting re-applies the Secret's password.
- DB broken: scale to 0, restore the dump into a fresh `mysql/` subPath, scale up. Videos on disk are untouched (the nightly videoRescan job may re-register them; unverified).
