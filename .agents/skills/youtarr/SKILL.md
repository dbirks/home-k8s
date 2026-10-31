---
name: youtarr
description: Operate Youtarr (apps/youtarr.yaml, issue 166) - the YouTube channel/playlist downloader that feeds the Jellyfin library from the HDD, its MariaDB sidecar, login, cookies, Jellyfin hookup, the youtarr-mcp adapter Hermes uses to queue videos and subscribe channels, server-side API routes the MCP omits, and importing an existing yt-dlp download library (e.g. the retired youtube-media-capture job) without re-downloading. Use when adding YouTube content, debugging a download, upgrading Youtarr or MariaDB, migrating imported media, or wiring another agent to it.
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

Layout is `/media/youtarr/<channel>/<channel> - <title> - <id>/<channel> - <title> [<id>].<ext>` + `poster.jpg` per channel.
The file->DB join is the **trailing `[<videoId>].<ext>` in the filename** (`scanForVideoFiles` walks the whole output dir);
the ` - <id>` folder suffix is a secondary extraction path. Channel dir = `%(uploader,channel,uploader_id)` truncated to
80 bytes; title in names is truncated to 64 bytes, yt-dlp-sanitized. NOTE: the `YOUTUBE_OUTPUT_DIR` env in the manifest is
display-only (`safeConfig.youtubeOutputDirectory`); the effective output dir is `DATA_PATH || /usr/src/app/data`, which is
where the media PVC `youtarr` subPath is mounted. The `youtube-media-capture` CronJob was retired 2026-10-31 and its
library was imported (see "Importing an existing yt-dlp library" below); `/media/.youtube-capture-archive` is kept as the
historical record.

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

## Server-side API calls the MCP does not expose (session token)
The youtarr-mcp allowlist covers subscribe/list/queue only. Following, schedules, rescan and config need a session
token, and the password is already an env var inside the youtarr container — call `localhost:3011` from `kubectl exec`
so nothing leaves the pod. Do the whole flow in one `node` script piped via `kubectl exec -i deploy/youtarr -c youtarr -- node`
(the image has node + curl; there is no jq). Node v22 has `fetch`:
```js
const login = await (await fetch('http://localhost:3011/auth/login', {method:'POST',
  headers:{'content-type':'application/json'}, body: JSON.stringify({username:'david', password: process.env.AUTH_PRESET_PASSWORD})})).json();
const H = { 'x-access-token': login.token, 'content-type': 'application/json' };
```
Routes used (all verified on v1.90.2):
- `POST /api/playlists/:id/following {}` -> turns `auto_download` on and stamps the "from now" baseline (new list entries only, never the backlog).
- `POST /api/playlists/:id/download {}` -> queues every eligible (not downloaded, not archived, not ignored) video. On a fully imported library it correctly returns `{"queued":0,"message":"No eligible videos to queue"}`.
- `POST /api/schedules/<key>/run` (202) + `GET /api/schedules` for status/lastRun. Task keys include `archiveBackfillFrequency` ("Repair library records"), `videoRescanFrequency`, `channelDownloadFrequency`.
- `POST /api/maintenance/rescan-files {ignoreLibraryCheck:false}` + `GET /api/maintenance/rescan-status`.
- `GET /getconfig` / `POST /updateconfig`: updateconfig **replaces the whole config** — GET it, mutate one key, POST the full object back. `channelAutoDownload: true` is what actually enables the hourly `channelDownloadFrequency` sweep (the sweep checks channels AND playlists; if it is false nothing auto-downloads regardless of per-target settings).
Gotchas: sending a bare `{}` body to routes that validate a field returns 400 with a null body — always send the documented fields. Playlist subscription via MCP (`add_playlist`) creates the playlist with `auto_download=false`; you must call `/following` afterwards to enable it.

## Importing an existing yt-dlp library (youtube-media-capture migration, 2026-10-31)
How Youtarr tracks downloads (source research at v1.90.2, `server/modules/jobModule.js` `_repairFromArchive`,
`archiveModule.js`, `videosModule.js`, `filesystem/constants.js`):
- `config/complete.list` = the yt-dlp `--download-archive`, one `youtube <id>` per line. Appending ids there is what
  stops re-downloads; playlist/channel sweeps also treat archive-only ids as downloaded even without a DB row.
- `videos` + `channelvideos` DB rows are NOT created by the disk rescan (`videoRescanFrequency` only updates existing
  rows' paths/sizes). Rows for foreign files are created by the **`archiveBackfillFrequency` ("Repair library records")**
  task, which walks `complete.list` from the END, max 300 entries/run, and needs `jobs/info/<id>.info.json` per id
  (plain yt-dlp `--write-info-json` output works; the old capture job left those sidecars next to every video).
- Repair locates each file on disk with the same `scanForVideoFiles` walk, so **the hardlinked file just needs its
  name to end in `[<id>].<ext>` anywhere under `/usr/src/app/data`** (use the full Youtarr layout for tidiness).
  No DB insert by hand is needed — that is the whole trick.
Recipe (2026-10-31, verified: 46 archive ids -> 45 imported, 1 id had no info.json+mp4 pair so it was left alone):
1. Back up first: `mariadb-dump` of the `youtarr` DB (exec into the mariadb sidecar; write into `/data` via a scratch
   pod or pipe out) plus `config/complete.list` and `config/config.json` into `/app/config/backups/` (mode 700, uid 1000).
2. File work needs to see BOTH PVCs (old folders on jellyfin-media-hdd + the `youtarr` subPath); the youtarr container
   only sees the subPath. Use a short-lived scratch pod mounting `jellyfin-media-hdd` at /media and `youtarr-data` at
   /data. RunAsUser 1000 cannot hardlink the root-owned capture files (protected_hardlinks) — run it as root with
   CHOWN/FOWNER/DAC_OVERRIDE (PSA only warns; same pattern as the `perms` initContainer) and `chown -R 1000:1000`
   everything you create under `youtarr/`, plus any file you *copy* (jobs/info, backups). NEVER chown a hardlinked
   media file — it would chown the original inode too (links share the inode).
3. For each id in `/media/.youtube-capture-archive`: find its `<name>.info.json` anywhere under /media (excluding
   youtarr/) whose `.id` matches (the capture filenames contain NO id — only the info.json does), then
   `os.link(src, "/media/youtarr/<chan>/<chan> - <title> - <id>/<chan> - <title> [<id>].<ext>")`
   (san: strip `\\/*?:"<>|`+control, collapse spaces, strip ` .`; chan 80 bytes, title 64 bytes) and copy the info
   json to `jobs/info/<id>.info.json`. Same filesystem -> hard links cost zero extra space; fall back to copy only on EXDEV.
4. Append the ids (deduped) to `config/complete.list` (this adds them to the END, which is where the 300/run repair
   window reads; >300 ids just need several runs). Keep mode 664 uid 1000.
5. Trigger registration: `POST /api/schedules/archiveBackfillFrequency/run`, poll `GET /api/schedules` until
   lastFinishedRun says `Recovered N video records and M channel video records`.
6. `POST /api/maintenance/rescan-files` to confirm paths/sizes on disk, then subscribe the targets and call
   `/following` + `POST /api/playlists/:id/download` (see API section) — with the archive merged it queues only the
   genuinely-missing items (verified: `queued: 0` on a fully captured playlist).
7. Verify: `videos` table count = imported + native, `SUM(removed=1)` = 0, every `file_path` under
   `/usr/src/app/data/...`, playlist `downloaded_count`, `download_status` shows no new activity.
Notes: Jellyfin will show hardlinked videos twice (old folder + /media/youtarr) until the old capture folders are
trimmed out of its libraries — do not delete the media to fix that. There is NO per-channel/playlist upload-date
filter in Youtarr (channels: min/max duration + title regex; playlists: title regex + baseline) — an `after=DATE` from
the old queue format has no equivalent; rely on the archive baseline and note it when migrating such lines.

## Recovery
- Locked out: impossible while AUTH_PRESET is set; restarting re-applies the Secret's password.
- DB broken: scale to 0, restore the dump into a fresh `mysql/` subPath, scale up. Videos on disk are untouched.
- Videos table lost/emptied but files+`complete.list`+`jobs/info` intact: run `archiveBackfillFrequency` — it re-creates
  `videos`/`channelvideos` rows from those three artifacts (verified by the 2026-10-31 import; 300 rows/run cap, newest
  first). Pure orphans (files with no archive line) are NOT registered by any job; the rescan only updates existing rows.
