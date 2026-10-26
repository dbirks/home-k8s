---
name: open-webui
description: Operate the tailnet Open WebUI chat app (apps/openwebui.yaml, openwebui.beluga-wyvern.ts.net) - tsidp SSO login, the LLM Collective as its model backend, Collective qwen3-asr speech-to-text through the WAV shim, Collective phonon-31m text-to-speech, the RAM-only voice cache, upgrades, and how to verify each piece headlessly. Use when changing Open WebUI settings, upgrading its image, rotating its Collective key, or when a setting in the manifest seems to be ignored or dictation fails.
---

# Open WebUI (tailnet chat app)

One raw manifest, `apps/openwebui.yaml`: PVC (`openwebui-data`, SQLite + uploads),
Deployment `openwebui` (Recreate, 1 replica), Service, Tailscale Ingress
(`openwebui.beluga-wyvern.ts.net`), plus `openwebui-stt-shim` (ConfigMap + Deployment
+ Service). Not the Helm chart: it drags in ollama/pipelines/tika subcharts.

Secrets (SOPS, never plaintext):
- `apps/openwebui-secrets.enc.yaml`: `WEBUI_SECRET_KEY`, `OAUTH_CLIENT_SECRET` (and a
  stale `OPENAI_API_KEY` that nothing reads any more).
- `apps/openwebui-llmc.enc.yaml`: `LLMC_API_KEY`, an LLM Collective key (minted under
  **Keys** in the gateway UI; shown once).

## Settings in the manifest are ignored? ENABLE_PERSISTENT_CONFIG

Open WebUI copies env vars into its DB `config` table on first boot and from then on
the DB wins: changing `OPENAI_API_BASE_URL` in the manifest silently did nothing. The
manifest sets `ENABLE_PERSISTENT_CONFIG=false`, so env wins on every boot and Admin
Panel edits are lost on restart. Make changes in the manifest. (OAuth settings are
already env-only via `ENABLE_OAUTH_PERSISTENT_CONFIG=false`.)

## Backend: the LLM Collective

- Base `https://llm-collective.e-gineering.com/v1`, header `Authorization: Bearer KEY`.
  `/v1/models` lists `auto` plus each contributed model (bare alias and `owner/alias`).
- `AUDIO_STT_OPENAI_API_KEY` must be set explicitly: Open WebUI only inherits
  `OPENAI_API_KEY` into the STT key when the base URL is `api.openai.com`.
- Rotate the key without printing it (sops isn't installed system-wide; a release
  binary works, encryption needs only the public age recipient from `.sops.yaml`):
  ```bash
  kubectl create secret generic openwebui-llmc -n default \
    --from-literal=LLMC_API_KEY="$LLMC_API_KEY" --dry-run=client -o yaml \
  | sops encrypt --filename-override apps/openwebui-llmc.enc.yaml \
      --input-type yaml --output-type yaml /dev/stdin > apps/openwebui-llmc.enc.yaml
  ```
  `--filename-override` is what makes sops pick the `.sops.yaml` rule when reading stdin.
  Decrypting existing files needs the age private key (`~/.config/sops/age/keys.txt`),
  which may not be on the workstation; you can still re-create a secret from scratch.

## Speech-to-text: why there is a shim

The Collective's `qwen3-asr-1-7b` (speech.cpp) reads **WAV only** (16/24/32-bit PCM or
f32) and caps uploads at 25 MB; anything else gets 400 "is not a WAV file".
Open WebUI never sends WAV: browser dictation is webm/opus, the call overlay is
webm labelled `recording.wav`, and its pydub step converts every non-flac/mp3 codec
(including `pcm_s16le`!) to mp3. So `openwebui-stt-shim` sits between them:

- Any ffmpeg-readable input becomes 16 kHz mono `pcm_s16le` WAV, cut into
  `SEGMENT_SECONDS=600` pieces (about 19 MB each), sent one by one, texts joined.
- It forwards the caller's own `Authorization` header and holds no key.
- It runs on the Open WebUI image (python3 + aiohttp + ffmpeg are already there), so
  bump its image digest together with the main one.
- It listens on `SHIM_PORT`, NOT `PORT`: the image bakes in `PORT=8080`, which once made
  the shim listen on 8080 while its probes hit 8000 (pod Running, never Ready).
- Editing only the ConfigMap script does not restart the pod; change something in the
  pod template too (or the shim keeps running the old code).
- `BYPASS_PYDUB_PREPROCESSING=true` stops Open WebUI's lossy mp3 pass first.
- If the Collective starts accepting other formats, point
  `AUDIO_STT_OPENAI_API_BASE_URL` straight at it and delete the shim.

Test the shim from the workstation (key from env, never echoed):
```bash
kubectl port-forward svc/openwebui-stt-shim 18000:80 &
ffmpeg -loglevel error -i /usr/share/sounds/alsa/Front_Center.wav -c:a libopus /tmp/x.webm
curl -s -H "Authorization: Bearer $LLMC_API_KEY" -F model=qwen3-asr-1-7b \
  -F file=@/tmp/x.webm http://127.0.0.1:18000/v1/audio/transcriptions   # {"text":"Front, center."}
```

## Text-to-speech: Collective phonon-31m (no shim)

Read aloud and voice calls use `AUDIO_TTS_ENGINE=openai`, with the Collective as the base URL, the
same `openwebui-llmc` key (it must be set explicitly, as for STT), `AUDIO_TTS_MODEL=phonon-31m`
and `AUDIO_TTS_VOICE=Harper`. The engine is Gradium Phonon on CPU; see the phonon-tts skill.
- Open WebUI posts `{model, input, voice}` and gets mp3 back, which is Phonon's default, so the
  formats already match.
- The voice list is OpenAI's six names. Open WebUI GETs `<base>/audio/voices` without the key, the
  Collective answers 401 (logged as an ERROR, which is harmless), and it falls back to
  alloy/echo/fable/onyx/nova/shimmer. The Phonon server aliases each of these to a catalog voice.
  An unaliased name returns a 400.
- `speed` must be 1.0: Phonon has no rate control. Open WebUI's frontend does not send it.
- The call overlay fires one request per sentence without waiting for the previous one. A 429
  (Collective concurrency is 3) shows as a toast and that sentence is skipped.

## Zero data retention for voice: the audio cache is RAM

Open WebUI writes every TTS result (the mp3 plus a JSON holding the spoken **text**) to
`cache/audio/speech/` and every dictation upload plus its transcript to `cache/audio/transcriptions/`,
and never deletes them. `/app/backend/data/cache/audio` is an `emptyDir` with `medium: Memory`
(256Mi), and the `audio-cache-sweeper` sidecar deletes files older than 10 minutes. Nothing from voice
reaches the PVC or survives a restart. Files written before 2026-10-26 still sit on the PVC, hidden
under that mount.
Check it:
`kubectl exec deploy/openwebui -c openwebui -- grep cache/audio /proc/mounts` should show `tmpfs`.

Verify TTS headlessly with the session JWT (see below):
```bash
curl -s -H "Authorization: Bearer $TOK" https://openwebui.beluga-wyvern.ts.net/api/v1/audio/config | jq '.tts|{ENGINE,MODEL,VOICE}'
curl -s -H "Authorization: Bearer $TOK" -H 'Content-Type: application/json' \
  -d '{"input":"Voice check.","voice":"nova"}' -o t.mp3 https://openwebui.beluga-wyvern.ts.net/api/v1/audio/speech
ffprobe -v error -show_entries format=format_name,duration -of csv=p=0 t.mp3   # mp3,1.2...
```

## SSO (tsidp)

Client `openwebui` is registered in tsidp (see the tsidp-oidc-client skill for
registration, hostAliases to `idp-egress`, and claims). Password login is off; the
**first** OAuth login becomes admin. tsidp sends no `name` claim, so
`OAUTH_USERNAME_CLAIM=preferred_username`. If the idp-egress Service is recreated,
update the `hostAliases` IP.

Quick checks:
```bash
curl -s https://openwebui.beluga-wyvern.ts.net/api/config | jq '{version, oauth}'
curl -s -o /dev/null -w '%{redirect_url}\n' https://openwebui.beluga-wyvern.ts.net/oauth/oidc/login
# expect https://idp.beluga-wyvern.ts.net/authorize?...client_id=0f777b7f...
```

## Logged-in checks with an API key

Log in with Playwright (tsidp auto-approves the workstation; the system has no Chrome,
so pass `--browser=chromium`) and use the session JWT in `localStorage.token`:
```bash
P="pnpm dlx @playwright/cli -s=owui"
$P open --browser=chromium https://openwebui.beluga-wyvern.ts.net/oauth/oidc/login
$P --raw eval "async () => JSON.stringify((await (await fetch('/api/models',{headers:{Authorization:'Bearer '+localStorage.token}})).json()).data.map(m=>m.id))"
TOK=$($P --raw eval "localStorage.token" | grep -v '^Done' | tr -d '"')   # don't echo it
curl -s -H "Authorization: Bearer $TOK" -F "file=@/tmp/x.webm;type=audio/webm" \
  https://openwebui.beluga-wyvern.ts.net/api/v1/audio/transcriptions        # full STT path
$P close
```
Chat: POST `/api/chat/completions` with `{"model":"auto","stream":false,"messages":[...]}`.

## Upgrading

- Pin tag + index digest:
  `T=$(curl -s "https://ghcr.io/token?scope=repository:open-webui/open-webui:pull" | jq -r .token); curl -sI -H "Authorization: Bearer $T" -H "Accept: application/vnd.oci.image.index.v1+json" https://ghcr.io/v2/open-webui/open-webui/manifests/vX.Y.Z | grep -i digest`
- Read release notes for "Database Migrations"; there is no downgrade across them
  without restoring `webui.db`. Back it up first if there are real chats:
  `kubectl exec deploy/openwebui -- python3 -c "import sqlite3; sqlite3.connect('/app/backend/data/webui.db').backup(sqlite3.connect('/app/backend/data/webui.db.bak'))"`
- Env var names drift between releases: grep `backend/open_webui/config.py` and
  `routers/audio/__init__.py` at the target tag instead of trusting old docs.
- The first pull of a new tag takes several minutes; Recreate means a short outage.
