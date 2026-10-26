---
name: phonon-tts
description: Run Gradium Phonon TTS (phonon-31m-en, q8, CPU only) on dave-droid behind the LLM Collective as phonon-31m-en, and as Open WebUI's read-aloud and voice-call voice. Covers offline asset staging, licences, voices and aliases, concurrency and CPU sizing, measuring latency and RTF, and upgrading. Use when changing apps/phonon-tts.yaml, adding voices, tuning its concurrency, debugging 429s, 400s or 502s on /v1/audio/speech, or swapping TTS engines.
---

# Phonon TTS (CPU) on dave-droid

## Where things live
- `apps/phonon-tts.yaml`: PVC `phonon-tts-assets` (local-path, binds on dave-droid), Deployment
  `phonon-tts` (the `fetch-assets` init container plus the server), ClusterIP Service `phonon-tts:8880`.
  It has no GPU and no `runtimeClassName`, but it still needs the `dedicated=pascal` toleration and
  `nodeSelector kubernetes.io/hostname: dave-droid`.
- Image: `ghcr.io/gradium-ai/ptts-openai-server:<ver>@sha256:<index digest>` (Rust server, official).
  The image holds no weights. At startup the wrapper picks the x86-64-v3 (AVX2) build; dave-droid's
  i7-6700K has AVX2.
- Collective backend `phonon` (alias `phonon-31m-en`, `types: [speech]`) is set in both
  `apps/llm-collective-contributor.yaml` (concurrency 3) and `-dev.yaml` (concurrency 1). The two
  add up to the server's `PTTS_MAX_CONCURRENT_REQUESTS=4`.
- Open WebUI calls the Collective directly (no shim): `AUDIO_TTS_MODEL=phonon-31m-en`,
  `AUDIO_TTS_VOICE=Harper`. See the open-webui skill.

## Licence gate (do not skip)
Only **phonon-31m-en is Apache-2.0**. Every 40m checkpoint (en/fr/de/es/pt/multilingual) is
**CC BY-NC 4.0**, so none of them may be served to the Collective. The voice catalog
`gradium/phonon-voices` is Apache-2.0.

## Offline / zero data retention
- The init container downloads the checkpoint at a pinned HF revision, using only the 5 files the q8
  server reads: `config.json`, `tokenizer.json`, `tokenizer.model`, `model.q8.gguf`,
  `default-voice.safetensors`. `model.safetensors` and `q4k` are not needed. It also downloads the
  English catalog voices: `voices.jsonl` is filtered to `"language": "en"` (37 voices), and every file
  is sha256-checked. A marker file `.verified-<mrev>-<vrev>` skips all of this on later starts.
- Server env: `PTTS_MODEL` and `PTTS_VOICE_CATALOG` are set to local paths, plus `HF_HUB_OFFLINE=1`.
  The root filesystem and the assets are read-only. Without a local `PTTS_VOICE_CATALOG` the server
  downloads `hf://gradium/phonon-voices@<rev>` on every cold start.
- Proven offline: the same binary served from those 5 files inside `unshare -rn` (no network at all).
  To repeat that, extract the binary from the image's last layer (see below).
- The server logs only startup lines and never request text. Flannel does not enforce NetworkPolicy,
  so offline-ness comes from configuration, not the network.
- **No voice cloning.** The speaker encoder is not released, so there is no local enrollment. Gradium's
  cloud cloning is a privacy regression and is deliberately not wired up. Catalog voices only.

## API facts (0.4.x)
- `POST /v1/audio/speech` takes `{model (ignored), input up to 4096 chars, voice, response_format}`.
  Formats: `mp3` (default), `opus`, `wav` (streamed, no length in the header), `pcm` (s16le 24 kHz).
  `aac`/`flac` return 400. `speed` other than 1.0 returns 400 (`PTTS_IGNORE_SPEED=true` would accept
  and ignore it). `instructions` is ignored.
- Voices: `GET /v1/audio/voices` on the pod. The Collective does not proxy it.
  Names are case-insensitive, e.g. Harper, Miles, Juniper, Everly, Easton, Toby, plus 12 character
  voices.
- Aliases: `PTTS_VOICE_ALIAS=alloy=Harper,echo=Easton,fable=Toby,onyx=Miles,nova=Juniper,shimmer=Everly`.
  `PTTS_OPENAI_VOICES=Harper` maps the other OpenAI names (ash, coral, marin, ...). An unknown name
  returns 400. Open WebUI needs these aliases: its voice list GETs `<base>/audio/voices` without the
  key, gets a 401 from the Collective (it logs this as an ERROR, which is harmless), and falls back to
  OpenAI's six names.
- Overload: past `PTTS_MAX_CONCURRENT_REQUESTS` the server returns an immediate 429 `server_busy`, with
  no queue. The contributor does the same past its `concurrency`.

## Sizing (measured 2026-10-26 on dave-droid, i7-6700K 4c/8t)
| load | result |
|---|---|
| 1 short (3.4 s audio) | 0.40-0.58 s total, first audio about 0.2 s; through the gateway 0.56-0.64 s |
| 1 long (33 s audio) | 3.4-3.9 s, RTF about 0.11 (about 9x real time), about 15 CPU-s |
| 2 / 3 / 4 concurrent long | 6.3 / 9.8 / 15 s each, all faster than real time |
| 4 concurrent short | 1.45-1.6 s each |
| RSS | 139 MiB idle, about 490 MiB peak for 1 request, 769 MiB peak at 4 concurrent |

- `RAYON_NUM_THREADS=4` matters. The default is 8 threads, and under the 6-CPU limit it throttled:
  2 concurrent long requests took 8.2 s each, against 6.3 s with 4 threads. Single-request speed is
  the same either way.
- Open WebUI's call overlay fires one TTS request per streamed sentence without waiting for the
  previous one. A burst then gets a 429, which shows as a toast and a skipped sentence. Keep at least
  3 slots for prod.

## Test it
```bash
kubectl port-forward svc/phonon-tts 18881:8880 &   # note $!; kill that PID afterwards, never pkill -f
curl -s localhost:18881/v1/audio/voices | jq -c .aliases
curl -s localhost:18881/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Hello from Phonon.","voice":"Harper"}' -o x.mp3
# through the gateway (key from env, never echoed):
curl -s https://llm-collective.e-gineering.com/v1/audio/speech -H "Authorization: Bearer $LLMC_API_KEY" \
  -H 'Content-Type: application/json' -d '{"model":"phonon-31m-en","input":"Hi.","voice":"nova"}' -o y.mp3
# intelligibility: ffmpeg -i y.mp3 -ar 16000 -ac 1 y.wav, then POST it to qwen3-asr-1-7b /v1/audio/transcriptions
```
- To measure time to first audio, request `response_format: pcm` and time the arrival of the first
  4800 bytes (0.1 s of audio). For CPU-seconds, take `/proc/1/stat` fields 14+15 before and after
  (`kubectl exec deploy/phonon-tts -c phonon -- ...`; the image is Debian, so `sh` and `awk` exist).
- When you run parallel curls next to a backgrounded port-forward, use `wait "${pids[@]}"` and not a
  bare `wait`. A bare `wait` also waits on the port-forward, so the command never returns.

## Upgrading
- Releases: `gh release list -R gradium-ai/xn-ptts`. GHCR tags are bare semver (`0.4.2`, no `v`).
  To get the index digest:
  `T=$(curl -s "https://ghcr.io/token?scope=repository:gradium-ai/ptts-openai-server:pull" | jq -r .token); curl -sI -H "Authorization: Bearer $T" -H "Accept: application/vnd.oci.image.index.v1+json" https://ghcr.io/v2/gradium-ai/ptts-openai-server/manifests/X.Y.Z | grep -i digest`
- Read `ptts-openai-server/README.md` at the new tag for env and flag changes. `--help` lists the
  default catalog revision it pins (`--voice-catalog` default); keep `VREV` in the init script in step
  with it.
- New model or voice revision: take the LFS sha256 from
  `https://huggingface.co/api/models/gradium/<repo>/revision/<sha>?blobs=true`, hash the small files
  yourself, update the init script's revisions and hashes, and the marker name will change.
- No docker access on the workstation. To test a binary, pull the last image layer with curl
  (`/v2/<repo>/blobs/<digest>`), untar it, and run
  `usr/local/libexec/ptts/ptts-openai-server-x86-64-v3` (it needs `libmp3lame` and `libopus`, which
  Arch has).
