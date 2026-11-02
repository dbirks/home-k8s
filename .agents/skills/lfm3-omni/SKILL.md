---
name: lfm3-omni
description: Operate LFM3-Omni-2B speech-to-speech (apps/lfm3-omni.yaml, issue 167) - llama.cpp llama-server Vulkan on dave-droid's GTX 1070, published on the LLM Collective realtime WebSocket door as lfm3-omni-2b. Use when testing or debugging the realtime voice model, upgrading llama.cpp or the GGUFs, checking co-residency with qwen3-asr on the 8 GB Pascal card, or rolling it back.
---

# LFM3-Omni-2B realtime voice on dave-droid

What runs: `ghcr.io/ggml-org/llama.cpp:server-vulkan-b12030` (pinned digest) serving
`LFM3-Omni-2B-Q4_K_M.gguf` + `mmproj-LFM3-Omni-2B-Q8_0.gguf` from the PVC `lfm3-omni-models`
(local-path on dave-droid), HF revision pinned in the init container, sha256-checked, downloaded once.
ClusterIP `lfm3-omni:8080`, no auth. Local model id = `--alias lfm3-omni-2b`.

Collective: backend `lfm3-omni` in BOTH `apps/llm-collective-contributor.yaml` and `-dev.yaml`,
`types: [realtime]`, concurrency 1 each. Public ids: `lfm3-omni-2b` and `david-birks/lfm3-omni-2b`.

## Why these choices
- **Vulkan, not CUDA**: GTX 1070 is sm_61; CUDA 13 dropped Pascal and the upstream `server-cuda`
  images can't be assumed to carry sm_61. The Vulkan build works out of the box with the nvidia CDI
  ICD (same env as qwen3-asr: `NVIDIA_DRIVER_CAPABILITIES=all`, `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`).
  `ERROR: init 250 result=11` at the top of the log is HAMi's CUDA hook failing to init under a
  Vulkan-only process; harmless.
- **mmproj Q8_0** with Q4_K_M weights: the mmproj holds the audio encoder and the speech decoder;
  Liquid's own llama-tts example pairs it this way. +215 MB.
- `--reasoning off` is required (non-reasoning model). `--temp 0` = greedy text; audio sampling
  (temp 0.4, rep-pen 1.2) comes from the GGUF metadata.
- Versioned image tags: not every `bNNNNN` gets a `server-vulkan-bNNNNN` tag (b12031/b12032 were
  404); probe ghcr before bumping:
  `curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOK" -H 'Accept: application/vnd.oci.image.index.v1+json' https://ghcr.io/v2/ggml-org/llama.cpp/manifests/server-vulkan-bNNNNN`
  (anonymous token from `https://ghcr.io/token?scope=repository:ggml-org/llama.cpp:pull`). Needs >= b11960.

## Measured (2026-11-02, b12030)
- VRAM (nvidia-smi on the card, via `kubectl exec gpu-vram-exporter-<pod on dave-droid> -- nvidia-smi`):
  llama-server ~2.5 GiB, qwen3-asr ~2.3 GiB, card total **4.86 GiB / 8 GiB peak** during a realtime
  response with a concurrent ASR request. HAMi gpumem 4000 is accounting only (Vulkan bypasses it).
- Realtime: input rate **16 kHz** PCM16 mono, output **24 kHz** PCM16 mono (read both from `session.created`).
  Warm time-to-first-audio **~0.3 s** after commit (0.27-0.35 s direct and via the gateway); the first
  request after a pod start took ~3.4 s. ~10 s of reply audio generated in ~4 s.
- Text: ~104 tok/s on /v1/chat/completions (works, but NOT advertised as `text` in the Collective).
- Co-residency: qwen3-asr transcribed a 4.5 s clip in 520 ms during a realtime response vs 479 ms
  alone, word-perfect. The model's spoken replies transcribe back word-for-word (intelligible).
- Quality caveat: a 2B model - it states wrong "fun facts" with confidence. Fine for a voice demo/agent
  front end, not for facts.

## Test it
```bash
kubectl port-forward svc/lfm3-omni 18080:8080 &
curl -s localhost:18080/health; curl -s localhost:18080/v1/models
curl -s localhost:18080/props | jq '{modalities, model_alias, build_info}'
```
Realtime client (python, `uv run --with websockets`): connect `ws://localhost:18080/v1/realtime?model=lfm3-omni-2b`,
read `session.created`, send `session.update` with `{"type":"realtime","instructions":...,"output_modalities":["text","audio"]}`,
stream `input_audio_buffer.append` (base64 PCM16 at the input rate, 100 ms chunks), then
`input_audio_buffer.commit` + `response.create`; collect `response.output_audio.delta` until `response.done`.
Test speech: phonon-tts (`POST /v1/audio/speech` on svc/phonon-tts:8880, 24 kHz WAV) - its streamed WAV
header has no length, so re-mux with `ffmpeg -i in.wav -ar 16000 -ac 1 out.wav` before sending it to qwen3-asr.

Through the Collective: same client against `wss://llm-collective.e-gineering.com/v1/realtime?model=lfm3-omni-2b`
with `Authorization: Bearer <your Collective API key>` on the handshake (server/desktop clients only:
browsers can't set that header, llm-collective#81). Expect `101` and
`X-LLM-Collective-Model: david-birks/lfm3-omni-2b`. A second session while one is open gets 429
(the WS holds the contributor slot for its whole life); it's released on close.

## Gotchas
- One llama-server slot is shared by prod and dev contributors; a long-lived WS on either blocks the other.
  Raise `--parallel` and the contributor concurrency together only after re-measuring VRAM.
- Dev gateway not verified end to end (needs a dev-gateway key); the dev contributor does report the
  backend healthy.

## Rollback
`replicas: 0` in apps/lfm3-omni.yaml and delete the `lfm3-omni` backend block from both contributor
ConfigMaps (reloader rolls the contributors). qwen3-asr is independent and unaffected.
