---
name: local-audio-transcription
description: Local speech-to-text on this cluster - what's already deployed (granite-stt), how to point clients at it, and the 2026 open ASR model landscape (Qwen3-ASR, Whisper, Voxtral, Parakeet, SenseVoice) with vLLM support and benchmark numbers. Use when asked to transcribe audio locally, to replace GPT-4o-mini-transcribe, or when adding/choosing an ASR model.
---

# Local audio transcription

## What's already here (zero-effort path)

`granite-stt` is a vLLM deployment of `ibm-granite/granite-speech-4.1-2b` serving the
OpenAI-compatible `/v1/audio/transcriptions` endpoint:

- `apps/granite-stt.yaml` - Deployment + `granite-stt-models` PVC (model is pre-cached there)
  + Service on 8001. Image `vllm/vllm-openai:v0.20.1`, `--gpu-memory-utilization=0.10`,
  `HF_TOKEN` from the `hf-token` secret. It was live 2026-07-20 and parked at
  `replicas: 0` on 2026-07-23 for the issue #57 burst - **bring it back with
  `replicas: 1`** when STT is wanted again (it has not been restored since).
- `apps/granite-stt-tailscale-ingress.yaml` - tailnet-only endpoint
  `https://stt.beluga-wyvern.ts.net` (IngressClass `tailscale`).
- Client usage (from anywhere on the tailnet; any OpenAI SDK works):
  ```bash
  curl -F "file=@audio.mp3" -F "model=granite-speech" \
    https://stt.beluga-wyvern.ts.net/v1/audio/transcriptions
  ```
  Python: `OpenAI(base_url="https://stt.beluga-wyvern.ts.net/v1", api_key="any")` then
  `.audio.transcriptions.create(model="granite-speech", file=...)`.

Granite Speech 4.1 2B (released 2026-04-29, Apache 2.0): ASR for en/fr/de/es/pt/ja with
punctuation + truecasing and keyword-list biasing (prompt "transcribe the speech to text.
Keywords: ..."); speech translation to/from English incl. it/zh. Variants:
`granite-speech-4.1-2b-plus` (speaker-attributed + word timestamps) and `-nar` (faster).
No Mandarin ASR. llama.cpp can also run its GGUF with `/v1/audio/transcriptions` (CPU path).

## 2026 model landscape (vLLM-native STT, verified from vLLM docs 2026-09)

vLLM has first-class STT: `/v1/audio/transcriptions`, `/v1/audio/translations`, and a
`/v1/realtime` WebSocket for streaming (docs: `serving/online_serving/speech_to_text`).
Supported transcription architectures (examples):

| Model | HF repo | Notes |
|---|---|---|
| Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B`, `-0.6B` | 52 langs + 22 zh dialects, LID, offline+streaming; 0.6B does realtime via `--hf-overrides '{"architectures":["Qwen3ASRRealtimeGeneration"]}'`; `Qwen3-ForcedAligner-0.6B` adds word/char timestamps (11 langs) |
| Whisper | `openai/whisper-large-v3-turbo`, `-large-v3`, ... | 99 langs; the thing OpenAI's own transcribe models are built from |
| Granite Speech (+Plus) | `ibm-granite/granite-speech-4.1-2b[-plus]` | see above; keyword biasing |
| Voxtral | `mistralai/Voxtral-Mini-3B-2507`, `-Small-24B-2507`; realtime `Voxtral-Mini-4B-Realtime-2602` | needs `mistral-common[audio]` |
| GLM-ASR | `zai-org/GLM-ASR-Nano-2512` | |
| FunASR | `FunAudioLLM/Fun-ASR-Nano-2512-vllm` (MLT-Nano covers 31 langs) | |
| MOSS-Transcribe-Diarize | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | transcription + speaker diarization |
| Cohere-Transcribe | `CohereLabs/cohere-transcribe-03-2026` | |
| Qwen3-Omni, FireRedASR2, Gemma3n | (see vLLM supported-models "Transcription" table) | |

## Benchmarks (vendor-reported WER, lower=better)

Qwen3-ASR-1.7B (arXiv 2601.21337, Jan 2026) vs the proprietary models it's usually
compared against:

| Set | GPT-4o-Transcribe | Whisper-large-v3 | Qwen3-ASR-1.7B |
|---|---|---|---|
| LibriSpeech clean/other | **1.39** / 3.75 | 1.51 / 3.97 | 1.63 / **3.38** |
| GigaSpeech | 25.50 | 9.76 | **8.45** |
| CommonVoice-en | 9.08 | 9.90 | **7.39** |
| Accented English | 28.56 | 21.30 | **16.07** |
| Fleurs multilingual (avg) | - | 5.27 | **4.90** |

Qwen3-ASR-1.7B is SOTA among open ASR models and competitive with the strongest
proprietary APIs (strongly beats gpt-4o-MINI-transcribe, the weaker sibling). Caveat:
these are the vendor's numbers; gpt-4o-mini-transcribe itself is Whisper-based, so
whisper-large-v3-turbo is the closest "same architecture" local analog.

Outside vLLM:
- Parakeet TDT 1.1B (NVIDIA NeMo) - top pure-English WER class, very fast; NeMo-only runtime, English-focused. (HF model card blocked this fetcher with 401; re-verify numbers.)
- faster-whisper / WhisperX (CTranslate2 + VAD + forced alignment) - CPU or GPU, OpenAI-compatible server.
- whisper.cpp - CPU-only, OpenAI-compatible server.
- SenseVoice (FunASR 1.4.14) - zh/yue/en/ja/ko + emotion/event tags, non-AR so 5-15x faster than Whisper at similar size, OpenAI-compatible service.

## Deployment pattern

Copy `apps/granite-stt.yaml` and change: model ref, `served-model-name`, PVC (each vLLM
pod wants its own RWO local-path PVC, model pre-downloaded or let it pull with
`HF_TOKEN` - the `hf-token` secret already exists), service/ingress names. Keep
`enableServiceLinks: false`, `runtimeClassName: nvidia`, small `--gpu-memory-utilization`
(0.10 is plenty for a <=2B model), probes on `/health`.

- vLLM version matters for new architectures: `Qwen3ASRForConditionalGeneration` needs a
  recent vLLM (day-0 support, 2026); the pinned `v0.20.1` predates it. Use the same
  vLLM image the KServe LLM engines run.
- Node constraint is host RAM (46GB, no swap), not VRAM - a 1-2B ASR pod adds only a few
  GB. The ~2-model co-residence limit in AGENTS.md is about 27B+ LLM engines.

## OpenAI API surface (what local is replacing)

2026: `gpt-4o-mini-transcribe` is legacy; OpenAI's current file-transcription model is
`gpt-transcribe` (accepts `prompt` + `keywords[]` + `languages[]`). Diarization is a
separate model (`gpt-4o-transcribe-diarize`); word timestamps only via `whisper-1`;
files <= 25 MB; input formats mp3/mp4/mpeg/mpga/m4a/wav/webm. Local equivalents: keyword
biasing (Granite), word/char timestamps (Qwen3-ForcedAligner, Granite Plus, WhisperX),
diarization (MOSS-Transcribe-Diarize, FunASR CAM++ pipeline).

## Qwen3-ASR via the Collective: queueing (2026-10-22)

Live ASR is `apps/qwen3-asr.yaml` (speech.cpp on dave-droid's GTX 1070), Collective backend `qwen3-asr`.
The contributor never queues: past its `concurrency` it answers 429 + `Retry-After: 1`, and the gateway
passes that to the client. speech.cpp `serve` DOES queue: it runs one transcription at a time and the rest
wait (log line `waited N s`; order is roughly, not strictly, FIFO; `/health` stays instant). So the
backend's `concurrency` is the queue depth: 8 in both contributors (prod + dev, so up to 16 can wait).
~17x realtime on a 114 s clip (6.8 s); the 25 MB cap is ~13 min of 16 kHz audio, ~48 s each.
Test: port-forward `svc/qwen3-asr`, fire N parallel `curl -F file=@x.wav -F model=Qwen3-ASR-1.7B
.../v1/audio/transcriptions`, all should return 200. Don't `pkill -f 'port-forward ...'` from the same
shell: the pattern matches your own command line and kills it (exit 144).
