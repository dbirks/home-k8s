---
name: qwen3-tts-serving
description: Run and size the Qwen3-TTS 1.7B Base and VoiceDesign servers (vLLM-Omni, apps/qwen3-tts.yaml) inside small HAMi slices, and use the design-a-voice then register-on-Base workflow with its persistent voice registry. Use when a TTS pod OOMs or will not fit, when tuning its gpu_memory_utilization, when enrolling or deleting a voice, or when exposing TTS through the LLM Collective.
---

# Qwen3-TTS on vLLM-Omni under HAMi

## Where things live
- Manifest: `apps/qwen3-tts.yaml` (ConfigMap `qwen3-tts-deploy` = trimmed copy of upstream
  `vllm_omni/deploy/qwen3_tts.yaml`, two Deployments + Services, port 8000).
- Weights: pinned HF snapshots on the HDD PVC `qwen3-tts-models-hdd` (apps/qwen3-tts-models.yaml);
  loaded by path with `HF_HUB_OFFLINE=1`. Image `vllm/vllm-omni:v0.30.0` pinned by digest (no
  ENTRYPOINT: the command must start with `vllm serve`).
- Voice registry: PVC `qwen3-tts-voices` mounted at `SPEAKER_SAMPLES_DIR=/voices` on Base only.

## Sizing (measured 2026-10-16/17)
- HAMi makes vLLM see the SLICE as the whole card, so `gpu_memory_utilization` is a fraction of the
  slice. Upstream's 0.3 + 0.3 (H100) cannot hold the 3.9 GiB talker in a 7 GB slice.
- Stage 0 (talker): 3.99 GiB weights+non-torch, 0.16 activation peak, 0.04 graphs, KV 112 KiB/token.
  One 4,096-token request needs 448 MiB of KV (vLLM refuses to start below max_model_len).
- Stage 1 (code2wav) builds NO KV pool (0.29 GiB weights + graphs); its utilization knob is inert.
- Pod total = talker reservation + ~2,500 MiB fixed (stage 1, two CUDA contexts, API process).
- Result: **8000 MiB slice, talker 0.62** (4,960 MiB, 0.76 GiB KV = ~7,100 tokens). Floor ~7,240;
  7000 does not fit a 1.7B server. Read vLLM's `Free memory on device ... Actual usage is ...` line
  after any change; it prints the exact `--kv-cache-memory` that would fit.
- Startup ~7 min (HDD load ~68 s + graph capture ~57 s per stage).

## Reusable voice workflow
1. VoiceDesign: `POST /v1/audio/speech` `{"model":"qwen3-tts-voicedesign","task_type":"VoiceDesign",
   "input":TEXT,"instructions":DESCRIPTION,"language":"English","response_format":"wav"}`.
2. Register on BASE: `POST /v1/audio/voices` multipart `audio_sample=@clip.wav`, `name`, `consent`
   (required), `ref_text` (the exact TEXT, improves cloning), optional `speaker_description`.
3. Base `POST /v1/audio/speech` `{"model":"qwen3-tts-base","task_type":"Base","voice":NAME,...}`.
4. `GET /v1/audio/voices` lists; `DELETE /v1/audio/voices/NAME` removes. Persists across restarts
   (verified). `VLLM_OMNI_SPEAKER_REGISTRATION_POLICY=immutable`: re-registering a name returns 400.
Test voice `home-test-narrator` (synthetic, from VoiceDesign) was left registered as a canary.

## LLM Collective
Backends with `types: [speech]`, `protocols: [openai]` (contributor >= v0.9). The gateway rewrites only
`model`, so Qwen fields pass through. `/v1/audio/voices` is NOT routed (good: the registry has no
per-user ownership; any caller can use any registered name). Do not enroll real people's voices until
gateway-side ownership/consent exists.

## Testing from the cluster
`kubectl run t --rm -i --restart=Never --image=curlimages/curl:8.11.1 -- sh -c 'curl ... -o /tmp/a.wav'`;
check `RIFF....WAVE` header and byte count (~48 KB per second of 24 kHz 16-bit mono).
