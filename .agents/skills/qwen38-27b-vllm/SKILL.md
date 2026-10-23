---
name: qwen38-27b-vllm
description: Operate the primary LLM - Qwen3.8-27B (nvidia NVFP4) on stock upstream vLLM (apps/qwen38-27b-vllm.yaml), Collective model qwen3-8-27b - flags and why, reasoning levels (what works, what 400s, default), pi config, memory sizing, how to test quality/speed/prefix caching, and the switch/rollback with flint. Use when changing, upgrading, resizing, debugging or rolling back the main coding model.
---

# Qwen3.8-27B on stock vLLM (live since 2026-10-23)

Chosen over Flint (needed our own vLLM patches) and NInfer (our fork; one prefill at a time at ~3K tok/s,
alternating with decode, so one long prompt dragged every agent). Stock image, no patches:
`vllm/vllm-openai:v0.32.0` (digest-pinned). Weights `nvidia/Qwen3.8-27B-NVFP4` @ 482ca0f3 on the HDD PVC
`qwen38-27b-models-hdd` (apps/qwen38-27b-models.yaml: pinned, sha-checked download Job + image pre-pull Job).

## Why this checkpoint
NVIDIA ModelOpt, Local-Hessian calibrated: NVFP4 MLPs + lm_head, FP8 attention and linear-attention, BF16
MTP head and vision tower. Published GPQA-D 88.92 -> 88.01, Terminal-Bench 75.56 -> 74.02, AA-LCR 72.63 ->
73.38 vs BF16. `dbirks/Qwen3.8-27B-W4A16-AutoRound` matches BF16 on its own evals but is weight-only int4
(Marlin, BF16 math), so prefill is slower on Blackwell; it is the pick for non-Blackwell cards.
`unsloth/...-NVFP4` has no published evals.

## Flags (see the manifest comments)
Recipe recipes.vllm.ai/Qwen/Qwen3.8-27B + NVIDIA's card: `--kv-cache-dtype fp8_e4m3`,
`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` (4 crashed 0.27.1),
`--tool-call-parser qwen3_coder --reasoning-parser qwen3`, `--enable-prefix-caching` (hybrid model: vLLM
picks mamba cache mode `align`), `--max-num-seqs 24`, `--max-num-batched-tokens 8192`,
`--gpu-memory-utilization 0.90` of the 56800 MiB HAMi slice -> weights 20.7 GiB, KV 26.26 GiB =
741,032 tokens (2.83x full 262K; KV only on the 16 full-attention layers of 64, 32 KiB/token fp8).
`--default-chat-template-kwargs '{"reasoning_effort":"medium"}'` sets the server default.
Startup ~7 min (2.5 min weights off the HDD, then compile + FlashInfer autotune + graphs). The autotuner logs
a few `HAMI-core ERROR ... OOM` + "falling back to default tactic" for the biggest fp4_gemm shapes: harmless
(3 of ~106 profiles); more headroom (lower utilization) would let them tune.
The log line "Disabled mm_prefix attention mode because multimodal inputs are configuration-disabled" is
NOT vision being off: a data-URI image is read correctly.

## Reasoning levels (probed live, scripts/reasoning_probe.py)
Qwen's template accepts `reasoning_effort` xhigh (Qwen's default), medium, low ONLY.
- top-level `reasoning_effort` low/medium/xhigh: thinks. `none`: thinking off (vLLM turns it into
  enable_thinking=false). high/minimal/max: HTTP 400 from the template.
- `chat_template_kwargs.enable_thinking=false`: off. Anthropic `thinking.type=disabled`: off; enabled/unset: on.
- Request effort overrides the server default. `reasoning_effort:"none"` as a default KWARG reaches the
  template raw and 400s every plain request; `enable_thinking:false` as the default disables ALL levels. So
  thinking cannot be off-by-default with working levels on stock vLLM: default is medium.
- The Collective web chat's "off" sends `chat_template_kwargs.thinking_enabled` (Qwen ignores it); API
  clients are fine (e-gineering/llm-collective#84). Collective declares levels none/low/medium/xhigh.

## pi
`~/.pi/agent/models.json` `collective` / `qwen3-8-27b` has `thinkingLevelMap`
{off:none, minimal:low, low:low, medium:medium, high:xhigh, xhigh:xhigh, max:xhigh} so no pi level 400s;
`settings.json` defaultThinkingLevel is xhigh. Test:
`pi -p --provider collective --model qwen3-8-27b --thinking xhigh --no-session "..."`.

## Measured 2026-10-23 (300 W cap, beside CLEF + 2x TTS, some live traffic)
- quality.py 15/15 (math, code run against tests, 5 needles at ~44K/132K/198K tokens); tools OK; image OK.
- decode 76 tok/s single, 211 / 454 / 810 tok/s aggregate at 4 / 8 / 16.
- prefix caching: 222K-token prompt TTFT 104 s cold (with traffic), 2.4 s resent, 2.7 s same prefix + new
  turn. This is the agent win over NInfer.
- Qwen's tokenizer splits digits: synthetic filler can be far more tokens than words (check `/tokenize`;
  over 262,144 is a 400, not a server bug).

## Switch / rollback
Cannot co-reside with flint or ninfer (same slice). One commit: replicas here and in apps/flint-vllm.yaml
(or ninfer), plus the Collective backend block (the parked ones are commented in
apps/llm-collective-contributor.yaml). Pre-pull the image first; the node pulls images one at a time, so a
big pull also delays local-path helper pods (a new HDD PVC can sit Pending until it finishes).
