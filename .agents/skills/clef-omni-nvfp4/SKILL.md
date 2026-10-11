---
name: clef-omni-nvfp4
description: Quantize, serve, grade and benchmark Cloudflare Clef-Omni (NVFP4, NVFP4A16, INT4, FP8). Quantize Cloudflare Clef-Omni (Qwen3-Omni-30B-A3B thinker plus a BF16 joint decision head) to NVFP4 W4A4 with NVIDIA ModelOpt on the single RTX PRO 6000 (SM120), and run any ModelOpt PTQ against a transformers-5 MoE model here. Use when re-running or extending the issue 145 quant Jobs, when a ModelOpt PTQ Job dies after a long weight load, or when grading a quantized Clef against its BF16 goldens.
---

# Clef-Omni NVFP4 (ModelOpt) on home-k8s

Tracking issue: **#145**. Manifests: `apps/clef-omni-*.yaml[.hold]`. Everything lives on the
`clef-omni-hdd` PVC (250Gi, local-path-hdd) mounted at `/work`:

| path | what |
|---|---|
| `/work/clef-omni` | BF16 release, pinned rev `0db1cd2`, `.clef-omni.sha256` manifest |
| `/work/venv` | torch 2.11.0+cu130 / transformers 5.10.2 (+ modelopt 0.47.0) venv, reused by every Job |
| `/work/bf16-results/` | Stage-1 goldens (`smoke_bf16.jsonl`, `summary.json`) |
| `/work/nvfp4-experts/{results,export}` | Stage 2a output: parity report, quantizer state, packed checkpoint |
| `/work/nvfp4-rehearsal/` | tiny-model rehearsal output (throwaway) |
| `/work/logs/` | per-stage logs + pip freezes |

## Before any GPU window

1. Pennyroyal owns ~97.8 of 97.9 GB. Set `replicas: 0` in `apps/pennyroyal-flashnext.yaml`
   **via git**, and restore `replicas: 1` after. Collective contributors lose their engine meanwhile.
2. Host RAM (46 GB, no swap) is the real limit. Measured: BF16 load with
   `device_map={"": "cuda"}` streams shards and leaves MemAvailable at ~25 GiB (the official
   loader does NOT CPU-stage the checkpoint). Keep the pod limit at ~34Gi as a backstop.
3. Keep artifacts on the HDD PVC, not NVMe (DiskPressure history).

## Facts that cost a run each (don't relearn them)

- **transformers 5 fuses MoE experts** into 3-D `gate_up_proj`/`down_proj` Parameters. Only
  **ModelOpt >= 0.47** (`_QuantFusedExperts`) quantizes them and re-splits them on export into
  the per-expert `experts.E.{gate,up,down}_proj.{weight,weight_scale,weight_scale_2,input_scale}`
  layout vLLM expects. It hooks `F.linear`, so experts must run **eager**: load with
  `experts_implementation="eager"` (ModelOpt also forces it). Eager vs the default grouped_mm
  path shifts BF16 probabilities by up to ~0.05 with no flipped answers. Gate on flips, not on a tight Δp.
- 0.47 also clamps per-block scales before the E4M3 cast (the 0.44 NaN-byte bug). Still scan
  the export for `0x7F`/`0xFF` bytes in `float8_e4m3fn` tensors.
- Install `nvidia-modelopt` **without `[hf]`**, under a constraints file built from
  `pip freeze`, or it drags deepspeed/diffusers and can float torch/transformers.
- ModelOpt 0.47 imports **`requests`** without declaring it. Preflight
  `import modelopt.torch.quantization, modelopt.torch.export` before loading weights;
  `import modelopt` alone proves nothing.
- The NVFP4 fake-quant forward JIT-compiles a **Triton kernel and needs gcc**:
  `apt-get install gcc libc6-dev` on `python:3.12-slim`. It also needs CUDA (`amax must be a CUDA
  tensor`), so a CPU rehearsal stops at the first quantized forward.
- `Qwen3OmniMoeForConditionalGeneration` has **no `forward()`** (generate-only), but
  `export_hf_checkpoint` runs `model(fake_input_ids)`. Bind a forward that delegates to
  `self.thinker(input_ids=..., use_cache=False)` before exporting.
- `ClefModel` swaps `thinker.lm_head` for `Identity` and keeps the real one as
  `output_embeddings` (option scoring). Put it back before export; exclude `*lm_head*`.
- The script's PASS/FAIL covers the artifact scan only. Parity gates are reported separately.
- Exclusions used: `*audio_tower* *visual* *talker* *code2wav* *lm_head* *mlp.gate.* *embed*`.
  The joint head never goes through ModelOpt. Its sha256 must match the original.

## The pattern: rehearse, then load

`quant_experts.py` (inlined in the Job) has `CLEF_REHEARSAL=1`. That mode builds a tiny random
Qwen3-Omni-MoE plus head from the real config (2 layers, 8 experts, shrunk towers, real
tokenizer/processor/vocab) and runs the identical calibrate → parity → export → scan path in
under a minute. The Job runs it first and refuses to load the real checkpoint if it fails. Local CPU
dry run (stops at the first NVFP4 forward, by design):

```bash
# model dir = the release's non-weight files (config, tokenizer, processor, joint_head_config, joint_schema_model.py)
CLEF_REHEARSAL=1 CLEF_MODEL_DIR=/tmp/clef/model CLEF_ROOT=/tmp/clef/out \
uv run --no-project --python 3.12 --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  --with torch==2.11.0 --with torchvision==0.26.0 --with transformers==5.10.2 --with nvidia-modelopt==0.47.0 \
  --with accelerate --with safetensors --with numpy --with pillow --with av --with pyarrow --with requests --with huggingface_hub \
  python -I quant_experts.py
```

## Grading

- Parity runs in the SAME process: score held-out records in BF16, quantize, score again with
  fake-quant. Report top-1 agreement, mean/p95/max total variation, and flips with BF16 margin.
  Provisional gates from #145: top-1 >= 0.98, mean TV <= 0.03.
- Calibration and eval records come from ultrachat_200k `test_sft` wrapped in Clef's
  `state`/`questions` schema (choice/noul/score), ~20 % with synthetic image/video/audio. They are
  disjoint by row. These are wiring-grade, not task-representative. Real game/screenshot
  records would be better calibration once available.
- Fake-quant parity is NOT proof of native FP4 execution. Serving still has to show
  FlashInfer/CUTLASS FP4 GEMMs (Stage 3/4).

## Results so far

**Stage 2a (experts-only NVFP4, `max`, 512 calibration records), 2026-10-10:** the artifact is
good. It is 22.0 GB (vs 70.8 GB BF16), with 18,432 packed FP4 expert tensors, 0 NaN scale bytes,
no BF16 expert weights left, and a matching head hash. Quality **missed** the gates: top-1
agreement 0.911 (gate 0.98), mean TV 0.055 (gate 0.03), p95 TV 0.184, max 0.372, with 23 flips in
258 questions. The script's `STAGE 2a PASS` line covers only the artifact scan. Read
`results/parity_fakequant.json` for quality.
Timings: load 573 s, calibration 3,097 s (~6 s/record, since eager experts are slow), fake-quant
eval of 128 records ~11 min, export 213 s. Peak VRAM 59.5 GiB; host RAM never below 19 GiB free.
Next levers: check whether flips sit on low-margin questions (parity rows), calibrate with
task-representative records, try awq_lite/awq_clip, or keep sensitive layers (first/last
expert layers, down_proj) in FP8/BF16.

## Retrying without re-calibrating

`results/quantizer_state.pt` holds `mto.modelopt_state(backbone)` plus every `*quantizer*` tensor.
Load BF16 the same way, `mto.restore_from_modelopt_state(backbone, state["modelopt_state"])`,
`backbone.load_state_dict(state["quantizer_state_dict"], strict=False)`, then export.

## The family pipeline (2026-10-11, issues #150-#154)

Everything lives in `scripts/clef-omni/` (source of truth). `render_configmap.py` renders it into the
`clef-omni-scripts` ConfigMap, which Jobs mount at `/scripts`.

| Script | Job pattern | Notes |
|---|---|---|
| `quant_int4_autoround.py` (`AR_SCHEME=W4A16\|NVFP4\|NVFP4A16`) | `apps/clef-omni-quant-{int4,nvfp4a16-ar,nvfp4-ar}-*.yaml.hold` | rehearsal first; host-RAM guard (keeps AutoRound from moving the whole model to CPU); fails on swallowed calibration errors |
| `quant_fp8_block.py` | `apps/clef-omni-quant-fp8-*.yaml.hold` (CPU) | per-expert BF16 to FP8 128x128 straight from the shards, transformers' Fp8Quantize math, 15 min, 1.2 GiB RSS |
| `serve_clef.py` (PEP 723) | published in every HF repo | BF16 / AutoRound INT4 (grouped int4 fast path) / ModelOpt + AutoRound NVFP4 (packed) / FP8 block (packed) |
| `eval_export.py` + `reparity.py` | `apps/clef-omni-grade-*.yaml.hold` | parity of the EXPORT through serve_clef.py; reparity rescores saved rows against a new reference |
| `model_cards.py` + `quants.json` + `hf_publish.py` | `apps/clef-omni-publish-*.yaml.hold` | card table, recommended-for list, per-benchmark section, version history; tags carry the version note |

Decision Index: `apps/decision-index-bench-all-*.yaml.hold`. It runs serve_clef.py per model on
localhost, then the kit's http engine over a 3,000-request 0.2.1 sample, score, and `score --reference` vs BF16.

## Lessons (each cost a run)
- **Parity noise floor: BF16 vs BF16 under eager vs grouped experts agrees only 95.7% (TV 0.028).**
  Always grade against a reference produced by the SAME code path (`eval_export.py --export /work/clef-omni`).
  A 98% top-1 target is unreachable. Judge by TV vs the floor.
- **ModelOpt NVFP4 exports differ from ModelOpt's in-memory fake-quant** (93.4% fake vs 91.1% file for W4A16),
  because export recomputes scales (shared gate/up global scale). Grade the file.
- **AutoRound 0.16 cannot load its own weight-only NVFP4 export** (no inference backend for act_bits 16).
  serve_clef.py dequantizes it (reciprocal `weight_global_scale`, low nibble first).
- **AutoRound INT4 inference was 2.7 s per decision:** its Qwen3-Omni unfuse plus QuantLinear falls back to a
  per-expert Python loop with a GPU sync per expert. serve_clef.py re-stacks the experts and does dequant +
  grouped_mm: 0.21 s.
- **AutoRound NVFP4 W4A4:** OOM at the defaults; `low_gpu_mem_usage` + seqlen 4096 fits at 90.9 GB but runs
  ~10 min per block (~9.5 h). With 128 records some experts get no tokens, so there is no `act_max` and the
  export asserts. Shelved.
- **The Decision Index headline index is coverage-adjusted over the full ~150K suite,** so it reads about 2 on a
  sample. Report per-benchmark scores and paired deltas instead.
- **Bench Job r1 exited 0 after every server died** (no fastapi/uvicorn in the venv). Check for results, not for Job
  completion.
- **Workstation /home is btrfs and nearly full.** uv caches from local tests filled the metadata space, which
  makes git fail with "unable to write new index file". Run `uv cache prune` / `uv cache clean`.
