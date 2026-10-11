---
license: apache-2.0
base_model: Cloudflare/clef-omni
base_model_relation: quantized
tags:
  - nvfp4
  - fp4
  - modelopt
  - qwen3_omni_moe
  - clef
  - decision-model
  - blackwell
---

# clef-omni-nvfp4

NVFP4 (W4A4) quantization of [Cloudflare/clef-omni](https://huggingface.co/Cloudflare/clef-omni) (revision `0db1cd2`) made with NVIDIA ModelOpt 0.47.0. **22.0 GB instead of 70.8 GB.** Only the thinker's MoE experts (about 91% of the loaded weights) are FP4; attention, router, embeddings, `lm_head`, the vision and audio towers and the Clef joint decision head stay BF16.

> **Status:** experimental (`max` calibration). Faithful W4A4 replay agrees with BF16 on 92.6% of held-out decisions (TV 0.059) and scores 0.76 points below BF16 on the Decision Index; the INT4 sibling is the better choice for most people.

## Which Clef-Omni quant should I use?

| Model | Format | Size on disk | VRAM (weights) | Agrees with BF16 (top answer) | Decision Index 0.2.1 vs BF16 | Best for |
|---|---|---|---|---|---|---|
| [Cloudflare/clef-omni](https://huggingface.co/Cloudflare/clef-omni) | BF16 (original) | 70.8 GB | 60 GiB | reference | reference (avg 70.3; Cloudflare's table 70.7) | Reference quality, ~70 GB VRAM |
| [dbirks/clef-omni-nvfp4](https://huggingface.co/dbirks/clef-omni-nvfp4) | NVFP4 W4A4 (ModelOpt) | 22.0 GB | 28 GiB | 92.6% | -0.76 pts avg (31/42 within 2 pts) | Blackwell, native FP4 engines |
| [dbirks/clef-omni-int4](https://huggingface.co/dbirks/clef-omni-int4) | INT4 W4A16 g128 (AutoRound) | 20.8 GB | 19 GiB | 96.5% | -0.13 pts avg (33/42 within 2 pts) | ⭐ Recommended: best accuracy, smallest VRAM, any recent GPU |

**Recommended for:**

- **clef-omni-int4**: any recent GPU when you want the smallest VRAM footprint today; loads in plain transformers. Also the best bet on AMD (ROCm): AutoRound's PyTorch/Triton kernels run there, though we have not tested it.
- **clef-omni-nvfp4**: Blackwell engines with native FP4 tensor cores (W4A4 is the only variant that can run faster than BF16 there).
- **clef-omni** (Cloudflare BF16): the reference, if you have ~70 GB of VRAM.

*Size on disk* is the download. *VRAM (weights)* is what `serve_clef.py` holds on the GPU: the NVFP4 checkpoints are unpacked to BF16 there (transformers has no packed-NVFP4 kernels), so today only INT4 actually saves VRAM, and the NVFP4 files are for engines with native FP4. *Agrees with BF16* is the share of 258 held-out decision questions where the quant picks the same top answer as Cloudflare's BF16 model run through the same code. BF16 against itself under a different expert kernel scores 95.7%, so about 96% is the ceiling. *Decision Index* is the chance-corrected public index from the [Decision Index kit](https://github.com/apolinario/decision-index), edition 0.2.1 (the edition behind Cloudflare's published Clef-Omni results), on our hardware.

## Quick start

Serve it with [`serve_clef.py`](serve_clef.py) (also in [the repo it is maintained in](https://github.com/dbirks/home-k8s/blob/main/scripts/clef-omni/serve_clef.py)), a single-file [PEP 723](https://peps.python.org/pep-0723/) script, so `uv` installs everything it needs:

```bash
# uv runs a PEP 723 script straight from its URL (or download serve_clef.py and `uv run` it locally)
uv run https://huggingface.co/dbirks/clef-omni-nvfp4/resolve/main/serve_clef.py --model dbirks/clef-omni-nvfp4 --demo
uv run https://huggingface.co/dbirks/clef-omni-nvfp4/resolve/main/serve_clef.py --model dbirks/clef-omni-nvfp4 --port 8000
```

The first command answers one built-in decision and exits; the second starts the SystemOne HTTP server (`GET /healthz` turns 200 once the model is loaded):

```bash
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "clef-omni",
  "state": {"observation": "The player is facing a wall."},
  "questions": {"action": {"type": "choice", "criteria": {"left": "Turn left", "right": "Turn right", "forward": "Move forward"}}}
}'
```

Answers come from Cloudflare's own `systemone()` in `joint_schema_model.py`: one forward pass, per-option probabilities from the original BF16 joint decision head. Inputs can include images, audio and video (see the [base model card](https://huggingface.co/Cloudflare/clef-omni)).

## How this checkpoint runs

`serve_clef.py` keeps the FP4 experts packed in VRAM and unpacks each layer just in time for transformers' grouped kernels: about 28 GB on the card (15.3 GiB of expert weights) at about 0.55 s per decision. `--no-packed-nvfp4` unpacks everything at load (about 60 GB, about 0.16 s). By default activations stay BF16; `--simulate-fp4-activations` also rounds expert inputs to NVFP4 exactly as ModelOpt calibrated them (faithful W4A4, several seconds per decision, used for the benchmarks). Native FP4 execution needs an engine with NVFP4 kernels plus a hidden-state path into the Clef head.

## What was quantized

| Part | Precision |
|---|---|
| Thinker MoE experts (48 layers x 128 experts, gate/up/down) | **NVFP4 W4A4**: E2M1 weights in blocks of 16 with FP8 E4M3 block scales + FP32 global scale; FP4 activations |
| Attention, router (`mlp.gate`), embeddings, `lm_head` | BF16 |
| Vision and audio towers | BF16 |
| Clef joint head (`joint_head.safetensors`) | BF16, byte-identical to the original |

## Recipe

ModelOpt `NVFP4_EXPERTS_ONLY_CFG`, `max` calibration, plus exclusions for towers, talker, `lm_head`, router and embeddings; torch 2.11.0+cu130, transformers 5.10.2 (fused MoE experts, quantized per expert by ModelOpt 0.47 and split back on export); 512 Clef-format calibration records (383K tokens), 52 min on one RTX PRO 6000.

## Evaluation

Parity: 128 held-out records (258 questions; text from ultrachat_200k test_sft in Clef's state/questions schema, about 20% with synthetic image/video/audio, never used for calibration). Each exported quant is loaded through `serve_clef.py` and compared with Cloudflare's BF16 model run through the same code path (grouped experts). We report top-answer agreement and total variation (TV) between the per-option distributions. **Noise floor:** the same BF16 weights run through a different but equally valid expert kernel (eager vs grouped) agree on only 95.7% of these questions (TV 0.028), because many are near-ties. Agreement around 96% and TV near 0.03 therefore means indistinguishable from BF16 at this precision. Benchmarks: the Decision Index reproduction kit, edition 0.2.1, over `/v1/systemone` on one RTX PRO 6000 Blackwell.

Benchmarks for this checkpoint use the faithful W4A4 replay (`--simulate-fp4-activations`).

## Benchmarks

Decision Index 0.2.1 (the edition behind Cloudflare's published Clef-Omni table), run with the public [decision-index kit](https://github.com/apolinario/decision-index) (suite rebuilt and hash-verified against the lab's manifest) over `/v1/systemone` on one RTX PRO 6000 Blackwell. We score a deterministic stratified sample of 3,000 requests (about 58 per benchmark, `suite sample --n 3000`) rather than all ~150K, so each benchmark carries roughly ±5-7 points of sampling noise; the averages and the paired deltas (both models on the identical questions, `score --reference`) are the reliable signals. The kit's headline index is coverage-adjusted over the full suite, so it is not meaningful on a sample and is not reported.

| Benchmark (metric) | n | Cloudflare BF16 (published) | BF16 (ours) | This quant | Δ vs BF16 (same questions) |
|---|---|---|---|---|---|
| BFCL (case exact accuracy) | 59 | 98.2 | 100.0 | 100.0 | +0.0 |
| ToolRet (nDCG@10) | 59 | 66.6 | 66.5 | 68.4 | +1.9 |
| API-Bank (accuracy) | 59 | 92.7 | 79.7 | 81.4 | +1.7 |
| BANKING77 (macro-F1) | 59 | 94.8 | 96.5 | 96.5 | +0.0 |
| CLINC150+OOS (macro-F1) | 59 | 97.7 | 94.8 | 94.8 | +0.0 |
| RouterBench (selected quality (quality objective)) | 118 | 79.9 | 78.8 | 78.8 | +0.0 |
| Home appliance simulator (case exact accuracy) | 59 | 69.3 | 67.8 | 71.2 | +3.4 |
| SGD/SGD-X (macro-F1) | 59 | 39.2 | 34.2 | 34.2 | +0.0 |
| ContractNLI (macro-F1) | 59 | 80.3 | 79.4 | 80.6 | +1.1 |
| ANLI (macro-F1) | 59 | 65.5 | 71.4 | 72.8 | +1.4 |
| BPoMP (accuracy) | 246 | 96.8 | 98.4 | 97.6 | -0.8 |
| Humicroedit (accuracy) | 59 | 70.1 | 64.4 | 64.4 | +0.0 |
| POP909-CL (accuracy) | 59 | 2.6 | 1.7 | 1.7 | +0.0 |
| cfcolor (accuracy) | 59 | 62.9 | 61.0 | 59.3 | -1.7 |
| MMLU (accuracy) | 59 | 92.7 | 91.5 | 91.5 | +0.0 |
| GPQA Diamond (accuracy) | 59 | 47.4 | 52.5 | 47.5 | -5.1 |
| ARC-Easy (accuracy) | 59 | 99.8 | 96.6 | 96.6 | +0.0 |
| ARC-Challenge (accuracy) | 59 | 97.8 | 96.6 | 96.6 | +0.0 |
| WinoGrande (accuracy) | 59 | 96.7 | 100.0 | 100.0 | +0.0 |
| HellaSwag (accuracy) | 59 | 99.1 | 94.9 | 94.9 | +0.0 |
| GSM8K (accuracy) | 118 | 79.0 | 78.0 | 75.4 | -2.5 |
| ChessBench (accuracy) | 58 | 22.1 | 17.2 | 17.2 | +0.0 |
| MuSR (accuracy) | 58 | 83.8 | 82.8 | 86.2 | +3.4 |
| SATA-Bench (case exact accuracy) | 58 | 31.9 | 31.0 | 31.0 | +0.0 |
| SimpleBench (accuracy) | 10 | n/a | 40.0 | 40.0 | +0.0 |
| BRIGHT (nDCG@10) | 58 | 42.0 | 38.0 | 36.6 | -1.3 |
| Amazon ESCI (macro-F1) | 58 | 57.8 | 53.2 | 52.2 | -1.0 |
| ACOS (per-review F1) | 228 | 28.4 | 26.1 | 24.9 | -1.3 |
| FinEntity (macro-F1) | 58 | 96.3 | 97.5 | 97.5 | +0.0 |
| VAST (macro-F1) | 58 | 50.9 | 59.3 | 54.0 | -5.3 |
| NLI4CT (macro-F1) | 58 | 76.8 | 80.2 | 81.0 | +0.8 |
| CRUXEval (accuracy) | 58 | 88.8 | 82.8 | 84.5 | +1.7 |
| CLadder (accuracy) | 58 | 97.2 | 98.3 | 98.3 | +0.0 |
| HLE (accuracy) | 58 | n/a | 13.8 | 8.6 | -5.2 |
| ForecastBench (Brier (lower is better)) | 58 | 11.7 | 14.3 | 14.2 | -0.2 |
| Habermas Machine (accuracy) | 58 | 67.8 | 67.2 | 70.7 | +3.4 |
| PhishNChips phishing decisions (accuracy) | 58 | 73.2 | 79.3 | 70.7 | -8.6 |
| MMLU-Pro (accuracy) | 58 | 70.6 | 74.1 | 79.3 | +5.2 |
| BBH fixed-option tasks (accuracy) | 58 | 69.0 | 69.0 | 69.0 | +0.0 |
| RAGTruth response-level hallucination (F1 on hallucinated class) | 58 | 42.0 | 56.2 | 41.4 | -14.9 |
| HoVer claim verification (accuracy) | 58 | 64.2 | 67.2 | 65.5 | -1.7 |
| When2Call MCQ (accuracy) | 58 | 63.3 | 56.9 | 55.2 | -1.7 |
| New Yorker caption matching (accuracy) | 58 | 64.2 | 67.2 | 62.1 | -5.2 |

## Version history

Each version is an annotated git tag on this repo (`revision="v0.1"` etc. pins it). Newest last.

| Version | Date | Notes |
|---|---|---|
| `v0.1` | 2026-10-10 | First release: ModelOpt NVFP4 W4A4, experts-only, `max` calibration on 512 Clef-format records. 91.1% top-answer agreement with BF16 (mean TV 0.055). Also tagged `v0.1-max`. |
| `v0.2` | 2026-10-10 | Docs only, weights unchanged: added `serve_clef.py` (PEP 723 `uv run` server that loads this checkpoint with the Clef head), the quant comparison table, and this version history. |
| `v0.3` | 2026-10-11 | README and serve_clef.py, weights unchanged: Decision Index results (-0.76 pts vs BF16, faithful W4A4 replay), same-path parity (92.6%), the public INT4 sibling in the comparison table, and serve_clef.py keeps NVFP4 experts packed in VRAM (about 28 GB instead of about 60 GB) with the FP4 activation replay opt-in. |

## License

Apache-2.0, as the base model. All credit for the model goes to Cloudflare; this repo only changes the weight precision.
