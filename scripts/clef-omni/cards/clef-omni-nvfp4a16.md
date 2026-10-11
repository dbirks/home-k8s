---
license: apache-2.0
base_model: Cloudflare/clef-omni
base_model_relation: quantized
tags:
  - nvfp4
  - w4a16
  - modelopt
  - qwen3_omni_moe
  - clef
  - decision-model
---

# clef-omni-nvfp4a16

NVFP4 weight-only (W4A16) quantization of [Cloudflare/clef-omni](https://huggingface.co/Cloudflare/clef-omni) (revision `0db1cd2`) made with NVIDIA ModelOpt 0.47.0: FP4 expert weights, 16-bit activations. Only the thinker's MoE experts are quantized; everything else, including the Clef joint decision head, stays BF16.

> **Status:** first pass (`max` calibration). The exported checkpoint agrees with BF16 on 91.1% of held-out decisions (mean TV 0.047). ModelOpt's in-memory fake-quant measured 93.4%, but export recomputes some scales; the file number is the one that applies. INT4 (AutoRound) is currently the better choice.

## Which Clef-Omni quant should I use?

| Model | Format | Size on disk | VRAM (weights) | Agrees with BF16 (top answer) | Decision Index 0.2.1 vs BF16 | Best for |
|---|---|---|---|---|---|---|
| [Cloudflare/clef-omni](https://huggingface.co/Cloudflare/clef-omni) | BF16 (original) | 70.8 GB | 60 GiB | reference | reference (avg 70.3; Cloudflare's table 70.7) | Reference quality, ~70 GB VRAM |
| [dbirks/clef-omni-nvfp4](https://huggingface.co/dbirks/clef-omni-nvfp4) | NVFP4 W4A4 (ModelOpt) | 22.0 GB | 60 GiB | 92.3% | pending | Blackwell, native FP4 engines |
| [dbirks/clef-omni-nvfp4a16](https://huggingface.co/dbirks/clef-omni-nvfp4a16) | NVFP4 W4A16 (ModelOpt) | 22.0 GB | 60 GiB | 91.1% | -0.24 pts avg (38/42 within 2 pts) | Blackwell / vLLM Marlin, higher fidelity than W4A4 |
| [dbirks/clef-omni-int4](https://huggingface.co/dbirks/clef-omni-int4) | INT4 W4A16 g128 (AutoRound) | 20.8 GB | 19 GiB | 96.1% | -0.13 pts avg (33/42 within 2 pts) | ⭐ Recommended: best accuracy, smallest VRAM, any recent GPU |
| [dbirks/clef-omni-nvfp4a16-autoround](https://huggingface.co/dbirks/clef-omni-nvfp4a16-autoround) | NVFP4 W4A16 (AutoRound, tuned) | 22.0 GB | 21 GiB | 94.6% | -0.57 pts avg (37/42 within 2 pts) | Highest-fidelity FP4 weights |
| [dbirks/clef-omni-fp8](https://huggingface.co/dbirks/clef-omni-fp8) | FP8 block 128x128 (W8A16 / dynamic) | 34.5 GB | 33 GiB | pending | -0.20 pts avg (40/42 within 2 pts) | Highest fidelity at about half of BF16's VRAM |

**Recommended for:**

- **clef-omni-int4**: any recent GPU when you want the smallest VRAM footprint today; loads in plain transformers. Also the best bet on AMD (ROCm): AutoRound's PyTorch/Triton kernels run there, though we have not tested it.
- **clef-omni-nvfp4a16**: Blackwell, or vLLM's Marlin MoE path on older cards; NVFP4 weights with 16-bit activations (higher fidelity than W4A4).
- **clef-omni-nvfp4**: Blackwell engines with native FP4 tensor cores (W4A4 is the only variant that can run faster than BF16 there).
- **clef-omni** (Cloudflare BF16): the reference, if you have ~70 GB of VRAM.

*Size on disk* is the download. *VRAM (weights)* is what `serve_clef.py` holds on the GPU: the NVFP4 checkpoints are unpacked to BF16 there (transformers has no packed-NVFP4 kernels), so today only INT4 actually saves VRAM, and the NVFP4 files are for engines with native FP4. *Agrees with BF16* is the share of 258 held-out decision questions where the quant picks the same top answer as Cloudflare's BF16 model. *Decision Index* is the chance-corrected public index from the [Decision Index kit](https://github.com/apolinario/decision-index), edition 0.2.1 (the edition behind Cloudflare's published Clef-Omni results), on our hardware.

## Quick start

Serve it with [`serve_clef.py`](serve_clef.py) (also in [the repo it is maintained in](https://github.com/dbirks/home-k8s/blob/main/scripts/clef-omni/serve_clef.py)), a single-file [PEP 723](https://peps.python.org/pep-0723/) script, so `uv` installs everything it needs:

```bash
# uv runs a PEP 723 script straight from its URL (or download serve_clef.py and `uv run` it locally)
uv run https://huggingface.co/dbirks/clef-omni-nvfp4a16/resolve/main/serve_clef.py --model dbirks/clef-omni-nvfp4a16 --demo
uv run https://huggingface.co/dbirks/clef-omni-nvfp4a16/resolve/main/serve_clef.py --model dbirks/clef-omni-nvfp4a16 --port 8000
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

`serve_clef.py` unpacks the FP4 expert weights to BF16 on load (bit-exact), so it runs anywhere transformers runs, with BF16-like memory. vLLM's ModelOpt loader runs `W4A16_NVFP4` MoE checkpoints on its Marlin kernels (memory saving, BF16-class speed), but no engine yet feeds the thinker's hidden states into the Clef head.

## What was quantized

| Part | Precision |
|---|---|
| Thinker MoE experts | **NVFP4 weights** (E2M1, blocks of 16, FP8 E4M3 block scales + FP32 global scale), BF16 activations |
| Attention, router, embeddings, `lm_head`, vision and audio towers | BF16 |
| Clef joint head | BF16, byte-identical to the original |

## Recipe

ModelOpt `NVFP4_EXPERTS_ONLY_CFG` with the expert input quantizers disabled (weight-only), `max` calibration straight from the weights (no calibration data needed), same exclusions as the W4A4 variant; torch 2.11.0+cu130, transformers 5.10.2, ModelOpt 0.47.0.

## Evaluation

Parity: the 128 held-out records (258 questions; text from ultrachat_200k test_sft in Clef's state/questions schema, about 20% with synthetic image/video/audio, never used for calibration) are scored by the BF16 original and by the exported quant loaded through `serve_clef.py`. We report top-answer agreement and total variation between the per-option distributions. Benchmarks: the Decision Index reproduction kit (apolinario/decision-index), edition 0.2.1, driven over `/v1/systemone`, on one RTX PRO 6000 Blackwell.


## Benchmarks

Decision Index 0.2.1 (the edition behind Cloudflare's published Clef-Omni table), run with the public [decision-index kit](https://github.com/apolinario/decision-index) (suite rebuilt and hash-verified against the lab's manifest) over `/v1/systemone` on one RTX PRO 6000 Blackwell. We score a deterministic stratified sample of 3,000 requests (about 58 per benchmark, `suite sample --n 3000`) rather than all ~150K, so each benchmark carries roughly ±5-7 points of sampling noise; the averages and the paired deltas (both models on the identical questions, `score --reference`) are the reliable signals. The kit's headline index is coverage-adjusted over the full suite, so it is not meaningful on a sample and is not reported.

| Benchmark (metric) | n | Cloudflare BF16 (published) | BF16 (ours) | This quant | Δ vs BF16 (same questions) |
|---|---|---|---|---|---|
| BFCL (case exact accuracy) | 59 | 98.2 | 100.0 | 100.0 | +0.0 |
| ToolRet (nDCG@10) | 59 | 66.6 | 66.5 | 68.6 | +2.1 |
| API-Bank (accuracy) | 59 | 92.7 | 79.7 | 81.4 | +1.7 |
| BANKING77 (macro-F1) | 59 | 94.8 | 96.5 | 96.5 | +0.0 |
| CLINC150+OOS (macro-F1) | 59 | 97.7 | 94.8 | 94.8 | +0.0 |
| RouterBench (selected quality (quality objective)) | 118 | 79.9 | 78.8 | 78.8 | +0.0 |
| Home appliance simulator (case exact accuracy) | 59 | 69.3 | 67.8 | 69.5 | +1.7 |
| SGD/SGD-X (macro-F1) | 59 | 39.2 | 34.2 | 34.2 | +0.0 |
| ContractNLI (macro-F1) | 59 | 80.3 | 79.4 | 79.1 | -0.3 |
| ANLI (macro-F1) | 59 | 65.5 | 71.4 | 69.5 | -1.9 |
| BPoMP (accuracy) | 246 | 96.8 | 98.4 | 97.6 | -0.8 |
| Humicroedit (accuracy) | 59 | 70.1 | 64.4 | 62.7 | -1.7 |
| POP909-CL (accuracy) | 59 | 2.6 | 1.7 | 1.7 | +0.0 |
| cfcolor (accuracy) | 59 | 62.9 | 61.0 | 59.3 | -1.7 |
| MMLU (accuracy) | 59 | 92.7 | 91.5 | 91.5 | +0.0 |
| GPQA Diamond (accuracy) | 59 | 47.4 | 52.5 | 50.8 | -1.7 |
| ARC-Easy (accuracy) | 59 | 99.8 | 96.6 | 96.6 | +0.0 |
| ARC-Challenge (accuracy) | 59 | 97.8 | 96.6 | 96.6 | +0.0 |
| WinoGrande (accuracy) | 59 | 96.7 | 100.0 | 100.0 | +0.0 |
| HellaSwag (accuracy) | 59 | 99.1 | 94.9 | 94.9 | +0.0 |
| GSM8K (accuracy) | 118 | 79.0 | 78.0 | 77.1 | -0.8 |
| ChessBench (accuracy) | 58 | 22.1 | 17.2 | 17.2 | +0.0 |
| MuSR (accuracy) | 58 | 83.8 | 82.8 | 82.8 | +0.0 |
| SATA-Bench (case exact accuracy) | 58 | 31.9 | 31.0 | 32.8 | +1.7 |
| SimpleBench (accuracy) | 10 | n/a | 40.0 | 40.0 | +0.0 |
| BRIGHT (nDCG@10) | 58 | 42.0 | 38.0 | 37.4 | -0.5 |
| Amazon ESCI (macro-F1) | 58 | 57.8 | 53.2 | 53.2 | +0.0 |
| ACOS (per-review F1) | 228 | 28.4 | 26.1 | 24.9 | -1.2 |
| FinEntity (macro-F1) | 58 | 96.3 | 97.5 | 97.5 | +0.0 |
| VAST (macro-F1) | 58 | 50.9 | 59.3 | 58.9 | -0.3 |
| NLI4CT (macro-F1) | 58 | 76.8 | 80.2 | 84.2 | +4.1 |
| CRUXEval (accuracy) | 58 | 88.8 | 82.8 | 82.8 | +0.0 |
| CLadder (accuracy) | 58 | 97.2 | 98.3 | 98.3 | +0.0 |
| HLE (accuracy) | 58 | n/a | 13.8 | 10.3 | -3.4 |
| ForecastBench (Brier (lower is better)) | 58 | 11.7 | 14.3 | 14.2 | -0.1 |
| Habermas Machine (accuracy) | 58 | 67.8 | 67.2 | 65.5 | -1.7 |
| PhishNChips phishing decisions (accuracy) | 58 | 73.2 | 79.3 | 72.4 | -6.9 |
| MMLU-Pro (accuracy) | 58 | 70.6 | 74.1 | 74.1 | +0.0 |
| BBH fixed-option tasks (accuracy) | 58 | 69.0 | 69.0 | 70.7 | +1.7 |
| RAGTruth response-level hallucination (F1 on hallucinated class) | 58 | 42.0 | 56.2 | 56.2 | +0.0 |
| HoVer claim verification (accuracy) | 58 | 64.2 | 67.2 | 65.5 | -1.7 |
| When2Call MCQ (accuracy) | 58 | 63.3 | 56.9 | 56.9 | +0.0 |
| New Yorker caption matching (accuracy) | 58 | 64.2 | 67.2 | 69.0 | +1.7 |

## Version history

Each version is an annotated git tag on this repo (`revision="v0.1"` etc. pins it). Newest last.

| Version | Date | Notes |
|---|---|---|
| `v0.1` | 2026-10-10 | First release: ModelOpt W4A16_NVFP4, experts-only, weight-only `max` calibration (no calibration data). 93.4% top-answer agreement with BF16 (mean TV 0.043). Ships serve_clef.py. |

## License

Apache-2.0, as the base model. All credit for the model goes to Cloudflare; this repo only changes the weight precision.
