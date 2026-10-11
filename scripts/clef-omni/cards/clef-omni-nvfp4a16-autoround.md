---
license: apache-2.0
base_model: Cloudflare/clef-omni
base_model_relation: quantized
tags:
  - nvfp4
  - w4a16
  - auto-round
  - qwen3_omni_moe
  - clef
  - decision-model
---

# clef-omni-nvfp4a16-autoround

NVFP4 weight-only (W4A16) quantization of [Cloudflare/clef-omni](https://huggingface.co/Cloudflare/clef-omni) (revision `0db1cd2`) made with Intel AutoRound 0.16.0: SignRound-tuned FP4 expert weights, 16-bit activations. Only the thinker's MoE experts are quantized; everything else, including the Clef joint decision head, stays BF16.

> **Status:** Agrees with BF16 on 94.2% of held-out decisions (TV 0.035) and scores 0.57 points below BF16 on the Decision Index: the most faithful FP4 variant, with packed FP4 experts in VRAM (20.6 GiB weights, about 0.6 s per decision).

## Which Clef-Omni quant should I use?

| Model | Format | Size on disk | VRAM (weights) | Agrees with BF16 (top answer) | Decision Index 0.2.1 vs BF16 | Best for |
|---|---|---|---|---|---|---|
| [Cloudflare/clef-omni](https://huggingface.co/Cloudflare/clef-omni) | BF16 (original) | 70.8 GB | 60 GiB | reference | reference (avg 70.3; Cloudflare's table 70.7) | Reference quality, ~70 GB VRAM |
| [dbirks/clef-omni-nvfp4](https://huggingface.co/dbirks/clef-omni-nvfp4) | NVFP4 W4A4 (ModelOpt) | 22.0 GB | 28 GiB | 92.6% | -0.76 pts avg (31/42 within 2 pts) | Blackwell, native FP4 engines |
| [dbirks/clef-omni-nvfp4a16](https://huggingface.co/dbirks/clef-omni-nvfp4a16) | NVFP4 W4A16 (ModelOpt) | 22.0 GB | 28 GiB | 91.1% | -0.24 pts avg (38/42 within 2 pts) | Blackwell / vLLM Marlin, higher fidelity than W4A4 |
| [dbirks/clef-omni-int4](https://huggingface.co/dbirks/clef-omni-int4) | INT4 W4A16 g128 (AutoRound) | 20.8 GB | 19 GiB | 96.5% | -0.13 pts avg (33/42 within 2 pts) | ⭐ Recommended: best accuracy, smallest VRAM, any recent GPU |
| [dbirks/clef-omni-nvfp4a16-autoround](https://huggingface.co/dbirks/clef-omni-nvfp4a16-autoround) | NVFP4 W4A16 (AutoRound, tuned) | 22.0 GB | 21 GiB | 94.2% | -0.57 pts avg (37/42 within 2 pts) | Highest-fidelity FP4 weights |
| [dbirks/clef-omni-fp8](https://huggingface.co/dbirks/clef-omni-fp8) | FP8 block 128x128 (W8A16 / dynamic) | 34.5 GB | 32 GiB | 95.7% | -0.20 pts avg (40/42 within 2 pts) | Highest fidelity at about half of BF16's VRAM |

**Recommended for:**

- **clef-omni-int4**: any recent GPU when you want the smallest VRAM footprint today; loads in plain transformers. Also the best bet on AMD (ROCm): AutoRound's PyTorch/Triton kernels run there, though we have not tested it.
- **clef-omni-nvfp4a16**: Blackwell, or vLLM's Marlin MoE path on older cards; NVFP4 weights with 16-bit activations (higher fidelity than W4A4).
- **clef-omni-nvfp4**: Blackwell engines with native FP4 tensor cores (W4A4 is the only variant that can run faster than BF16 there).
- **clef-omni** (Cloudflare BF16): the reference, if you have ~70 GB of VRAM.

*Size on disk* is the download. *VRAM (weights)* is what `serve_clef.py` holds on the GPU: the NVFP4 checkpoints are unpacked to BF16 there (transformers has no packed-NVFP4 kernels), so today only INT4 actually saves VRAM, and the NVFP4 files are for engines with native FP4. *Agrees with BF16* is the share of 258 held-out decision questions where the quant picks the same top answer as Cloudflare's BF16 model run through the same code. BF16 against itself under a different expert kernel scores 95.7%, so about 96% is the ceiling. *Decision Index* is the chance-corrected public index from the [Decision Index kit](https://github.com/apolinario/decision-index), edition 0.2.1 (the edition behind Cloudflare's published Clef-Omni results), on our hardware.

## Quick start

Serve it with [`serve_clef.py`](serve_clef.py) (also in [the repo it is maintained in](https://github.com/dbirks/home-k8s/blob/main/scripts/clef-omni/serve_clef.py)), a single-file [PEP 723](https://peps.python.org/pep-0723/) script, so `uv` installs everything it needs:

```bash
# uv runs a PEP 723 script straight from its URL (or download serve_clef.py and `uv run` it locally)
uv run https://huggingface.co/dbirks/clef-omni-nvfp4a16-autoround/resolve/main/serve_clef.py --model dbirks/clef-omni-nvfp4a16-autoround --demo
uv run https://huggingface.co/dbirks/clef-omni-nvfp4a16-autoround/resolve/main/serve_clef.py --model dbirks/clef-omni-nvfp4a16-autoround --port 8000
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

AutoRound 0.16 cannot load its own weight-only NVFP4 export (no inference backend for `act_bits=16`), so `serve_clef.py` unpacks the FP4 expert weights to BF16 itself on load (same dequantization as AutoRound's NVFP4 path, verified bit-exact) and runs them on transformers' grouped kernels. BF16-like speed and memory.

## What was quantized

| Part | Precision |
|---|---|
| Thinker MoE experts | **NVFP4 weights** (blocks of 16), AutoRound-tuned, BF16 activations |
| Attention, router, embeddings, `lm_head`, vision and audio towers | BF16 |
| Clef joint head | BF16, byte-identical to the original |

## Recipe

`scheme={bits: 4, group_size: 16, data_type: nv_fp, act_bits: 16}`, AutoRound 0.16.0, `iters=200`, `nsamples=512` Clef-format records fed through Clef's real multimodal forward path, `batch_size=1`, `gradient_accumulate_steps=8`, `ignore_layers=mlp.gate,lm_head,self_attn`; AutoRound's Qwen3-Omni handler unfuses the transformers-5 fused experts into per-expert linears; torch 2.11.0+cu130, transformers 5.10.2; one RTX PRO 6000 Blackwell.

## Evaluation

Parity: 128 held-out records (258 questions; text from ultrachat_200k test_sft in Clef's state/questions schema, about 20% with synthetic image/video/audio, never used for calibration). Each exported quant is loaded through `serve_clef.py` and compared with Cloudflare's BF16 model run through the same code path (grouped experts). We report top-answer agreement and total variation (TV) between the per-option distributions. **Noise floor:** the same BF16 weights run through a different but equally valid expert kernel (eager vs grouped) agree on only 95.7% of these questions (TV 0.028), because many are near-ties. Agreement around 96% and TV near 0.03 therefore means indistinguishable from BF16 at this precision. Benchmarks: the Decision Index reproduction kit, edition 0.2.1, over `/v1/systemone` on one RTX PRO 6000 Blackwell.


## Benchmarks

Decision Index 0.2.1 (the edition behind Cloudflare's published Clef-Omni table), run with the public [decision-index kit](https://github.com/apolinario/decision-index) (suite rebuilt and hash-verified against the lab's manifest) over `/v1/systemone` on one RTX PRO 6000 Blackwell. We score a deterministic stratified sample of 3,000 requests (about 58 per benchmark, `suite sample --n 3000`) rather than all ~150K, so each benchmark carries roughly ±5-7 points of sampling noise; the averages and the paired deltas (both models on the identical questions, `score --reference`) are the reliable signals. The kit's headline index is coverage-adjusted over the full suite, so it is not meaningful on a sample and is not reported.

| Benchmark (metric) | n | Cloudflare BF16 (published) | BF16 (ours) | This quant | Δ vs BF16 (same questions) |
|---|---|---|---|---|---|
| BFCL (case exact accuracy) | 59 | 98.2 | 100.0 | 100.0 | +0.0 |
| ToolRet (nDCG@10) | 59 | 66.6 | 66.5 | 66.5 | -0.0 |
| API-Bank (accuracy) | 59 | 92.7 | 79.7 | 79.7 | +0.0 |
| BANKING77 (macro-F1) | 59 | 94.8 | 96.5 | 96.5 | +0.0 |
| CLINC150+OOS (macro-F1) | 59 | 97.7 | 94.8 | 94.8 | +0.0 |
| RouterBench (selected quality (quality objective)) | 118 | 79.9 | 78.8 | 78.8 | +0.0 |
| Home appliance simulator (case exact accuracy) | 59 | 69.3 | 67.8 | 66.1 | -1.7 |
| SGD/SGD-X (macro-F1) | 59 | 39.2 | 34.2 | 34.2 | +0.0 |
| ContractNLI (macro-F1) | 59 | 80.3 | 79.4 | 81.2 | +1.7 |
| ANLI (macro-F1) | 59 | 65.5 | 71.4 | 71.4 | +0.0 |
| BPoMP (accuracy) | 246 | 96.8 | 98.4 | 98.4 | +0.0 |
| Humicroedit (accuracy) | 59 | 70.1 | 64.4 | 66.1 | +1.7 |
| POP909-CL (accuracy) | 59 | 2.6 | 1.7 | 3.4 | +1.7 |
| cfcolor (accuracy) | 59 | 62.9 | 61.0 | 55.9 | -5.1 |
| MMLU (accuracy) | 59 | 92.7 | 91.5 | 91.5 | +0.0 |
| GPQA Diamond (accuracy) | 59 | 47.4 | 52.5 | 49.2 | -3.4 |
| ARC-Easy (accuracy) | 59 | 99.8 | 96.6 | 98.3 | +1.7 |
| ARC-Challenge (accuracy) | 59 | 97.8 | 96.6 | 96.6 | +0.0 |
| WinoGrande (accuracy) | 59 | 96.7 | 100.0 | 100.0 | +0.0 |
| HellaSwag (accuracy) | 59 | 99.1 | 94.9 | 94.9 | +0.0 |
| GSM8K (accuracy) | 118 | 79.0 | 78.0 | 76.3 | -1.7 |
| ChessBench (accuracy) | 58 | 22.1 | 17.2 | 19.0 | +1.7 |
| MuSR (accuracy) | 58 | 83.8 | 82.8 | 82.8 | +0.0 |
| SATA-Bench (case exact accuracy) | 58 | 31.9 | 31.0 | 31.0 | +0.0 |
| SimpleBench (accuracy) | 10 | n/a | 40.0 | 30.0 | -10.0 |
| BRIGHT (nDCG@10) | 58 | 42.0 | 38.0 | 37.9 | -0.1 |
| Amazon ESCI (macro-F1) | 58 | 57.8 | 53.2 | 52.6 | -0.6 |
| ACOS (per-review F1) | 228 | 28.4 | 26.1 | 25.3 | -0.8 |
| FinEntity (macro-F1) | 58 | 96.3 | 97.5 | 97.5 | +0.0 |
| VAST (macro-F1) | 58 | 50.9 | 59.3 | 59.3 | +0.1 |
| NLI4CT (macro-F1) | 58 | 76.8 | 80.2 | 82.2 | +2.0 |
| CRUXEval (accuracy) | 58 | 88.8 | 82.8 | 84.5 | +1.7 |
| CLadder (accuracy) | 58 | 97.2 | 98.3 | 98.3 | +0.0 |
| HLE (accuracy) | 58 | n/a | 13.8 | 12.1 | -1.7 |
| ForecastBench (Brier (lower is better)) | 58 | 11.7 | 14.3 | 14.0 | -0.4 |
| Habermas Machine (accuracy) | 58 | 67.8 | 67.2 | 69.0 | +1.7 |
| PhishNChips phishing decisions (accuracy) | 58 | 73.2 | 79.3 | 74.1 | -5.2 |
| MMLU-Pro (accuracy) | 58 | 70.6 | 74.1 | 75.9 | +1.7 |
| BBH fixed-option tasks (accuracy) | 58 | 69.0 | 69.0 | 69.0 | +0.0 |
| RAGTruth response-level hallucination (F1 on hallucinated class) | 58 | 42.0 | 56.2 | 46.7 | -9.6 |
| HoVer claim verification (accuracy) | 58 | 64.2 | 67.2 | 65.5 | -1.7 |
| When2Call MCQ (accuracy) | 58 | 63.3 | 56.9 | 56.9 | +0.0 |
| New Yorker caption matching (accuracy) | 58 | 64.2 | 67.2 | 69.0 | +1.7 |

## Version history

Each version is an annotated git tag on this repo (`revision="v0.1"` etc. pins it). Newest last.

| Version | Date | Notes |
|---|---|---|
| `v0.1` | 2026-10-11 | First release: AutoRound 0.16.0 NVFP4 weight-only (W4A16), experts-only, 200 SignRound steps per block on 512 Clef-format records (141 min, low_gpu_mem_usage). Export graded through serve_clef.py: 94.6% top-answer agreement with BF16 (mean TV 0.032), 20.6 GiB packed weights, 0.59 s per decision. |
| `v0.2` | 2026-10-11 | README only: same-path parity (94.2%, TV 0.035) and Decision Index results (-0.57 pts vs BF16). |

## License

Apache-2.0, as the base model. All credit for the model goes to Cloudflare; this repo only changes the weight precision.
