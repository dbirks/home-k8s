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

> **Status:** first pass (`v0.1`). Agrees with BF16 on 93.4% of held-out decisions (W4A4 sibling: 91.1%), so rounding activations costs about 2 points and most of the gap is weight rounding under plain `max` calibration. The target is 98%; tuned calibrations are in progress.

## Which Clef-Omni quant should I use?

| Model | Format | Size on disk | VRAM (weights) | Agrees with BF16 (top answer) | Decision Index 0.2.1 | Best for |
|---|---|---|---|---|---|---|
| [Cloudflare/clef-omni](https://huggingface.co/Cloudflare/clef-omni) | BF16 (original) | 70.8 GB | 60 GiB | reference | pending | Reference quality, ~70 GB VRAM |
| [dbirks/clef-omni-nvfp4](https://huggingface.co/dbirks/clef-omni-nvfp4) | NVFP4 W4A4 (ModelOpt) | 22.0 GB | 60 GiB | 91.1% | pending | Blackwell, native FP4 engines |
| [dbirks/clef-omni-nvfp4a16](https://huggingface.co/dbirks/clef-omni-nvfp4a16) | NVFP4 W4A16 (ModelOpt) | 22.0 GB | 60 GiB | 93.4% | pending | Blackwell / vLLM Marlin, higher fidelity than W4A4 |
| [dbirks/clef-omni-int4](https://huggingface.co/dbirks/clef-omni-int4) | INT4 W4A16 g128 (AutoRound) | pending | pending | pending | pending | Any recent NVIDIA GPU, smallest VRAM |

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


## Version history

Each version is an annotated git tag on this repo (`revision="v0.1"` etc. pins it). Newest last.

| Version | Date | Notes |
|---|---|---|
| `v0.1` | 2026-10-10 | First release: ModelOpt W4A16_NVFP4, experts-only, weight-only `max` calibration (no calibration data). 93.4% top-answer agreement with BF16 (mean TV 0.043). Ships serve_clef.py. |

## License

Apache-2.0, as the base model. All credit for the model goes to Cloudflare; this repo only changes the weight precision.
