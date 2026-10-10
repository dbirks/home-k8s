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

> **Status:** experimental first pass (`v0.1`, `max` calibration). It agrees with BF16 on 91.1% of held-out decisions; the target is 98%. See the table for higher-fidelity variants.

## Which Clef-Omni quant should I use?

| Model | Format | Size on disk | VRAM (weights) | Agrees with BF16 (top answer) | Decision Index 0.2.1 | Best for |
|---|---|---|---|---|---|---|
| [Cloudflare/clef-omni](https://huggingface.co/Cloudflare/clef-omni) | BF16 (original) | 70.8 GB | 60 GiB | reference | pending | Reference quality, ~70 GB VRAM |
| [dbirks/clef-omni-nvfp4](https://huggingface.co/dbirks/clef-omni-nvfp4) | NVFP4 W4A4 (ModelOpt) | 22.0 GB | 60 GiB | 91.1% | pending | Blackwell, native FP4 engines |

**Recommended for:**

- **clef-omni-nvfp4**: Blackwell engines with native FP4 tensor cores (W4A4 is the only variant that can run faster than BF16 there).
- **clef-omni** (Cloudflare BF16): the reference, if you have ~70 GB of VRAM.

*Size on disk* is the download. *VRAM (weights)* is what `serve_clef.py` holds on the GPU: the NVFP4 checkpoints are unpacked to BF16 there (transformers has no packed-NVFP4 kernels), so today only INT4 actually saves VRAM, and the NVFP4 files are for engines with native FP4. *Agrees with BF16* is the share of 258 held-out decision questions where the quant picks the same top answer as Cloudflare's BF16 model. *Decision Index* is the chance-corrected public index from the [Decision Index kit](https://github.com/apolinario/decision-index), edition 0.2.1 (the edition behind Cloudflare's published Clef-Omni results), on our hardware.

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

`serve_clef.py` unpacks the FP4 expert weights to BF16 on load (bit-exact) and, because this is a W4A4 checkpoint, also rounds every expert input to NVFP4 exactly the way ModelOpt calibrated it, so answers match the quantized model. Memory and speed are BF16-like. Native FP4 execution needs an engine with NVFP4 kernels (vLLM's ModelOpt NVFP4 MoE path on Blackwell) and a way to feed the thinker's hidden states into the Clef head, which no engine provides yet.

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

Parity: the 128 held-out records (258 questions; text from ultrachat_200k test_sft in Clef's state/questions schema, about 20% with synthetic image/video/audio, never used for calibration) are scored by the BF16 original and by the exported quant loaded through `serve_clef.py`. We report top-answer agreement and total variation between the per-option distributions. Benchmarks: the Decision Index reproduction kit (apolinario/decision-index), edition 0.2.1, driven over `/v1/systemone`, on one RTX PRO 6000 Blackwell.


## Version history

Each version is an annotated git tag on this repo (`revision="v0.1"` etc. pins it). Newest last.

| Version | Date | Notes |
|---|---|---|
| `v0.1` | 2026-10-10 | First release: ModelOpt NVFP4 W4A4, experts-only, `max` calibration on 512 Clef-format records. 91.1% top-answer agreement with BF16 (mean TV 0.055). Also tagged `v0.1-max`. |
| `v0.2` | 2026-10-10 | Docs only, weights unchanged: added `serve_clef.py` (PEP 723 `uv run` server that loads this checkpoint with the Clef head), the quant comparison table, and this version history. |

## License

Apache-2.0, as the base model. All credit for the model goes to Cloudflare; this repo only changes the weight precision.
