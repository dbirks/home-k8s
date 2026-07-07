# DSpark Training Pipeline — Technical Reference

Deep dive into how DSpark speculative decoding works internally, based on reading
the `vllm-project/speculators` source. Relevant to training a Qwen3.6-27B checkpoint
via `apps/dspark-train.yaml.hold`.

---

## 1. What DSpark Is Architecturally

DSpark is a speculative decoding draft model with three stacked components on top of
a frozen verifier:

**DFlash backbone** — A Qwen3-architecture transformer (3–6 layers typically). Takes the
verifier's intermediate hidden states and produces base draft logits.

**Markov head** (`--markov-rank 256`) — Low-rank logit-bias factored as
`B = W1[prev_token] @ W2^T`, where `W1 ∈ R^{|V_verifier| × rank}` embeds the previous
draft token and `W2 ∈ R^{|V_draft| × rank}` projects to a vocabulary-sized bias. This
adds a bigram (first-order Markov) correction to the backbone logits conditioned on what
was just drafted.

**Confidence head** (`--enable-confidence-head`) — A single linear layer
`R^{input_dim} → R^1` that predicts per-position acceptance probability as a scalar
logit. At inference its sigmoid output gates dynamic draft-length decisions.

DSpark inherits from DFlash which inherits from the base `SpeculatorModelConfig`. The
config key is `speculators_model_type = "dspark"`.

---

## 2. Block-Based Drafting Structure

The fundamental unit is the **block**. With `--block-size 8`:

```
Block k: [anchor | draft_1 | draft_2 | draft_3 | draft_4 | draft_5 | draft_6 | draft_7]
               ↑                                                              ↑
         position 0: real input token (never predicted)    positions 1–7: what DSpark drafts
```

The draft model processes up to `--max-anchors 3072` blocks per forward pass, giving
`3072 × 8 = 24,576` total positions. Anchor positions are selected from assistant turns
(`loss_mask == 1`). Each anchor gets a block of `block_size` mask tokens appended.

---

## 3. How vLLM Extracts Hidden States

`launch_vllm.py` wraps `vllm serve` with two JSON configs injected as flags:

```json
speculative_config = {
  "method": "extract_hidden_states",
  "num_speculative_tokens": 1,
  "draft_model_config": {
    "hf_config": {
      "eagle_aux_hidden_state_layer_ids": [2, 32, 61, 64]
    }
  }
}
kv_transfer_config = {
  "kv_connector": "ExampleHiddenStatesConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {"shared_storage_path": "/output/hidden-states"}
}
```

vLLM runs the full verifier forward pass and collects hidden states at the specified
layers. The `ExampleHiddenStatesConnector` writes them to disk as `hs_{idx}.safetensors`.
The API response's `kv_transfer_params.hidden_states_path` points to that file — tensors
never travel over the network.

**What's in each `.safetensors` file:**

```
hidden_states: [seq_len, num_target_layers, hidden_size]
              e.g. [4096, 4, 5120] for Qwen3.6-27B with 4 target layers
token_ids:    [seq_len]
```

The training data pipeline further extracts:
- `hidden_states` — concatenated across layers, flattened: `[seq_len, num_layers * hidden_size]`
- `verifier_last_hidden_states` — only the final target layer: `[seq_len, hidden_size]`
- `input_ids`, `loss_mask` (1 = assistant, 0 = user/system), `position_ids`

---

## 4. target-layer-ids — Why They Must Match

**Formula:** `[2, num_layers // 2, num_layers - 3, num_layers]`

For Qwen3.6-27B (64 layers): **`2 32 61 64`**

This value must be **identical** in both `launch_vllm.py` and `train.py`. The draft
model's `fc` projection layer is sized `len(target_layer_ids) × hidden_size → hidden_size`.
Mismatch = shape error at load time.

`--include-last-layer` defaults to True, which appends `num_layers` automatically.
If you pass `--target-layer-ids 2 32 61`, it silently becomes `2 32 61 64`.
Best to pass all 4 explicitly to avoid confusion.

---

## 5. The Backbone Forward Pass Step by Step

**Step 1 — Anchor selection**
`select_anchors(loss_mask, max_anchors, block_size)` finds positions where
`loss_mask == 1`. Returns `anchor_positions: [num_anchors]` and a padding mask.

**Step 2 — Build block indices**
`get_base_indices_for_anchored_blocks(anchor_positions, block_size)` produces
`anchored_block_indices: [num_anchors × block_size]`. For anchor at position `p`:
`[p, p, p, p, p, p, p, p]` — repeated `block_size` times. These are the KV positions
the block's query positions attend to.

**Step 3 — Build mask tokens**
```python
mask_token_ids = full([1, num_anchors * block_size], mask_token_id)
mask_token_ids[:, ::block_size] = input_ids[:, anchor_positions]  # position 0 = real anchor
noise_embedding = embed_tokens(mask_token_ids)
```
Block position 0 gets the real anchor token; all other positions get the mask embedding.

**Step 4 — FC projection of verifier hidden states**
```python
fc_output = fc(hidden_states)   # [1, seq_len, num_layers*hidden_size] → [1, seq_len, hidden_size]
fc_output = hidden_norm(fc_output)
```
This is added into each draft decoder layer as a conditioning signal (analogous to
cross-attention, but implemented as an additive residual).

**Step 5 — Anchored-block attention mask**
Custom `FlexAttention` mask where:
- Each query block position attends to all KV positions up to and including its anchor
- Each query block position also attends to all prior positions within the same block
- Blocks do NOT attend to each other's mask positions

Position 0 of block k → attends to `[0..anchor_k]`
Position 1 of block k → attends to `[0..anchor_k]` ∪ `[block_k_pos_0]`
Position 7 of block k → attends to `[0..anchor_k]` ∪ `[block_k_pos_0..6]`

**Step 6 — Draft transformer layers**
`noise_embedding` passes through N Qwen3 decoder layers, each receiving:
- `hidden_states`: growing draft representation
- `target_hidden`: FC-projected verifier hidden states (conditioning)
- `attention_mask`: the anchored-block mask
- `position_embeddings`: RoPE from original sequence positions

**Step 7 — Compute soft targets from verifier**
```python
verifier_logits = verifier_lm_head(verifier_norm(verifier_last_hidden_states))
verifier_logits = torch.roll(verifier_logits, 1, dims=1)  # shift: logits[i] predicts token[i+1]
targets = verifier_logits[:, anchored_block_indices]       # gather at block positions
```
Targets are the verifier's **soft logit distributions** (not one-hot). This is knowledge
distillation — the draft learns to mimic the full verifier distribution.

`verifier_lm_head` and `verifier_norm` weights are loaded from the frozen verifier
checkpoint and are **excluded** from the DSpark checkpoint save.

**Step 8 — Project to draft logits**
```python
hidden = norm(noise_embedding)
logits = lm_head(hidden)    # [1, num_anchors*block_size, draft_vocab_size]
```

**Step 9 — Mask out anchor positions from loss**
```python
aligned_loss_mask = loss_mask[:, anchored_block_indices]
aligned_loss_mask[:, ::block_size] = 0   # position 0 never predicted
```

---

## 6. Markov Head Forward Pass

After the backbone returns `(hidden, logits, targets, loss_mask, block_indices)`:

```python
block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block_size)

# Previous token at each position (with teacher forcing during training)
prev_token_ids = cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)

# Embed previous tokens: W1[prev_token] → [num_blocks, block_size, markov_rank]
prev_emb = markov_w1(prev_token_ids)

# Compute bigram bias: W2(prev_emb) → [num_blocks, block_size, draft_vocab]
markov_bias = markov_w2(prev_emb)

logits = (logits.view(num_blocks, block_size, -1) + markov_bias).view(1, T, -1)
```

**Three Markov head variants:**
- `vanilla` — pure bigram: `W2(prev_emb)`, no hidden state influence
- `gated` — `sigmoid(gate_proj([hidden; prev_emb])) * prev_emb` through W2, hidden state modulates Markov strength
- `rnn` — GRU-like recurrence across the 8 positions within a block; bias is history-dependent

**Teacher forcing caveat:** During training, `prev_token_ids` are ground-truth tokens.
During inference, they are previously *sampled* draft tokens. This is the standard
exposure bias issue. `--use-off-policy-tokens` exists to mitigate it.

---

## 7. Confidence Head Forward Pass

```python
if confidence_head_with_markov:
    conf_features = cat([hidden_blocks, prev_emb.to(hidden_blocks.dtype)], dim=-1)
    # shape: [num_blocks, block_size, hidden_size + markov_rank]
else:
    conf_features = hidden_blocks
    # shape: [num_blocks, block_size, hidden_size]

confidence_logits = confidence_head.proj(conf_features).squeeze(-1).reshape(1, T)
# shape: [1, num_anchors * block_size]  — pre-sigmoid scalar per position
```

---

## 8. Loss Functions

### Draft logit loss (`compound_loss`)

Applied between draft `logits` and soft `targets` (verifier distributions):

| Key | Loss | Notes |
|-----|------|-------|
| `tv` | Total variation: `1 - Σ min(p_draft, p_verifier)` | Directly minimizes `1 - acceptance_rate`. Most theoretically motivated. Can be unstable alone. |
| `ce` | Cross-entropy: `H(p_verifier, log p_draft)` | Stable gradient, spreads mass broadly |
| `kl_div` | `KL(p_verifier ∥ p_draft)` — mass-covering | |
| `rkl` | `KL(p_draft ∥ p_verifier)` — mode-seeking | |

**Recommended:** `{"ce": 0.1, "tv": 0.9}` — TV for acceptance rate alignment, CE for stability.

### Positional decay

```python
decay = exp(-max(pos - 1, 0) / gamma)   # gamma=4.0 default
# pos 0 → weight 0 (anchor, never predicted)
# pos 1 → weight 1.0
# pos 2 → weight 0.78
# pos 4 → weight 0.47
# pos 7 → weight 0.22
```

Earlier positions matter more; they have higher acceptance probability and contribute
more to throughput. Later positions in the block are speculative in both senses.

### Confidence BCE loss

```python
# Analytical acceptance rate per position:
c_star = Σ_v min(softmax(logits), softmax(targets))   # [1, T]

# BCE against this analytical target:
conf_loss = BCE(confidence_logits, c_star)
conf_loss_mean = masked_decayed_mean(conf_loss, loss_mask, pos_idx, decay_fn)
```

The confidence target is computed analytically — no sampling needed. BCE trains the
confidence head to be a calibrated predictor of actual acceptance probability.

**Total loss:** `L = compound_loss + confidence_head_alpha × conf_loss`

---

## 9. What "Acceptance Rate" Means

With the standard max-entropy speculative sampling scheme, when the draft proposes
token `t` and distributions are `p_D` (draft) and `p_V` (verifier):

- Accept with probability `min(1, p_V(t) / p_D(t))`
- Expected acceptance rate: `E_{t∼p_D}[min(1, p_V/p_D)] = Σ_t min(p_D(t), p_V(t)) = 1 - TV(p_D, p_V)`

**accept_len** (the real throughput metric):
```
accept_len = 1 + Σ_{k=1}^{block_size-1} Π_{j=1}^{k} accept_rate_j
```
The `1` is the anchor. Each subsequent position multiplies the probability that all
prior positions also accepted. At 0.9 accept_rate per position with block_size=8:
```
accept_len ≈ 1 + 0.90 + 0.81 + 0.73 + 0.66 + 0.59 + 0.53 + 0.48 ≈ 5.7 tokens/step
```
vs 1.0 tokens/step without speculative decoding — a theoretical ~5.7× throughput gain.

Metrics computed in training (no inference needed):
```python
draft_p  = softmax(logits.float(), dim=-1)
target_p = softmax(targets.float(), dim=-1)
accept_rate = torch.minimum(draft_p, target_p).sum(dim=-1)   # per position
```

### DSpark vs DFlash acceptance rates (Qwen3-8B benchmark)

| Method | Acceptance Rate |
|--------|----------------|
| DFlash | 0.309          |
| DSpark | 0.415          |
| **Gain** | **+34%**   |

---

## 10. Online vs Offline Training

### Offline
1. Run `launch_vllm.py` to generate all hidden states with `--on-generate cache`
2. Stop vLLM to free VRAM
3. Train against cached `.safetensors` files with `--on-missing raise`

Pros: No generation overhead during training; full GPU for training.
Cons: Huge disk requirement (500GB–3TB for large datasets at Qwen3.6's `hidden_size=5120`);
distribution fixed to a single forward pass.

### Online (our setup)
vLLM and trainer share the GPU concurrently via CUDA time-slicing.
- `--on-missing generate` — trainer POSTs to vLLM endpoint when a sample is missing
- `--on-generate delete` — files deleted after loading (zero disk accumulation)
- `--on-generate cache` — files kept; subsequent epochs reuse them (hybrid mode)

The dataloader uses multiple workers with prefetch to hide vLLM generation latency.

---

## 11. Complete Flag Reference

| Flag | Controls |
|------|----------|
| `--num-layers` | Draft transformer depth. 3 = ~1.4B params, 5 = ~2B. More capacity but slower. |
| `--block-size` | Tokens per block. `block_size - 1` draft positions produced per step. |
| `--max-anchors` | Max blocks per training forward pass. Controls memory/throughput. |
| `--markov-rank` | Low-rank dim for bigram correction. 0 = disable Markov head. |
| `--markov-head-type` | `vanilla` / `gated` / `rnn`. |
| `--enable-confidence-head` | Adds per-position acceptance predictor. |
| `--confidence-head-with-markov` | Confidence head also uses Markov prev-token embedding. |
| `--loss-fn` | JSON dict of loss weights, e.g. `{"ce": 0.1, "tv": 0.9}`. |
| `--confidence-head-alpha` | Weight of BCE confidence loss term. |
| `--target-layer-ids` | Verifier layers to extract. Must match `launch_vllm.py`. |
| `--dflash-decay-gamma` | Positional decay rate. Higher = shallower decay. Default 4.0. |
| `--checkpoint-freq` | Float. 1.0 = once/epoch, 0.25 = 4×/epoch. |
| `--on-missing` | `generate` / `skip` / `warn` / `raise` |
| `--on-generate` | `cache` / `delete` |
| `--data-path` | Root dir; hidden states default to `{data_path}/hidden_states`. |
| `--hidden-states-path` | Override hidden states path explicitly. |
| `--save-path` | Checkpoint output dir. Resume is automatic on restart. |
| `--use-off-policy-tokens` | Sample draft tokens (vs teacher-forcing) to reduce exposure bias. |

---

## 12. Checkpoint Structure

A standard HuggingFace directory saved via `save_pretrained()`:

**`config.json`** — Full `DSparkSpeculatorConfig` (corrected for Qwen3.6-27B):
```json
{
  "speculators_model_type": "dspark",
  "architectures": ["DSparkSpeculator"],
  "block_size": 16,
  "aux_hidden_state_layer_ids": [1, 16, 31, 46, 61],
  "markov_rank": 256,
  "markov_head_type": "vanilla",
  "enable_confidence_head": true,
  "confidence_head_with_markov": true,
  "speculators_config": {
    "proposal_methods": [{"speculative_tokens": 15}],
    "verifier": {"name_or_path": "unsloth/Qwen3.6-27B-NVFP4"}
  }
}
```

Note: `block_size=16` → max `num_speculative_tokens=15` (block_size - 1). No `draft_vocab_size` field = full vocab (248320).

**`model.safetensors`** — Trained weights:
- `fc.weight`, `hidden_norm.weight` — verifier hidden state projection
- `layers.N.*` — draft Qwen3 transformer layers (N = 0..num_layers-1)
- `norm.weight`, `lm_head.weight`, `embed_tokens.weight`
- `markov_head.markov_w1.weight`, `markov_head.markov_w2.weight`
- `confidence_head.proj.weight`, `confidence_head.proj.bias`

**NOT saved:** `verifier_lm_head.weight`, `verifier_norm.weight` — reloaded from the
verifier at inference time, saving ~2GB in the checkpoint.

**NOT present** (full vocab mode): `d2t`, `t2d` vocab mapping tensors.

---

## 13. Loading into vLLM for Inference

Update `apps/vllm.yaml` speculative config:

```yaml
- "--speculative-config"
- '{"method":"dspark","model":"your-hf-repo/Qwen3.6-27B-DSpark","num_speculative_tokens":7}'
```

vLLM's speculative decoding engine will:
1. Load the DSpark config from the checkpoint
2. Reconstruct the `DSparkSpeculator` (3 Qwen3 layers + Markov + confidence heads)
3. Re-attach `verifier_lm_head` from the verifier
4. At each decode step:
   - Extract hidden states at `aux_hidden_state_layer_ids` from the verifier forward pass
   - Run `_backbone_forward` to get base logits
   - Markov head conditions on last drafted token, adds bigram bias
   - Confidence head optionally gates draft length
   - Propose `num_speculative_tokens` draft tokens to verifier for parallel verification
   - Vocab mappings `d2t`/`t2d` project between draft and verifier vocabularies

---

## 14. Qwen3.6-27B Specific Notes

- **64 transformer layers** (hybrid DeltaNet + standard attention)
- `hidden_size = 5120`, `intermediate_size = 17408`
- `vocab_size = 248320` (full Qwen3 vocabulary)
- `mask_token_id = 248070` (`<|extra_0|>` — used as block mask token)
- `Qwen3NextForCausalLM` implements `SupportsEagle3` ✅
- **Target layers (confirmed from z-lab DFlash config.json):** `1 16 31 46 61`
  (5 layers, evenly distributed — NOT the generic `[2, num//2, num-3, num]` formula)
- Draft model `fc` input dim: `5 × 5120 = 25600 → 5120`
- At 3 draft layers: ~1.4B params; z-lab used 5 for DFlash
- **`block_size = 16`** confirmed from z-lab (produces up to 15 speculative tokens/step)
- **Do NOT use `--draft-vocab-size`** — z-lab uses full vocab, no compression

## 15. Data Pipeline (prepare_data.py required before train.py)

The pipeline has a mandatory tokenization step before training:

```bash
# Step 1: Tokenize conversations, compute token frequencies
python3 scripts/prepare_data.py \
    --model unsloth/Qwen3.6-27B-NVFP4 \
    --data sharegpt \
    --output ./output/data \
    --max-samples 5000 \
    --seq-length 8192
# Outputs: output/data/*.arrow (tokenized), output/data/token_freq.pt

# Step 2: Start vLLM via launch_vllm.py (hidden state extraction)
# Step 3: Run train.py with --data-path ./output/data --on-missing generate
```

**Dataset options:**
- `sharegpt` — 5K general instruct, good fast baseline (~5 min prep)
- `ultrachat` — 200K diverse, longer prep
- Custom JSONL (field: `"conversations"` list of `{from, value}` dicts)
- Best for coding: `Magpie-Align/Magpie-Qwen3-Pro-200K-Filtered`

## 16. Benchmarking a Trained DSpark Checkpoint

```bash
# With vLLM running the new speculator:
python scripts/evaluate/evaluate.py throughput \
    --target http://vllm.hoam.lan \
    --dataset RedHatAI/speculator_benchmarks \
    --output-dir ./benchmark_results
```

**Key metrics:**

| Metric | Formula | Direction |
|--------|---------|-----------|
| `acceptance_length` | `1 + (accepted / drafts)` | ↑ higher |
| `acceptance_at_pos_i` | per-position breakdown | ↑ higher |
| `output_tps_median` | tokens/sec | ↑ higher |
| `itl_median_ms` | inter-token latency | ↓ lower |

**Published baselines (z-lab/Qwen3.5-27B-DFlash, block_size=16, @C=1):**

| Dataset | accept_len | Throughput | Speedup |
|---------|-----------|------------|---------|
| HumanEval | **8.9** | 572 tok/s | 6.2× |
| MBPP | 7.6 | 495 tok/s | 5.3× |
| GSM8K | 7.5 | 461 tok/s | 5.0× |
| MT-Bench | 5.6 | 350 tok/s | 3.8× |

**DSpark first-pass target:** `acceptance_length > 4.0` at epoch 3
(matches RedHatAI GLM-5.2 DSpark training result at that stage).

**Iteration workflow:**
1. Train 1–2 epochs → benchmark → check `acceptance_at_pos_i` for which positions decay fastest
2. If pos_1 acceptance < 0.7, consider `--loss-fn '{"tv": 1.0}'` (pure TV) or lower LR
3. If early positions good but later collapse, try `--num-layers 5` for more draft capacity
4. Compare against z-lab DFlash baseline: `{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash"}`
