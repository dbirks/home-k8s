# DSpark Training Run — 2026-07-06

Iteration 1: first pass on Qwen3.6-27B-NVFP4 using online training mode.
Training job: `dspark-train-qwen27-20260706` in namespace `default`.

## Config

| Parameter | Value | Notes |
|-----------|-------|-------|
| Verifier | `unsloth/Qwen3.6-27B-NVFP4` | 27B, NVFP4 quant |
| Speculator type | DSpark | DFlash backbone + Markov head + confidence head |
| Draft layers | 3 | ~1.4B params |
| Block size | 16 | 15 speculative tokens/step |
| Target layer IDs | `1 16 31 46 61` | Verified from z-lab DFlash config.json |
| Markov rank | 256 | Vanilla bigram head |
| Loss | `{"ce": 0.1, "tv": 0.9}` | TV dominates (acceptance rate), CE for stability |
| Dataset | sharegpt 5K samples | seq_len=8192 |
| Epochs | 5 | checkpoint every 25% of epoch |
| LR | 3e-4 | |
| vLLM GPU util | 0.35 | ~33.6 GB for model loading |
| Training mode | Online | `--on-missing generate --on-generate delete` |

## Timeline

| Time | Event |
|------|-------|
| 2026-07-07 02:30 UTC | Committed replicas=0 for vllm-qwen27 + vllm-qwen4b, created dspark-train-20260706.yaml |
| 2026-07-07 02:30 UTC | Flux reconciled at `71949bf` — both vLLM deployments scaled to 0/0 |
| 2026-07-07 02:32 UTC | First job pod `zpv8w` Pending — FailedScheduling: Insufficient memory (32Gi request) |
| 2026-07-07 02:35 UTC | Fixed memory request 32Gi→8Gi, pushed `0f12863`, deleted old job |
| 2026-07-07 02:57 UTC | Pod `kz686` Running — `pip install speculators` started |
| 2026-07-07 03:05 UTC | **CRASH** — `set -euo pipefail` tripped: scripts not in pip wheel, `git` not in vllm image |
| 2026-07-07 03:05 UTC | Root cause: `speculators v0.6.0` wheel doesn't bundle `scripts/`; `git` absent in vllm image |
| 2026-07-07 03:05 UTC | Job hit `BackoffLimitExceeded` after 5 container restarts in ~5 min |
| 2026-07-07 03:08 UTC | Fix: replace `git clone` fallback with `curl` downloads of individual scripts from GitHub raw |
| 2026-07-07 03:08 UTC | Pushed `3710258`, deleted failed job, triggered Flux reconcile |
| 2026-07-07 03:12 UTC | Pod `wm47q` Running — `pip install speculators` started |
| 2026-07-07 03:13 UTC | `speculators v0.6.0` installed OK; scripts downloaded via curl |
| 2026-07-07 03:13 UTC | `prepare_data.py` skipped — `token_freq.pt` already on PVC from first run |
| 2026-07-07 03:13 UTC | vLLM (port 8001, util=0.35) starting — loading Qwen3.6-27B-NVFP4 |
| | vLLM healthy |
| | Training started |
| | Epoch 1/5 checkpoint (25%) |
| | Epoch 1/5 checkpoint (50%) |
| | Epoch 1/5 checkpoint (75%) |
| | Epoch 1 complete |
| | ... |
| | Training complete |

## Log Excerpts

### speculators install
```
(to be filled)
```

### prepare_data.py output
```
(to be filled)
```

### vLLM startup
```
(to be filled)
```

### train.py initial output
```
(to be filled)
```

### Epoch 1 metrics
```
(to be filled)
```

## Metrics Per Checkpoint

| Checkpoint | Epoch | acceptance_rate | accept_len | tv_loss | ce_loss |
|------------|-------|----------------|------------|---------|---------|
| | | | | | |

## Issues Encountered

### Issue 1: Memory request too large (32Gi)
- **Symptom:** First pod `zpv8w` stuck in Pending with `Insufficient memory`
- **Root cause:** Node has ~32GB total CPU RAM; 9GB already claimed by other pods; 32Gi request left no headroom
- **Fix:** Reduced requests to `8Gi` — GPU VRAM handles model tensors, CPU RAM only needs ~8Gi for dataloader workers

### Issue 2: `git` not installed in vllm image; scripts not in pip wheel
- **Symptom:** Pod `kz686` CrashLoopBackOff, BackoffLimitExceeded after ~5 min
- **Root cause:** `speculators v0.6.0` does not bundle `scripts/` in the PyPI wheel. The fallback `git clone` silently failed (git absent from `vllm/vllm-openai:v0.24.0`). `set -euo pipefail` then tripped when train.py was missing.
- **Fix:** Replace git clone with curl downloads of individual script files from `raw.githubusercontent.com/vllm-project/speculators/main/scripts`
- **Note:** `token_freq.pt` (from `prepare_data.py`) survived on the PVC — data prep step was completed before the crash, not wasted

## Next Steps After This Run

1. Benchmark: `python scripts/evaluate/evaluate.py throughput --target http://vllm.hoam.lan --dataset RedHatAI/speculator_benchmarks`
2. Compare `acceptance_length` vs z-lab DFlash baseline (HumanEval: 8.9, MBPP: 7.6)
3. If acceptance_length < 4.0 at epoch 3: try `--loss-fn '{"tv": 1.0}'` or lower LR
4. If good results: upload to HuggingFace, update `vllm.yaml` to `method: dspark`
5. Future iterations: more data (ultrachat 200K or Magpie-Qwen3-Pro), more draft layers (5)
