---
name: gpu-power-management
description: Set, verify, and persistently enforce per-GPU power limits on the home-k8s Talos node (RTX PRO 6000 Blackwell, and any second GPU). Use when a card needs capping for the GSP FULLCHIP_RESET crash mitigation, when adding a GPU and giving it its own watt budget, when a power limit "keeps reverting", or when auditing what nvidia-smi is actually set to.
---

# Per-GPU power limits on home-k8s

`nvidia-smi -pl <W>` on this node **does not stick** — something re-asserts the default
within ~1 min and reboots reset it. The fix is a GitOps DaemonSet that re-applies forever:
`apps/gpu-power-limit.yaml` (60s loop, `priorityClassName: system-node-critical`,
requests **no** `nvidia.com/gpu` so it never steals the allocation from a serving pod).

## Never run two enforcers
`apps/nvidia-power-cap.yaml` was a second, hourly writer of the same register and was
**retired to `.yaml.hold` on 2026-10-10**. On 2026-09-03 the 60s enforcer silently
overrode it and the card ran at the most transient-prone 600W while both claimed 400W.
If you revive it, delete one — two writers of one register is the bug, not the cure.

## Configuration (env on the DaemonSet)
| env | meaning |
|---|---|
| `TARGET_WATTS` | fallback limit for any GPU no matcher selects (currently `400`) |
| `GPU_POWER_MAP` | `"<matcher>=<watts>,..."`, **first match wins**; empty = everything gets `TARGET_WATTS` |
| `INTERVAL` | seconds between passes (60) |
| `HEARTBEAT_PASSES` | passes between state-table logs + `-pm 1` re-assert (30 ≈ 30 min) |
| `DRY_RUN` | set to log intended writes without touching the GPU — use it to trial a map |

`matcher` = GPU UUID or prefix (`GPU-9d6cf286`), a 0-based `nvidia-smi` index (`0`,
exact-field match only), or a case-insensitive **substring of the GPU name**.

**Trap:** `"RTX"` matches BOTH `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` and
`NVIDIA GeForce RTX 5090`. Use a substring unique to one card, or the UUID. Prefer UUID
matchers for anything load-bearing — index order is exactly what a FULLCHIP_RESET
re-enumeration can change.

## Recipes
```bash
# what is actually applied, per card (the source of truth)
kubectl exec ds/gpu-power-limit -- nvidia-smi \
  --query-gpu=index,uuid,name,power.limit,power.max_limit,persistence_mode --format=csv

# trial a new map against the real card with ZERO writes (logs "DRY-RUN would set")
kubectl set env ds/gpu-power-limit DRY_RUN=1 'GPU_POWER_MAP=PRO 6000=400,RTX 5090=300'
kubectl logs -l app=gpu-power-limit --tail=20
# Undo out-of-band drift the GitOps way (this DS lives in the apps Kustomization, so a
# reconcile re-applies the committed env). Do NOT use `set env KEY-` to undo — that
# deletes the key, and if the committed value was the default it silently changes
# behaviour instead of restoring it.
flux -n flux-system reconcile kustomization apps --with-source
# ...then confirm the live env matches the manifest again:
kubectl get ds gpu-power-limit -o jsonpath='{.spec.template.spec.containers[0].env}' | tr ',' '\n' | grep -E "DRY_RUN|GPU_POWER_MAP|TARGET_WATTS"

# add a second card's budget (edit the manifest, never leave the trial env behind)
#   GPU_POWER_MAP: "PRO 6000=400,RTX 5090=300"

# which pod is writing the register? (find rogue enforcers)
grep -rl "nvidia-smi -pl" apps/ infra/ prereqs/ --include=*.yaml*
```

## Why 400 W and not more
The card defaults to 600 W and FULLCHIP-resets every ~18 min under load at that ceiling
(issue #46, GSP dI/dt transients). 400 W is the documented mitigation; 350 W is the next
lever if resets return, 300 W after that. Decode throughput is memory-bandwidth-bound so
600→400 W cost almost nothing (clocks 2020→2520 MHz, tok/s ~flat). **Do not raise it.**
For a GeForce RTX 5090 (575 W TGP) the same dI/dt class of crash is reported on 50-series,
so start it at ~300-350 W and lift deliberately.

## Verify your change before merging
`scripts/test-power-map.sh` extracts the DaemonSet's `/bin/sh` payload, runs it against a
fake `nvidia-smi`, and asserts exactly which `-pl` calls it would make (10 cases: single
card no-op, name/uuid/index matchers, the `RTX` trap, two-card split, unmatched-matcher
fallback, DRY_RUN). ~15 s, no GPU needed:
```bash
.agents/skills/gpu-power-management/scripts/test-power-map.sh
```
