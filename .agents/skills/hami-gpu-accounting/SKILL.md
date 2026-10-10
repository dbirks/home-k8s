---
name: hami-gpu-accounting
description: Audit and reason about HAMi vGPU accounting on the home-k8s single-node cluster — per-pod VRAM caps, the defaultMemory:0 whole-card claim trap, pinning a pod to one GPU when the node has different-sized cards, and slice arithmetic. Use when adding/reviving any GPU workload, when a pod unexpectedly blocks another, when a GPU pod is Pending with insufficient memory, or before plugging a second GPU into the node.
---

# HAMi GPU accounting on home-k8s

One node, `projecthami/hami:v2.9.0` (chart 2.9.0) via `prereqs/hami.yaml`, scheduler
`hami-scheduler`, `deviceSplitCount: 10`, `deviceMemoryScaling: 1` (no oversubscription —
46 GB host RAM, no swap). HAMi's `libvgpu.so` intercepts CUDA/NVML so a pod sees ONLY its
`nvidia.com/gpumem` quota as total VRAM: that is the hard isolation this cluster relies on.

## The trap that costs the whole card
A GPU pod that requests `nvidia.com/gpu` **without** `nvidia.com/gpumem` claims the ENTIRE
card in HAMi accounting → nothing else can co-schedule. Every GPU pod must set an explicit
`nvidia.com/gpumem`. Never set `nvidia.com/gpucores` — compute capping is unreliable on
Blackwell here.

```bash
.agents/skills/hami-gpu-accounting/scripts/audit-gpu-accounting.sh          # manifests + live pods
.agents/skills/hami-gpu-accounting/scripts/audit-gpu-accounting.sh --live    # live pods only (fast)
```
It flags manifests that request `nvidia.com/gpu` with no `gpumem` (incl. parked `replicas: 0`
ones — those are latent, they fire the moment someone revives them) and prints the live
node register + what each running pod actually reserved.

## Where the memory really went
```bash
kubectl get node -o jsonpath='{.metadata.annotations.hami\.io/node-nvidia-register}' | jq .
# [{"id":"GPU-9d6cf286-...","count":10,"devmem":97887,"devcore":100,"type":"NVIDIA RTX PRO 6000 ...","mode":"hami-core","health":true}]
kubectl get pod <p> -o jsonpath='{.metadata.annotations}' | tr ',' '\n' | grep -i hami   # per-pod binding
```
`health:false` or a device missing from that list = the card fell off the bus (see the
gpu-wedge-recovery skill). `count` is `deviceSplitCount` **per device**.

## Pinning when cards differ (second-GPU prep)
Annotations are read by the HAMi scheduler (verified in `pkg/device/nvidia/device.go` at the
pinned v2.9.0 tag) — type matching is **case-insensitive substring** on the registered name:
```yaml
metadata:
  annotations:
    nvidia.com/use-gputype: "PRO 6000"        # allow-list; "RTX" would match BOTH cards
    nvidia.com/nouse-gputype: "RTX 5090"      # deny-list
    nvidia.com/use-gpuuuid: "GPU-9d6cf286-..."  # stable; immune to a rename/re-enumeration
```
Slice arithmetic that bites: capacity is per-device, so a 2nd card doubles the advertised
units (10 → 20) and a 32 GB card split 10 ways yields **3.2 GB units** — a pod asking for
`gpumem: 4096` can NEVER land on it. Don't raise `deviceSplitCount` to paper over that; pin.
Placement default is set explicitly in `prereqs/hami.yaml` to
`gpuSchedulerPolicy: binpack` (chart default is `spread`, which would scatter small pods
across both cards and fragment the new one). Treat that only as a tie-breaker — anything
size-sensitive must pin.

## The single-node rollout trap (this bit us on 2026-10-10, #146/#147)
The chart hard-codes a **required** `podAntiAffinity` on `kubernetes.io/hostname` for
`hami-scheduler` and renders **no** `spec.strategy`, so the Deployment gets the API default
`RollingUpdate, maxSurge 25%`. On a one-node cluster the surged pod can never be placed →
```
Helm upgrade failed ... timeout waiting for: [Deployment/kube-system/hami-scheduler status: 'InProgress']
```
Flux **rolls the change back** (so your edit silently vanishes) and `prereqs` stays unhealthy,
which blocks `infra` then `apps` via `dependsOn` — the entire repo stops reconciling, 5 min per
retry. Symptom to look for: `flux get kustomization` shows `prereqs` "Reconciliation in progress"
and a `Pending` hami-scheduler pod, and no new commits are being applied anywhere.

```bash
.agents/skills/hami-gpu-accounting/scripts/hami-rollout-guard.sh --check     # strategy + wedge + chain
.agents/skills/hami-gpu-accounting/scripts/hami-rollout-guard.sh --enforce   # strategy.type=Recreate
.agents/skills/hami-gpu-accounting/scripts/hami-rollout-guard.sh --unstick   # recovery for a CONFIRMED wedge
.agents/skills/hami-gpu-accounting/scripts/hami-rollout-guard.sh --post      # caps/register/serving afterwards
```
`--unstick` refuses unless it sees a Pending scheduler pod **and** an UpgradeFailed/timeout HR, so
it cannot be fired at an unrelated outage. Recovery = force `Recreate` → resume the HR if suspended
→ `reconcile hr hami` → reconcile `prereqs`, `infra`, `apps` in that order. Cost is a few seconds
with no GPU scheduler: already-bound pods (Pennyroyal) keep running, new GPU pods can't bind.
`Recreate` is durable here precisely *because* the chart renders no `spec.strategy` — Helm's 3-way
merge never rewrites a field the chart doesn't own. Express it declaratively with HelmRelease
`spec.postRender` once the Flux CRDs are refreshed (the deployed `helmreleases` v2 CRD has no
`postRender` field even though helm-controller is v1.6.2 — the toolkit's CRDs are behind its
controllers, issue #148); until then this is intentional out-of-band drift, so don't "clean it up".

## Talos-specific overrides already baked in (don't lose them)
`/usr/local` is read-only → hook dir is `/var/lib/hami` (`global.gpuHookPath`,
`devicePlugin.libPath`, `monitor.ctrPath`); bundled kube-scheduler tag MUST equal the
control-plane minor (bump on every k8s upgrade or version skew breaks it);
`createRuntimeClass: false` (the `nvidia` RuntimeClass already exists); cert-manager for the
webhook cert under `scheduler.certManager`; `prometheus.enabled: false` (no ServiceMonitor CRD).
