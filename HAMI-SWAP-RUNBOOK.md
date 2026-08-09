# HAMi swap runbook — hard GPU memory partitioning (training 86GB / inference 8–12GB)

**Goal:** replace the stock NVIDIA k8s-device-plugin (even-only memory split) with
**HAMi**, so the single RTX PRO 6000 Blackwell can host the big training job AND an
always-on tiny vLLM (LFM2.5-2.6B) simultaneously, each with a **hard memory cap** that
can't OOM the other.

**Do this ONLY in an idle window** (no GPU workload running) — the swap removes and
re-adds the device plugin. Decided 2026-08-09; see memory `project_gpu_sharing_hami_plan`.

## Verified environment (talos-210-73x)
Talos **v1.13.6**, k8s **v1.36.2**, containerd 2.2.5, cert-manager present, CDI live at
`/var/run/cdi/nvidia.yaml`, driver 580.167.08 (open modules), node label
`feature.node.kubernetes.io/pci-10de.present=true`. Driver bump NOT needed (a newer 580
does not fix the Blackwell SM-util NVML gap; memory isolation works on 580.167.08).

## Staged files (already in repo, parked)
- `prereqs/hami.yaml.hold` — HAMi HelmRepository + HelmRelease (Talos-correct values).
- `apps/lfm2.yaml.hold` — LFM2.5-2.6B Deployment/Service/Ingress (HAMi `gpumem` stanza, `--enforce-eager`).
- Stock plugin to remove: `prereqs/nvidia-device-plugin.yaml`.

---

## Swap steps

1. **Quiesce GPU work.** Ensure the training Job has finished (or checkpointed then
   deleted) and all vLLM deployments are `replicas: 0` / scaled down. Confirm the GPU is
   idle: `kubectl exec -n kube-system <nvidia-power-cap-pod> -- nvidia-smi` shows ~0 MiB used.

2. **Commit 1 — remove the stock plugin.** Delete the HelmRelease + `nvidia-device-plugin-configs`
   ConfigMap from `prereqs/nvidia-device-plugin.yaml` (delete the file, or empty it). Commit + push.
   Flux prunes it → `kubectl get node -o json | jq '.status.allocatable'` shows `nvidia.com/gpu`
   gone / 0. Confirm the `nvidia-device-plugin` DaemonSet is deleted from kube-system.
   **Keep the old file in git history for rollback.**

3. **Verify chart version.** `helm repo add hami https://project-hami.github.io/HAMi/ && helm search repo hami/hami --versions | head` — confirm the chart semver for app v2.9.0+ and update `version:` in `prereqs/hami.yaml.hold` if it differs.

4. **Commit 2 — add HAMi.** `git mv prereqs/hami.yaml.hold prereqs/hami.yaml`. Commit + push.
   Flux installs HAMi (HelmRepository + HelmRelease). Watch:
   - `kubectl get pods -n kube-system | grep hami` → `hami-scheduler` and `hami-device-plugin` Running.
   - **`kubectl logs` the `kube-scheduler` container in the hami-scheduler pod** on first boot —
     if it crashloops on config apiVersion, that's the (low-risk) thing to fix.
   - `kubectl get node talos-210-73x -o json | jq '.status.allocatable'` →
     `nvidia.com/gpu: "10"`, `nvidia.com/gpumem: "~98304"`, `nvidia.com/gpucores: "100"`.

5. **Smoke test the cap** (throwaway pod):
   ```
   kubectl run hami-smoke --rm -it --restart=Never --image=vllm/vllm-openai:v0.25.1 \
     --overrides='{"spec":{"runtimeClassName":"nvidia","containers":[{"name":"c","image":"vllm/vllm-openai:v0.25.1","command":["nvidia-smi"],"resources":{"limits":{"nvidia.com/gpu":1,"nvidia.com/gpumem":4000}}}]}}'
   ```
   Inside, `nvidia-smi` total VRAM should read ~4000 MiB (cap enforced) — NOT 96GB.

6. **Bring up inference.** `git mv apps/lfm2.yaml.hold apps/lfm2.yaml`; commit + push. Verify the
   pod Runs and `curl http://lfm2.hoam.lan/v1/models` returns `lfm2.5-2.6b`. If it OOMs on the
   slice, bump `nvidia.com/gpumem` (12000→14000) and/or lower `--gpu-memory-utilization`.

7. **Author the NEXT training run with an explicit `gpumem`** (see trap below).

---

## ⚠️ The one trap — `defaultMemory: 0` = 100% of the card
HAMi's `defaultMemory: 0` means **a pod that requests `nvidia.com/gpu: 1` WITHOUT
`nvidia.com/gpumem` gets the ENTIRE 96GB** in HAMi's accounting → the inference pod can
then never schedule. **Every** GPU pod must carry an explicit `nvidia.com/gpumem`.

Training pod stanza (trainer + its co-resident verifier share this one cap):
```yaml
resources:
  limits:
    nvidia.com/gpu: 1
    nvidia.com/gpumem: 86000    # MB (~84 GiB). NEVER omit. Do NOT set gpucores.
```
Re-check the in-job math: the verifier's `--gpu-memory-utilization 0.33` now means
0.33 × ~84GB ≈ 28GB (HAMi hooks `cudaMemGetInfo`), not 0.33 × 96GB. 86000 + 12000 = 98000 <
98304, leaving ~a little for the two CUDA contexts — tune if it's tight.

## Rollback
If HAMi advertises 0 GPUs (NVML init wall, like stock 0.19.3 hit on this driver):
`git revert` Commit 2, restore `prereqs/nvidia-device-plugin.yaml` (stock v0.17.1) from
history, push. Running pods aren't killed by a plugin swap; on one node just scale to 0/1.

## Known caveats (accepted)
- **Compute capping unusable on Blackwell** (`gpucores`): NVML per-process SM-util returns
  NOT_SUPPORTED; no driver fixes it. Leave training compute uncapped. Memory isolation is unaffected.
- **vLLM on a slice**: always `--enforce-eager` + conservative util (vLLM #40937, HAMi #1381).
- **SPOF**: hami-scheduler down → NEW gpu pods Pending (running ones fine). Webhook
  `failurePolicy: Ignore` → if the webhook is down, a gpu pod is admitted *without* isolation.
- **On every k8s minor upgrade**, bump `scheduler.kubeScheduler.image.tag` to match, or the
  bundled scheduler violates version-skew.
