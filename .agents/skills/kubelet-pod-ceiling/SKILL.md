---
name: kubelet-pod-ceiling
description: New pods sit Pending with "Too many pods" even though CPU/RAM/GPU are free, or you need to change a kubelet flag on the live Talos node. Use when scheduling fails on pod count, when adding yet another tailnet app, or when day-2 Talos machine-config tweaks seem needed.
---

# kubelet pod ceiling (this node is chronically near max-pods)

Single-node cluster; Talos defaulted `max-pods` to **110**, and every tailnet
app costs **two** pods (the app Deployment + a `ts-<ingress>` Tailscale proxy
StatefulSet pod the operator creates). With ~100 services, new deployments
routinely land Pending with:

    0/1 nodes are available: 1 Too many pods.

CPU/RAM/GPU look free — the binding constraint is the pod *count*.

## 1. Confirm

```bash
kubectl get node -o jsonpath='{.status.capacity.pods}{"\n"}'   # current ceiling (now 250)
kubectl get pods -A --field-selector=status.phase=Running --no-headers | wc -l
kubectl -n <ns> describe pod <pending> | tail   # FailedScheduling message
```

## 2. Free zombie slots first (safe, always do this)

Completed/Failed pods **hold** max-pods slots even though they run nothing.
This repo once had **128** of them (old rollout replicas + finished Jobs):

```bash
kubectl get pods -A --field-selector=status.phase=Succeeded -o \
  custom-columns=NS:.metadata.namespace,NM:.metadata.name --no-headers |
  while read ns nm; do kubectl delete pod -n "$ns" "$nm"; done
```

Also hunt orphan pods with no owner (`kubectl get pod X -o jsonpath='{.metadata.ownerReferences}'`
empty = manual leftover, usually safe to delete). Note: `prune: true` Flux will
NOT recreate manually-deleted completed Job pods — this is housekeeping, not GitOps drift.

## 3. Still short? The ceiling is raised via merge-style `patch mc`

`patches/kubelet-max-pods.yaml` (committed) sets `max-pods: "250"`. To apply
day-2 changes like this to the live node:

```bash
NODE_IP=$(kubectl get node -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')
talosctl --talosconfig _newconfig/talosconfig -e "$NODE_IP" -n "$NODE_IP" \
  patch mc -p "$(cat talos/patches/kubelet-max-pods.yaml | grep -v '^#')"
# -> "Applied configuration without a reboot" (kubelet restarts, pods keep running)
```

Gotchas (both cost time once, see talos/README.md):
- The live machine config is **multi-document**, so JSON6902 (`-p '[{"op":...}]'`)
  is rejected. Plain `machine:`-keyed merge YAML is what works.
- **Never** `talhelper gencommand apply | bash` blindly: `talsecret.sops.yaml`
  currently does NOT match the live cluster's CAs and a real apply swaps the
  PKI and breaks the cluster. `--dry-run` first, always. talhelper is render-only
  until the secret is re-bootstrapped.
- Mirror every live patch into `talos/patches/` + `talconfig.yaml` so it survives
  a future legitimate apply.

## 4. Prevent: budget pods for new tailnet apps

New `ingressClassName: tailscale` app = +2 pods (+1 transient local-path helper
on first PVC). Near the ceiling, the PVC helper itself can't schedule, and the
PVC (WaitForFirstConsumer) never binds — looks like a storage bug, is really
the same pod-count wall. Free/raise slots first.

Verify after: `kubectl get node -o jsonpath='{.status.capacity.pods}'` and the
pending pods should move to ContainerCreating within ~30s.
