---
name: gpu-wedge-recovery
description: Detect and recover a NVIDIA FULLCHIP_RESET GPU wedge on the home-k8s Talos node (RTX PRO 6000 Blackwell GSP crash, issue #46) — including the multi-GPU case where the node keeps advertising GPUs. Use when GPU pods land in ContainerStatusUnknown, the node advertises fewer GPUs, hami-device-plugin goes RunContainerError, dmesg is full of NV_ERR_GPU_IN_FULLCHIP_RESET, or you need to change/test the workstation watchdog that auto-power-cycles the node.
---

# GPU wedge detection + auto-recovery

**The wedge:** GSP firmware crashes (Xid 79/154) and the card falls off the bus. Symptoms:
node keeps its API up but advertises `nvidia.com/gpu: 0` (single-GPU node), every GPU pod
goes `ContainerStatusUnknown`, `hami-device-plugin` → `RunContainerError`
(`error getting device handle for index '0': Unknown Error`), dmesg is a wall of
`NV_ERR_GPU_IN_FULLCHIP_RESET`.

**A warm reboot does NOT fix it** (`load nvidia failed: no such device` — the card never
re-enumerates). Only a BMC power-cycle does:
```bash
NODE_IP=$(kubectl get node -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')
talosctl --talosconfig _newconfig/talosconfig -e "$NODE_IP" -n "$NODE_IP" reboot --mode powercycle
```
`--mode powercycle` is the key — it escalates to the BMC. Back in ~2-3 min; clean up dead
pods with `kubectl delete pods --field-selector=status.phase=Failed -A`.
Before reaching for it, check the power cap is right (see the gpu-power-management skill) —
a fixed-interval crash storm (~every 18 min) has meant the cap was being overridden.

## The watchdog (lives on David's workstation, NOT in-cluster)
```bash
~/.local/share/home-k8s-auto/gpu-wedge-watchdog.sh          # the live copy, fired every 2 min
systemctl --user list-timers gpu-wedge-watchdog.timer        # is it alive?
tail -40 ~/.local/share/home-k8s-auto/gpu-wedge-watchdog.log
```
Canonical tracked copy: `scripts/gpu-wedge-watchdog.sh` here. **Edit the tracked copy, then
install it** (keep a `.bak`): `cp -p ~/.local/share/home-k8s-auto/gpu-wedge-watchdog.sh{,.bak}`
then copy over and `systemctl --user restart gpu-wedge-watchdog.timer`.

Acting requires ALL of: `GPU_IN_FULLCHIP_RESET` in the node's dmesg (hard proof, non-negotiable),
a device that is actually unserviceable, persistence across a 120 s confirm window, and a
30 min cooldown. dmesg clears on the recovery reboot, so stale assertions can't cause a
double power-cycle.

Signals it uses (`gpu_unserviceable` / `missing_uuids`):
1. node `capacity nvidia.com/gpu == 0`, **or**
2. `hami-device-plugin` not `2/2` / in `RunContainerError|CrashLoopBackOff|StartError` —
   the reliable EARLY signal, capacity lags and can read a stale 10 for minutes, **or**
3. an expected GPU UUID has vanished from `nvidia-smi --query-gpu=uuid`.

Signal 3 is what makes this work with **more than one GPU**: on a 2-GPU node a single
wedged card leaves capacity > 0 and the plugin healthy on the survivor, so a cap==0-only
watchdog would silently stop recovering anything. It reads the live GPU list by
`kubectl exec` into the always-on privileged `ds/gpu-power-limit` pod. A *failed* query is
never treated as a missing GPU (node down ≠ card gone).

```bash
gpu-wedge-watchdog.sh --status   # node IP, capacity, expected vs live UUIDs, applied power caps
gpu-wedge-watchdog.sh --learn    # re-capture the expected GPU set (after ANY intentional hw change)
gpu-wedge-watchdog.sh --check    # evaluate + log only; exits 2 if it WOULD have fired. Never power-cycles.
```
Expected UUIDs persist in `~/.local/share/home-k8s-auto/gpu-uuids.list` (auto-learned on the
first run if absent). **After you physically add or move a card, run `--learn`** — otherwise
the removed card looks "missing" forever (harmless: it still needs dmesg proof to act, but
the signal becomes noise).

`scripts/test-watchdog.sh` runs the real script in `--check` mode with a mocked
`talosctl dmesg` and asserts the three gates: no-marker-but-missing-GPU → no action;
marker+missing-GPU → reaches the confirmed-wedge branch (rc=2, still refuses to act);
healthy set → no action. Takes ~3 min (two real 120 s confirm windows), cannot touch the node.
