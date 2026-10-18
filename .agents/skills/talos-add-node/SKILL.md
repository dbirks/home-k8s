---
name: talos-add-node
description: Add a second machine to the home Talos cluster as a GPU worker, including a Pascal-era NVIDIA card (proprietary driver), and convert a running Arch Linux box into a Talos node remotely by kexec-booting Talos then applying a worker config. Use when enrolling any new node, picking the NVIDIA extension for an older GPU, or fixing the cluster join endpoint.
---

# Add a node (and turn an Arch box into one)

STATUS: researched 2026-10-17, NOT yet run end-to-end. Update this file with what really happened.

## 0. Blocker first: the join endpoint is stale
The machine config says `cluster.controlPlane.endpoint: https://10.0.0.30:6443`, but the control plane is
`10.0.0.194` (DHCP) and 10.0.0.30 is some other device (port 6443 refused, different MAC). The existing
node does not care (kubelet uses KubePrism on localhost), but a NEW node joins through that endpoint, so
it would never join. Fix before enrolling anything:
1. DHCP reservation at the router for the control plane (MAC `30:56:0f:25:2d:0b`, keep 10.0.0.194) and
   one for the new box. Do not use a DNS name: Pi-hole runs inside the cluster.
2. Set `endpoint: https://10.0.0.194:6443` in `talos/talconfig.yaml` (and fix the stale `ipAddress`),
   regenerate, and `apply-config` to the control plane. The API server cert already covers the node IP
   (kubectl works against .194 today). Re-check with `curl -k https://10.0.0.194:6443/version` style tests.
   Alternative: a Talos VIP (`machine.network.interfaces[].vip`) on a free static IP outside the DHCP pool.

## 1. Pick the NVIDIA extensions by GPU generation
| GPU | kernel module extension | toolkit |
|---|---|---|
| Turing and newer (RTX 20xx+, Blackwell) | `nvidia-open-gpu-kernel-modules-lts` (what talos-210-73x runs) | `nvidia-container-toolkit-lts` |
| **Pascal / Maxwell / Volta** (GTX 10xx, P40, P100, Titan Xp/V, V100) | **`nonfree-kmod-nvidia-lts`** | `nvidia-container-toolkit-lts` |
- Open kernel modules do not support Pascal. 595 (the `-production` flavor) does not support it either:
  NVIDIA made **580 the last branch for Maxwell/Pascal/Volta** (LTS, supported to ~mid-2028). Talos v1.13.6
  ships `nonfree-kmod-nvidia-lts:580.167.08-v1.13.6` + toolkit `580.167.08-v1.19.1`. Kmod and toolkit
  driver versions must match, so use `-lts` for both. Never `-production` on Pascal.
- Image Factory schematic for that pair (verified 2026-10-17):
  `af8eb82417d3deaa94d2ef19c3b590b0dac1b2549d0b9b35b3da2bc325de75f7`.
- Same module-load patch as the PRO 6000 (`talos/patches/nvidia-module-options.yaml`: nvidia, nvidia_uvm,
  nvidia_drm, nvidia_modeset). Its `NVreg_DynamicPowerManagement=0x00` is harmless on Pascal.
- Software limits on Pascal (compute capability 6.x): **vLLM and SGLang need 7.0+, so they will not run.**
  CUDA 13 cannot build for Pascal, so images built only for CUDA 13 have no Pascal kernels. Use
  CUDA 12.x builds: llama.cpp (`ghcr.io/ggml-org/llama.cpp:server-cuda`), Ollama, whisper.cpp, TTS
  in PyTorch with a cu12x wheel. No FP8/FP4, little FP16 speed except P100.
- HAMi lists only "driver >= 440", so Pascal should work, but nobody has tested it here.

## 2. Repo changes for the new node
- `talos/talconfig.yaml`: add a node with `controlPlane: false`, its reserved IP, an `installDiskSelector`
  (by serial/model/size, NOT `/dev/sda`), the Pascal schematic above, and the module patch. Talos
  version must equal the cluster's (v1.13.6). Each node may have its own schematic.
- `talos/talsecret.sops.yaml` does not exist yet (README step "Bootstrap talsecret.sops.yaml from an
  existing running cluster"). Without it a regenerated worker would get NEW secrets and could not join.
  Do that step first, or hand-make the worker from `_newconfig/worker.yaml` (gitignored, same secrets).
- `nodeLabels` on the worker: `gpu: "on"` (HAMi's device plugin selects `gpu=on` +
  `pci-10de.present`; NFD adds the second automatically).
- Consider a `nodeTaints` entry (e.g. `dedicated=pascal:NoSchedule`) at first, so existing untainted pods
  do not drift onto a slower box; tolerate it only in workloads you mean to place there.
- `apps/gpu-power-limit.yaml`: the DaemonSet lands on every GPU node, and the `TARGET_WATTS` fallback is
  350 W, above any Pascal card's max. Add a `GPU_POWER_MAP` entry for the new card (by UUID or
  a unique name substring) and test with `DRY_RUN` first.
- GPU pods: pin with `nodeSelector` / `nvidia.com/use-gputype`. The PRO 6000 workloads stay put because
  their local-path PVCs carry node affinity.
- The workstation wedge watchdog and several skills assume one node (`{.items[0]}`). Re-check them, and
  run `gpu-wedge-watchdog.sh --learn` once the new card is up.

## 3. Arch box to Talos remotely (the "magic reboot")
Yes, it is possible, with no USB stick. Talos boots from RAM in maintenance mode and only writes the disk
when you apply a config. Until that moment a power-cycle returns the box to Arch, which makes kexec the
safe path.
Requirements: SSH + sudo on the Arch box, `kexec-tools`, Secure Boot off (lockdown blocks unsigned kexec),
wired DHCP. BACK UP the Arch disk: install wipes the target disk (other disks are left alone).
```bash
# on the Arch box
S=af8eb82417d3deaa94d2ef19c3b590b0dac1b2549d0b9b35b3da2bc325de75f7; V=v1.13.6
mkdir -p ~/talos && cd ~/talos
curl -LO https://factory.talos.dev/image/$S/$V/kernel-amd64           # ~20 MB
curl -LO https://factory.talos.dev/image/$S/$V/initramfs-amd64.xz     # ~566 MB (nvidia ext); too big for many ESPs, fine for kexec
CMD="$(curl -sL https://factory.talos.dev/image/$S/$V/cmdline-metal-amd64)"   # talos.platform=metal ... module.sig_enforce=1
sudo kexec -l kernel-amd64 --initrd=initramfs-amd64.xz --command-line="$CMD"
sudo systemctl kexec        # SSH drops; ~1 min later Talos maintenance mode is on DHCP (reservation helps)
```
```bash
# from the workstation
talosctl -n NEW_IP get disks --insecure                  # confirm the install disk before committing
talosctl apply-config --insecure -n NEW_IP -f worker.yaml   # installs to disk, reboots, joins
kubectl get nodes -o wide                                 # new node Ready; then check nvidia.com/gpu
talosctl -n NEW_IP read /proc/driver/nvidia/version       # 580.167.08, proprietary
```
- Skip `talos.config=URL` (fully hands-off) unless you accept serving the worker config, which carries
  the cluster join token and CA, over plain HTTP on the LAN. A one-line `apply-config` is the safer "magic".
- If kexec hangs (some GPUs and firmware dislike a warm handoff, especially with the nvidia module loaded):
  stop the display manager and `rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia` before kexec. Or boot
  the Image Factory ISO/USB once.
- After install there is no SSH. Everything goes through `talosctl` with `_newconfig/talosconfig`.
  Add the new IP to talosconfig `endpoints`/`nodes`.

## Sources
- Talos proprietary NVIDIA guide: docs.siderolabs.com/talos/v1.13 (hardware-and-drivers/nvidia-gpu-proprietary)
- Config acquisition order and `talos.config*`: docs.siderolabs.com/talos/v1.13 (system-configuration/acquire)
- 580 = last Pascal branch: archlinux.org/news/nvidia-590-driver-drops-pascal-support-main-packages-switch-to-open-kernel-modules/
- CUDA 13 arch drop: developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/
