---
name: talos-add-node
description: Add a second machine to the home Talos cluster as a GPU worker, including a Pascal-era NVIDIA card (proprietary driver), and convert a running Arch Linux box into a Talos node remotely by kexec-booting Talos then applying a worker config. Use when enrolling any new node, picking the NVIDIA extension for an older GPU, or fixing the cluster join endpoint.
---

# Add a node (and turn an Arch box into one)

STATUS: DONE 2026-10-17 for dave-droid (GTX 1070, 64 GB RAM, 10.0.0.195). Files: talos/patches/dave-droid-worker.yaml,
talos/render-dave-droid.sh, the dave-droid entry in talos/talconfig.yaml. Lessons are folded in below.

## 0. The join endpoint (fixed 2026-10-17)
The endpoint was a stale `https://10.0.0.30:6443`: dave-droid's old reservation (`talos-eol-mza`), from when that
box WAS the Talos node. Now: EdgeRouter reservations talos-210-73x = .194, dave-droid = .195, pool shrunk to
.100-.193 (ISC dhcpd does not protect a fixed address inside the pool), endpoint = `https://10.0.0.194:6443`.
Router: `ssh ubnt@10.0.0.1` (EdgeOS 3; creds are the box default, ask David), configure with
`/opt/vyatta/sbin/vyatta-cfg-cmd-wrapper begin|set|delete|commit|save|end`; copy `/config/config.boot` first.

**Changing `cluster.controlPlane.endpoint` on a live control plane is NOT harmless (~8 min outage seen):**
- It re-renders kube-apiserver with a NEW `--service-account-issuer` and `--api-audiences`. Every existing pod
  token (old issuer) then fails "invalid bearer token": Flux, HAMi plugin, cert-manager crash-loop. Fix: accept both:
  `cluster.apiServer.extraArgs: {service-account-issuer: [NEW, OLD], api-audiences: "NEW,OLD"}` (list form works
  on 1.13; first = issuing). Do it IN THE SAME PATCH as the endpoint change next time.
- Both apiserver restarts stalled: kubelet sat on `mounted volumes=[...]: context deadline exceeded` for the new
  static pod (its volume manager was blocked on PVC lookups through the dead API). `talosctl service kubelet
  restart` brought the API back in ~40 s each time. KubePrism then reported unhealthy for ~4 min and healed itself.
- Moving a machine's reservation makes it NAK on renewal; if the new IP is still held by another device the client
  declines it and NetworkManager can stay offline for 10+ min. dave-droid needed a manual reboot.

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
- HAMi on Pascal WORKS (2026-10-17): a 4000 MiB slice showed as 4000 MiB in `nvidia-smi` in-pod (it also
  prints a harmless-looking `ERROR: init 250 result=11`). HAMi limits CUDA only, not Vulkan.
- Vulkan works in pods on Talos: the nvidia CDI spec injects `/etc/vulkan/icd.d/nvidia_icd.json` plus
  libGLX_nvidia/libnvidia-glvkspirv. Set `NVIDIA_DRIVER_CAPABILITIES=all` and
  `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` (no Mesa llvmpipe fallback). Worked example:
  apps/qwen3-asr.yaml (speech.cpp Vulkan build, Qwen3-ASR 1.7B Q6_K, 0.41 s for 5 s / 1.17 s for 30 s warm).

## 2. Repo changes for the new node
- `talos/talconfig.yaml`: add a node with `controlPlane: false`, its reserved IP, an `installDiskSelector`
  (NOT `/dev/sda`; SATA disks may report NO `serial` in `talosctl get disks`, the serial sits inside `wwid`, so
  use `wwid: "*SERIAL"`; a non-matching selector is rejected at apply time and nothing is written), the Pascal schematic above, and the module patch. Talos
  version must equal the cluster's (v1.13.6). Each node may have its own schematic.
- `talos/talsecret.sops.yaml` does not exist yet (README step "Bootstrap talsecret.sops.yaml from an
  existing running cluster"). Without it a regenerated worker would get NEW secrets and could not join.
  What we did: patch `_newconfig/worker.yaml` (gitignored, SAME secrets as the live CP: compare sha256 of
  cluster.token / cluster.ca.crt / machine.token with `talosctl get machineconfig`). A strategic-merge patch cannot
  delete keys, so the render script drops `machine.install.disk` and swaps HostnameConfig `auto` for `hostname`.
- `nodeLabels` on the worker: `gpu: "on"` (HAMi's device plugin selects `gpu=on` +
  `pci-10de.present`; NFD adds the second automatically).
- Taint the new node (dave-droid: `dedicated=pascal:NoSchedule`) so existing untainted pods do not drift onto
  it. `machine.nodeTaints` ALONE FAILS on k8s 1.36: NodeRestriction says "node ... is not allowed to modify
  taints" and Talos' NodeApplyController sends labels+taints together, so the labels fail too. Register with
  the taint instead: `machine.kubelet.extraConfig.registerWithTaints: [{key, value, effect}]` (keep nodeTaints
  too). Put it in the config BEFORE the first boot. If you have to fix it afterwards, `kubectl delete node`
  plus a kubelet restart re-registers, BUT the node gets a NEW podCIDR, so flannel fails with "cni0 already has
  an IP address different from ..." until `talosctl reboot` of that node.
- The old base `_newconfig/worker.yaml` pins `kubelet:v1.35.0`: set `machine.kubelet.image` to the control
  plane's version (v1.36.2).
- DaemonSets that do NOT tolerate the taint (by design they stay off it): metallb-speaker, alloy. HAMi's
  device plugin got an explicit toleration (prereqs/hami.yaml).
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
wired DHCP. BACK UP the Arch disk: install wipes the target disk (other disks are left alone). dave-droid's
backup (home minus caches/weights + /etc, 3.1 GB) is on spruce-goose at ~/backups/dave-droid-2026-10-17.
Measured: kexec to maintenance API ~60 s. The initramfs is zstd despite the `.xz` name (`zstd -t` to check).
Before kexec: `sudo modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia` (nothing must hold the GPU), then
`sudo systemd-run --on-active=3 systemctl kexec` so the SSH command returns first.
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
