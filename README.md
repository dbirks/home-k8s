# home-k8s

Single-node k8s cluster running on Talos Linux.

## Current setup

Running Talos v1.12.6 on a 256GB SSD.

Control plane: `10.0.0.30` (or DHCP-assigned, check after boot)

### Hardware

- 256GB SSD as install disk
- NVIDIA RTX 5090 (Blackwell) with open-source GPU drivers

### Talos install image

Factory image with NVIDIA open driver support:

```
factory.talos.dev/installer/036d341b186bfa76a1c0a545125bbd667908a09a50dfe5e7ab32cc93901b84a2:v1.12.6
```

Schematic uses `nvidia-open-gpu-kernel-modules` (required for RTX 50-series/Blackwell GPUs - the proprietary `nonfree-kmod-nvidia` does not support them).

Schematic config: `talos-nvidia-schematic.yaml`

Docs:
- https://www.talos.dev/v1.12/learn-more/image-factory/
- https://factory.talos.dev/
- https://www.talos.dev/v1.12/talos-guides/configuration/nvidia-gpu/

## Install process (April 2026)

### 1. Download the ISO

The ISO is built via Talos Image Factory with NVIDIA extensions baked in.

```bash
# If you need to regenerate the schematic ID:
SCHEMATIC_ID=$(curl -sX POST --data-binary @talos-nvidia-schematic.yaml \
  https://factory.talos.dev/schematics \
  -H "Content-Type: application/yaml" | jq -r '.id')

# Download the ISO
curl -LO "https://factory.talos.dev/image/${SCHEMATIC_ID}/v1.12.6/metal-amd64.iso"
```

### 2. Flash to USB drive

```bash
lsblk                    # identify your USB device
sudo dd if=metal-amd64.iso of=/dev/sdX bs=4M status=progress && sync
```

### 3. Boot and note the IP

Boot the server from the USB drive. Talos enters maintenance mode and displays the DHCP-assigned IP.

### 4. Generate Talos config

```bash
talosctl gen config home https://10.0.0.30:6443
```

### 5. Edit controlplane.yaml

Set the install disk (find the device name with `talosctl disks` after booting):

```yaml
install:
    disk: /dev/sda    # or /dev/nvme0n1, wherever the 256GB SSD is
    image: factory.talos.dev/installer/036d341b186bfa76a1c0a545125bbd667908a09a50dfe5e7ab32cc93901b84a2:v1.12.6
```

Enable scheduling on control plane (single-node cluster):

```yaml
cluster:
    allowSchedulingOnControlPlanes: true
```

NVIDIA kernel modules (should already be in the generated config if using the factory image):

```yaml
machine:
    kernel:
        modules:
            - name: nvidia
            - name: nvidia_uvm
            - name: nvidia_drm
            - name: nvidia_modeset
```

### 6. Apply config

```bash
talosctl apply-config --insecure -n <DHCP_IP> --file controlplane.yaml
```

Wait for install to complete. Screen shows `Installing`, then `Booting`.

### 7. Bootstrap the cluster

```bash
talosctl bootstrap -n <NODE_IP> -e <NODE_IP> --talosconfig ./talosconfig
```

### 8. Get kubeconfig

```bash
talosctl kubeconfig -n <NODE_IP> -e <NODE_IP> --talosconfig talosconfig
```

### 9. Verify NVIDIA GPU

```bash
talosctl get extensions          # should show nvidia-open-gpu-kernel-modules
talosctl dmesg | grep -i nvidia  # check for probe errors
kubectl run gpu-test --rm -it --restart=Never \
  --image=nvidia/cuda:12.8.0-base-ubuntu24.04 \
  --overrides='{"spec":{"runtimeClassName":"nvidia"}}' \
  -- nvidia-smi
```

## GitOps with Flux

Bootstrap Flux after the cluster is running:

```bash
export GITHUB_TOKEN=ghp_...
flux bootstrap github --owner dbirks --repository home-k8s --branch main --personal
```

This adds a deploy key to the repo and commits Flux manifests to `flux-system/`.

Flux reconciles in dependency order: `prereqs` -> `infra` -> `apps`

### Suspended apps (.yaml.hold)

Apps renamed to `.yaml.hold` are not reconciled by Flux. Current holds:

- `apps/happy-little-claude-coders.yaml.hold` - waiting for secrets setup
- `apps/happy-server.yaml.hold` - waiting for secrets setup
- `infra/nfs-client-provisioner.yaml.hold` - waiting for TrueNAS/NFS server

To re-enable, rename back to `.yaml` and commit.

## Storage

### NFS (when TrueNAS is available)

NFS provisioner connects to `10.0.0.43:/mnt/tetons/kubernetes` and creates a `nfs` StorageClass. Re-enable by renaming `infra/nfs-client-provisioner.yaml.hold` back to `.yaml`.

### Local storage (temporary)

While NFS is unavailable, a local-path-provisioner can provide storage backed by the SSD. See `infra/local-path-provisioner.yaml` (if created).

---

## Archive - old install notes

<details>
<summary>Talos v1.9.5 setup (Dec 2024)</summary>

Previous setup ran Talos v1.9.5 on a 63GB USB stick at `/dev/sda` with IP `10.0.0.30`.

Used proprietary NVIDIA drivers (`nonfree-kmod-nvidia`) which worked for the old GPU but do not support RTX 50-series.

Factory image:
```
factory.talos.dev/installer/0412a9a6369c0fb55e913cdfcbf4ad6ca3fab6e56ab71198ec4b58ad7e7a4ddd:v1.9.5
```

</details>

<details>
<summary>Kairos / k3s setup (pre-Talos)</summary>

### Current setup

- Kairos as the OS
  - Picked the Debian-based image
  - Switched to it after k3os stopped being developed
  - Operates basically the same as k3os as far as I can see... a purpose-built linux distro for running k3s

### Installation

#### k3os

Homepage: https://kairos.io
Github repo: https://github.com/kairos-io/kairos

Followed their quickstart guide roughly: https://kairos.io/docs/getting-started

Grabbed a recent .iso from [their releases](https://github.com/kairos-io/kairos/releases). Wrote to usb stick with:

```
sudo ddrescue ~/Downloads/kairos-debian-bookworm-standard-amd64-generic-v2.4.2-k3sv1.28.2+k3s1.iso /dev/sdb --force
```

Navigated to port 8080 on the machine from my laptop to reach the Kairos WebUI, and entered this as the cloud-config:

```
#cloud-config

users:
  - name: kairos
    ssh_authorized_keys:
      - github:dbirks

k3s:
  enabled: true
```

Took maybe 1-2 min to install after that.

After installation, connected to it over ssh (though my IP changed) with the `kairos` user.

To use kubectl from my laptop, I grabbed the kubeconfig from `/etc/rancher/k3s/k3s.yaml`. Substituted 127.0.0.1 with its DHCP address. Pointed the `KUBECONFIG` env var to the yaml file and connected successfully.

Some quick checks on the status when you're ssh'd on the machine:
```
sudo systemctl status k3s
sudo kubectl --kubeconfig /etc/rancher/k3s/k3s.yaml get pods -A
```

#### flux v2

Installed the cli with:

```
nix-env -i fluxcd
```

Then checked that prereqs were met:

```
flux check --pre
```

Ran the bootstrap (which required a `GITHUB_TOKEN` env var set up locally), which added a deploy key to my github repo, and committed some `kube-system` manifests:

```
flux bootstrap github --owner dbirks --repository home-k8s --branch main --personal
```

</details>

<details>
<summary>k3s on Ubuntu notes</summary>

- Install Ubuntu server 22.04
  - Picking the latest Ubuntu version supported by the Nvidia GPU Operator: [docs](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/23.9.2/platform-support.html)

- Install k3s
  - I ended up with this:
  ```
  curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable=traefik --disable=servicelb" sh -s -
  ```
  - Disabling Traefik because I wanted to use ingress-nginx instead
  - Disabling ServiceLB because I wanted to use MetalLB instead
  - No super solid reason, except I had used both before and think I'm more likely to use them out in the wild

- Copy the kubeconfig file to your local
  - Here my server is named crow. I copied it with:
  ```
  sudo cp /etc/rancher/k3s/k3s.yaml .
  sudo chown david: k3s.yaml
  scp crow:k3s.yaml ~/.kube/configs/k3s.yaml
  ```
- Edit 127.0.0.1 to your server's domain name

- https://github.com/settings/tokens
- export GITHUB_TOKEN=

</details>

<details>
<summary>kubeadm notes</summary>

- Install containerd
  ```
  sudo apt install containerd
  ```
- Followed the instructions here in the Kubernetes to set up the package repos and install kubectl, kubelet, and kubeadm:
  https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/#installing-kubeadm-kubelet-and-kubectl
- enable ip forwarding
  ```
  sudo sysctl -w net.ipv4.ip_forward=1
  /etc/sysctl.conf
  ```
- sudo kubeadm init

</details>

<details>
<summary>Miscellaneous k3s links</summary>

```
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable=traefik" sh -s -
```

https://docs.k3s.io/advanced?_highlight=gpu#nvidia-container-runtime-support

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#debian

</details>
