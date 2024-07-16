# home-k8s 🏡☸

🚧 Under construction 🚧

The readme is currently a slurry of old setup notes: kubeadm, Kairos, then k3s, and now Talos.

## Overview


## Installing Talos

https://www.talos.dev/v1.7/learn-more/image-factory/

https://factory.talos.dev/

`Reset Talos installation`

https://www.talos.dev/v1.7/talos-guides/configuration/nvidia-gpu-proprietary/

factory.talos.dev/installer/0412a9a6369c0fb55e913cdfcbf4ad6ca3fab6e56ab71198ec4b58ad7e7a4ddd:v1.7.5 

talosctl gen config home https://10.0.0.30:6443

- changed the disk to sdb
  `talosctl disks`
- 


```diff
     install:
-        disk: /dev/sda # The disk used for installations.
-        image: ghcr.io/siderolabs/installer:v1.7.5 # Allows for supplying the image used to perform the installation.
+        disk: /dev/sdb
+        image: factory.talos.dev/installer/0412a9a6369c0fb55e913cdfcbf4ad6ca3fab6e56ab71198ec4b58ad7e7a4ddd:v1.7.5
```


talosctl apply-config --insecure -n 10.0.0.30 --file controlplane.yaml

The `Stage` turned to `Installing`, and after a couple of minutes, it rebooted. The `Stage` then was `Booting`, some more log messages went across the screen, and then a minute or so later, a message came up saying that it's time to run `talosctl bootstrap`.

talosctl bootstrap -n 10.0.0.30 -e 10.0.0.30 --talosconfig ./talosconfig

talosctl kubeconfig -n 10.0.0.30 -e 10.0.0.30 --talosconfig talosconfig



export GITHUB_TOKEN=ghp_...
flux bootstrap github --owner dbirks --repository home-k8s --branch main --personal


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





🚧 Under construction 🚧

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
- 


curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable=traefik" sh -s -

https://docs.k3s.io/advanced?_highlight=gpu#nvidia-container-runtime-support

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#debian




## Current setup

- Kairos as the OS
  - Picked the Debian-based image
  - Switched to it after k3os stopped being developed
  - Operates basically the same as k3os as far as I can see... a purpose-built linux distro for running k3s

## Installation

### k3os

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


### flux v2

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
