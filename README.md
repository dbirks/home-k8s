# home-k8s 🏡☸

## Current setup

k3os as the OS




## Installation

### k3os

Grabbed a recent .iso (0.11.1 currently) from [their releases](https://github.com/rancher/k3os/releases). Wrote to usb stick with:

```
sudo ddrescue ~/Downloads/k3os-amd64.iso /dev/sdb --force
```

Did the install with a monitor attached. Entered my github username when prompted to get my public ssh key set up. 

After installation, connected to it over ssh with the `rancher` user.

To use kubectl from my laptop, I grabbed the kubeconfig from `/etc/rancher/k3s/k3s.yaml`. Substituted 127.0.0.1 with its DHCP address. Pointed the `KUBECONFIG` env var to the yaml file and connected successfully.

### flux v2

Installed the cli with:

```
nix-env -i fluxcd
```

Then checked that prereqs were met:

```
flux check --pre
```

