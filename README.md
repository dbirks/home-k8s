# home-k8s 🏡☸

## Current setup
Ubuntu as OS

containerd as container runtime

Cilium as CNI network plugin

Kubernetes as orchestrator

Sealed-secrets for encrypting secrets for version control

Flux for continuous deployment, using a pull model from inside the cluster

[nfs-client-provisioner](https://github.com/kubernetes-incubator/external-storage/tree/master/nfs-client) to set up a StorageClass to dynamically provision PersistentVolumes from a separate NFS server

MetalLB to fulfil LoadBalancer services

Tiller (with TLS enabled)

Flux's HelmOperator to deploy Helm charts

## Initial setup

### Install containerd

Instructions [here](https://kubernetes.io/docs/setup/cri/#containerd).

In short:
```
modprobe overlay
modprobe br_netfilter

# Setup required sysctl params, these persist across reboots.
cat > /etc/sysctl.d/99-kubernetes-cri.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.ipv4.ip_forward                 = 1
net.bridge.bridge-nf-call-ip6tables = 1
EOF

sysctl --system

# Install containerd
## Set up the repository
### Install packages to allow apt to use a repository over HTTPS
apt-get update && apt-get install -y apt-transport-https ca-certificates curl software-properties-common

### Add Docker’s official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

### Add Docker apt repository.
add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"

## Install containerd
apt-get update && apt-get install -y containerd.io

# Configure containerd
mkdir -p /etc/containerd
containerd config default > /etc/containerd/config.toml

# Restart containerd
systemctl restart containerd
```

### Install Kubernetes

```
kubeadm init --cri-socket /run/containerd/containerd.sock
```

If single-master setup, allow scheduling on the master node:
```
kubectl taint nodes <node-name> node-role.kubernetes.io/master:NoSchedule-
```

### Install Cilium

### Install Flux, Tiller, and HelmOperator
