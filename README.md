# home-k8s

Ubuntu 18.04

### Install Containerd

Instructions [here](https://kubernetes.io/docs/setup/cri/#containerd).

Install Kubernetes, pointing to containerd:
```
kubeadm init --cri-socket /run/containerd/containerd.sock
```

If single-master setup, allow scheduling on the master node:
```
kubectl taint nodes <node-name> node-role.kubernetes.io/master:NoSchedule-
```