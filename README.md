# home-k8s

Ubuntu 18.04

### Install Containerd

Instructions [here](https://kubernetes.io/docs/setup/cri/#containerd).

### Install Kubernetes

```
kubeadm init --cri-socket /run/containerd/containerd.sock
```

If single-master setup, allow scheduling on the master node:
```
kubectl taint nodes <node-name> node-role.kubernetes.io/master:NoSchedule-
```

Or more generally:
```
kubectl taint nodes --all node-role.kubernetes.io/master-
```



### Start Flux

scp the flux 