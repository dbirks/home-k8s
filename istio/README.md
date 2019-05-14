Assuming istio is cloned to ~/dev/istio, reset to the latest tagged release:

```
git reset --hard 1.1.6
```

And from the home-k8s/istio dir, run:

```
helm template ~/dev/istio/install/kubernetes/helm/istio-init --name istio-init --namespace=istio-system > istio-init.yml
```