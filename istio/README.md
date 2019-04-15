For istio-cni:

```
helm template cni-1.1.3/deployments/kubernetes/install/helm/istio-cni/ --name=istio-cni --namespace=istio-system > istio-cni.yml
```

For istio:
```
helm template istio-1.1.3/install/kubernetes/helm/istio --name istio --namespace=istio-system --set istio_cni.enabled=true > istio.yml
```