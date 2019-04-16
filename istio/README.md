For istio-cni:

https://preliminary.istio.io/docs/setup/kubernetes/additional-setup/cni/

```
helm fetch istio.io/istio-cni --untar --untardir istio-fetch
helm  template istio-fetch/istio-cni/ --name=istio-cni --namespace=istio-system > istio-cni.yml
```

For istio:
```
helm template istio-1.1.3/install/kubernetes/helm/istio --name istio --namespace=istio-system --set istio_cni.enabled=true > istio.yml
```