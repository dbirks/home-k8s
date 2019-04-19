```
helm init --dry-run --debug --service-account=tiller > deploy_tiller.yml
kubectl -n kube-system create sa tiller --dry-run -o yaml > sa_tiller.yml
kubectl create clusterrolebinding tiller-cluster-rule --clusterrole=cluster-admin --serviceaccount=kube-system:tiller --dry-run -o yaml > crb_tiller-cluster-rule.yml
```


Flux's [docs on this](https://github.com/weaveworks/flux/blob/master/site/helm-operator.md) are really helpful.

Relevant parts here for my notes:

```
helm init --dry-run --debug --service-account=tiller --override='spec.template.spec.containers[0].command'='{/tiller,--storage=secret}' --tiller-tls --tiller-tls-cert=tls/server.pem --tiller-tls-key=tls/server-key.pem --tiller-tls-verify --tls-ca-cert=tls/ca.pem > deploy_tiller.yml
```