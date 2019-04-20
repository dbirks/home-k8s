# Tiller and the Helm Operator

Flux's [docs on this](https://github.com/weaveworks/flux/blob/master/site/helm-operator.md) are really helpful.

Relevant parts here for my notes:

Tiller:

```
helm init --dry-run --debug --service-account=tiller --override='spec.template.spec.containers[0].command'='{/tiller,--storage=secret}' --tiller-tls --tiller-tls-cert=tls/server.pem --tiller-tls-key=tls/server-key.pem --tiller-tls-verify --tls-ca-cert=tls/ca.pem > deploy_tiller.yml
```

Helm client:

```
kubectl create secret tls helm-client --dry-run -o yaml --cert=tls/flux-helm-operator.pem --key=tls/flux-helm-operator-key.pem --namespace=kube-system | kubeseal --cert ~/home-k8s-public-key.pem --format yaml > secret_helm-client.yml
helm template --debug ~/dev/flux/chart/flux/ --name flux --set git.url=git@github.com:dbirks/home-k8s.git --set git.pollInterval=2m --set helmOperator.create=true --set helmOperator.tls.enable=true --set helmOperator.tls.verify=true --set helmOperator.tls.secretName=helm-client --set helmOperator.tls.caContent="$(cat ./tls/ca.pem)" > deploy_flux.yml
```
