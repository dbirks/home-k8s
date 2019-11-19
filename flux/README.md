# Flux

Mainly following [this page](https://helm.workshop.flagger.dev/prerequisites/#flux) on a workshop. Uses helm v3.

```
kubectl apply -f ns_fluxcd.yaml
helm repo add fluxcd https://charts.fluxcd.io
helm install flux fluxcd/flux --wait -n fluxcd --set registry.pollInterval=1m --set git.pollInterval=1m --set git.url=git@github.com:dbirks/home-k8s
kubectl -n fluxcd logs deployment/flux | grep identity.pub | cut -d '"' -f2
```

## Helm operator

Applied some CRDs:

```
kubectl apply -f flux-helm-release-crd.yaml
```

And used this:

```
helm upgrade -i helm-operator fluxcd/helm-operator --wait \
--namespace fluxcd \
--set git.ssh.secretName=flux-git-deploy \
--set git.pollInterval=1m \
--set chartsSyncInterval=1m \
--set configureRepositories.enable=true \
--set configureRepositories.repositories[0].name=stable \
--set configureRepositories.repositories[0].url=https://kubernetes-charts.storage.googleapis.com \
--set extraEnvs[0].name=HELM_VERSION \
--set extraEnvs[0].value=v3 \
--set image.repository=docker.io/fluxcd/helm-operator-prerelease \
--set image.tag=helm-v3-dev-fb98e2db
```

