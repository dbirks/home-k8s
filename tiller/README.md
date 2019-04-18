```
helm init --dry-run --debug --service-account=tiller > deploy_tiller.yml
kubectl -n kube-system create sa tiller --dry-run -o yaml > sa_tiller.yml
kubectl create clusterrolebinding tiller-cluster-rule --clusterrole=cluster-admin --serviceaccount=kube-system:tiller --dry-run -o yaml > crb_tiller-cluster-rule.yml
```