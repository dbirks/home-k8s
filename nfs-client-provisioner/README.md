Made from helm chart stable/nfs-client-provisioner:

```
helm template . --set storageClass.name=nfs-share \
                --set storageClass.provisionerName=ostrich \
                --set storageClass.defaultClass=true \
                --set nfs.server=10.1.1.2 \
                --set nfs.path=/mnt/tetons/kubernetes
```