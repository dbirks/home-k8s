Made from helm chart stable/nfs-client-provisioner:

```
helm template ~/dev/charts/stable/nfs-client-provisioner \
                --name ostrich \
                --set storageClass.name=nfs \
                --set storageClass.provisionerName=ostrich \
                --set storageClass.defaultClass=true \
                --set nfs.server=10.1.1.2 \
                --set nfs.path=/mnt/tetons/kubernetes
```