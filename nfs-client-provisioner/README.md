Made from helm chart stable/nfs-client-provisioner:

```
helm template . --set nfs.server=10.1.1.2 \
                --set nfs.path=/mnt/tetons/kubernetes \
                --set storageClass.name=nfs-share \
                --set storageClass.defaultClass=true
```