```
helm template ~/dev/charts/stable/unifi \
    --name unifi-controller \
    --namespace unifi \
    --set timezone="America/New York" \
    --set image.tag=stable
```