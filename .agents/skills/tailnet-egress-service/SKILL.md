---
name: tailnet-egress-service
description: Reach a tailnet-only device (e.g. a phone running a model server) from inside the cluster, where pods have no route to 100.64.0.0/10. Use when a workload needs to call a MagicDNS name or tailnet IP from a pod, when someone asks to "connect to my phone" from a service, or when a pod fails DNS/times out against a 100.x address.
---

# Cluster → tailnet egress via a tailscale-operator Egress Service

Pods cannot route to CGNAT 100.64.0.0/10 and kube-dns does not resolve `.ts.net`.
The tailscale-operator (installed in `default`, OAuth creds in secret `operator-oauth`,
CRDs incl. `proxygroups.tailscale.com`) solves this with one Service — no sidecars,
no authkeys. Canonical example: `apps/iphone-tailnet-egress.yaml`.

## Recipe

```yaml
apiVersion: v1
kind: Service
metadata:
  name: iphone-llm            # consumers use http://iphone-llm.default.svc.cluster.local:8889
  namespace: default
  annotations:
    tailscale.com/tailnet-fqdn: iphone185.beluga-wyvern.ts.net   # full MagicDNS name
    # OR tailscale.com/tailnet-ip: 100.102.205.4  (exactly one of the two)
spec:
  type: ExternalName
  externalName: placeholder   # the operator overwrites this with the proxy Service
  ports:
    - { name: http, port: 8889, protocol: TCP }   # LIST EVERY PORT USED, or the proxy won't forward it
```

Apply via GitOps (commit to `apps/`). The operator provisions a proxy StatefulSet
(`ts-<name>-xxxxx-0`, image `tailscale/tailscale:v1.98.x`) and rewrites
`externalName` → `ts-<name>-....default.svc.cluster.local`. This survives Flux
re-reconciles (observed stable 2026-10-13).

## Verify

1. `kubectl wait --for=condition=TailscaleProxyReady=true svc/iphone-llm -n default --timeout=5m`
   (condition reason `ProxyCreated`)
2. `kubectl get svc iphone-llm -o wide` — EXTERNAL-IP must be the `ts-...` Service, not `placeholder`
3. Proxy log (`kubectl logs ts-<name>-xxxxx-0`) shows
   `Installing forwarding rules for destination <resolved tailnet IP>` — proof MagicDNS
   resolution happened inside the proxy, not in the pod.
4. From any pod: `kubectl run tmp --rm -i --restart=Never --image=curlimages/curl -- curl -m 15 http://<svc>.default.svc.cluster.local:<port>/...`

## Gotchas

- `tailscale ping <name>` INSIDE the proxy pod fails with `no such host` — the CLI uses
  the pod's resolv.conf (kube-dns). That's not the forwarding path; test through the
  Service instead.
- Tailnet node asleep/offline → `open-conn-track: timeout ... online=yes, lastRecv=never`
  and the pod sees a connect timeout (not refused). For an iPhone this is iOS suspending
  the app: the server only lives while the app is foregrounded and auto-lock is off.
- Flux needs `infra` reconciled before `apps` (dependsOn), so a fresh push can take
  a couple of minutes before the Service appears.
- Find tailnet nodes/names with `tailscale status` / `getent hosts <node>.beluga-wyvern.ts.net`
  on the workstation (the tailnet domain here is `beluga-wyvern.ts.net`).
- David's iPhone = tailnet node `iphone185` (100.102.205.4); its OpenAI-compatible
  model server (Gemma 3n-E4B Q2_K + apple-foundation) listens on port **8889** and
  reports usage token counts as 0.
