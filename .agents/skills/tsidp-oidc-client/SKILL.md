---
name: tsidp-oidc-client
description: Give an in-cluster app tailnet single sign-on through tsidp (idp.beluga-wyvern.ts.net). Covers registering the OIDC client headlessly, wiring the pod to reach the tailnet-only issuer, and working out the sub claim for admin allow-lists. Use when adding OIDC login to an app, or when an app logs a user in but doesn't make them admin.
---

# tsidp OIDC client for an app

tsidp (infra/tsidp.yaml) authenticates by **tailnet identity**: there are no passwords,
and from David's workstation it signs in as `dbirks@github`.

## 1. Register the client (no browser needed)

The admin UI accepts a plain form POST from a tailnet-admin machine:

```bash
curl -s -X POST https://idp.beluga-wyvern.ts.net/new \
  --data-urlencode name=APP \
  --data-urlencode redirect_uris=https://APP.beluga-wyvern.ts.net/CALLBACK_PATH \
  -o "$SCRATCH/tsidp-new.html"   # chmod 600; holds the secret, delete after use
```

The response HTML shows the client ID and **client secret once** (input fields labelled
"Client ID" and "Client Secret"). Parse the secret straight into a SOPS secret and never
print it. The client ID isn't secret and can go in the manifest.

- Token auth is `client_secret_post` or `client_secret_basic` only, so it's always a
  confidential client. PKCE S256 is also supported.
- To list existing clients: `curl -s https://idp.beluga-wyvern.ts.net/ | grep /edit/`.

## 2. Reach the issuer from a pod

`idp.beluga-wyvern.ts.net` doesn't resolve in-cluster. Map it with hostAliases to the
idp-egress ClusterIP (`kubectl get svc -n default | grep idp-egress`, currently
`10.109.232.240`):

```yaml
      hostAliases:
        - ip: "10.109.232.240"
          hostnames: [idp.beluga-wyvern.ts.net]
```

The issuer is `https://idp.beluga-wyvern.ts.net`, with no path.

## 3. The `sub` claim is `userid:<DECIMAL>`

It is NOT hex, even though tailcfg's `UserID.String()` looks like it should be. This was
verified from the stored identity after a real login (communication, 2026-10-15). Get
yours:

```bash
echo "userid:$(tailscale status --json | jq .Self.UserID)"
```

`email` comes through as `LOGIN@github.idp.beluga-wyvern.ts.net`. If an app keys admin
on email, use that form.

## 4. Verify headlessly

```bash
pnpm dlx @playwright/cli open https://APP.beluga-wyvern.ts.net/LOGIN_PATH   # tsidp auto-approves
pnpm dlx @playwright/cli eval "async () => (await fetch('/api/me').then(r => r.text()))"
pnpm dlx @playwright/cli close
```

## Recovery

If the tsidp PVC is lost, every client disappears. Re-register each one and re-capture
its secret into SOPS (see the infra/tsidp.yaml header).
