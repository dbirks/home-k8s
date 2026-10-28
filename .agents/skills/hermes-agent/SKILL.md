---
name: hermes-agent
description: Operate the in-cluster NousResearch Hermes Agent (apps/hermes-agent.yaml, issue 162) - s6 process model and security context, tsidp OIDC dashboard login, config.yaml settings we set by hand, the LLM Collective provider and its key, how to exec in and verify, the Signal sidecar and its landline registration, and what is next (Gmail draft-only broker, Bitwarden). Use when changing, upgrading, debugging or extending Hermes.
---

# Hermes Agent (in-cluster since 2026-10-28)

- Dashboard: `https://hermes.beluga-wyvern.ts.net` (tailnet only) via `apps/hermes-dashboard-tailscale-ingress.yaml`
  (Service `hermes-dashboard` now selects `app: hermes`; the old workstation EndpointSlice is gone).
- Deployment `hermes` (default ns, 1 replica, Recreate) on **dave-droid**, PVC `hermes-data` (local-path, 20Gi) at
  `/opt/data`. Image `docker.io/nousresearch/hermes-agent:vX.Y.Z@sha256:…` (tags carry the `v`; `0.22.5` 404s).
  Resolve a digest: token from `auth.docker.io/token?service=registry.docker.io&scope=repository:nousresearch/hermes-agent:pull`,
  then HEAD `registry-1.docker.io/v2/nousresearch/hermes-agent/manifests/<tag>` with an OCI index Accept header.
  The image is big (full Chromium): pre-pull with `talosctl … -n 10.0.0.195 image pull --namespace cri <ref>`.

## Process model / security context (why it looks like this)
ENTRYPOINT is s6-overlay `/init` as PID 1. It must start as **root** to chown `/opt/data`, seed config and sync
bundled skills, then drops gateway + dashboard to `hermes` (uid 10000) with s6-setuidgid. So `runAsUser: 0`,
`allowPrivilegeEscalation: false`, caps = drop ALL + `CHOWN DAC_OVERRIDE FOWNER SETUID SETGID KILL`. Verify the
workers are non-root: `kubectl exec deploy/hermes -- ps -eo user,pid,args | grep hermes_cli`.
`gateway run` CMD is a heartbeat; s6 restarts the real gateway. No liveness probe on purpose; startup/readiness use
`/api/status` (unauthenticated, reports `auth_required`/`auth_providers`). Logs: `kubectl logs deploy/hermes`
(dashboard + boot), gateway in `/opt/data/logs/gateways/default/current`, auth audit in `/opt/data/logs/dashboard-auth.log`.
`kubectl exec deploy/hermes -- hermes …` auto-drops to uid 10000 (exec-shim), so files stay hermes-owned.

## Dashboard login = tsidp OIDC (direct, no proxy)
Hermes' docs say "confidential clients are not supported yet" — stale: v0.22.5's `plugins/dashboard_auth/self_hosted`
layers `client_secret` on PKCE (`HERMES_DASHBOARD_OIDC_CLIENT_SECRET`), and tsidp only issues confidential clients.
- tsidp client `hermes`, id `49e3c44121567a4fab67116283f88af4` (in the manifest), redirect
  `https://hermes.beluga-wyvern.ts.net/auth/callback`; secret in `apps/hermes-secrets.enc.yaml`.
- Pod reaches the issuer via `hostAliases` -> idp egress ClusterIP (see tsidp-oidc-client skill).
- No allowlist exists in Hermes: any tsidp identity (= anyone on the tailnet allowed to reach `hermes`) can log in.
- Verify headlessly (tsidp auto-approves from a tailnet machine):
  ```bash
  J=$(mktemp); curl -s -L -c $J -b $J -o /dev/null "https://hermes.beluga-wyvern.ts.net/auth/login?provider=self-hosted"
  curl -s -b $J https://hermes.beluga-wyvern.ts.net/api/auth/me | jq '{provider,display_name}'; rm -f $J
  ```
  Cookies must come back `Secure` with the `__Host-` prefix; if not, `dashboard.trusted_proxies` is missing.

## config.yaml is Hermes-owned state (on the PVC), set by hand — re-apply after a PVC loss
```bash
kubectl exec deploy/hermes -- hermes config set dashboard.trusted_proxies '["10.244.0.0/16"]'   # ts proxy pods -> trust X-Forwarded-Proto
kubectl exec deploy/hermes -- hermes config set providers.collective "{api: \"https://llm-collective.e-gineering.com/v1\", key_env: COLLECTIVE_API_KEY, transport: openai_chat, default_model: qwen3-8-flash-next, models: {qwen3-8-flash-next: {context_length: 524288}}}"
kubectl exec deploy/hermes -- hermes config set model.provider collective
kubectl exec deploy/hermes -- hermes config set model.default qwen3-8-flash-next
```
(`hermes config set model '{…}'` is refused; set `model.*` keys individually.) Restart to apply dashboard changes:
`kubectl delete pod -l app=hermes`.

## Model key
`COLLECTIVE_API_KEY` env (optional secretKeyRef) from `hermes-secrets`. A dedicated Collective key minted in the gateway
UI (/api-keys; there is no key API). Re-encrypting the secret needs BOTH values, since the age private key isn't on the
workstation (it can't be decrypted locally): rebuild it with `kubectl create secret … --from-file` for each key, piped
through `sops encrypt --filename-override apps/hermes-secrets.enc.yaml`.

## Signal (signal-cli sidecar, Hermes has its OWN number)
- Sidecar `signal-cli` in the hermes pod (official `ghcr.io/asamk/signal-cli:0.14.9`, JVM, uid 999), state on PVC
  `hermes-signal-data` at `/var/lib/signal-cli` (identity keys: treat as a secret). HTTP daemon on
  `127.0.0.1:8080`, pod-local only, no auth (Host header pinned to localhost). Hermes reads `SIGNAL_HTTP_URL` +
  `SIGNAL_ACCOUNT`; the number is ONLY in `apps/hermes-signal.enc.yaml` (never in git plaintext, issues, skills).
- The number is a LANDLINE registered as a PRIMARY device (not a linked device). The sidecar idles until the marker
  file `/var/lib/signal-cli/hermes-registered` exists, so register/verify can run via exec with the account unlocked.
- (Re-)registration (only after losing the PVC; it logs out any other primary on that number). Every command below
  is `kubectl exec [-i] deploy/hermes -c signal-cli -- bash -c '/opt/signal-cli/bin/signal-cli --config=/var/lib/signal-cli -a "$SIGNAL_ACCOUNT" ...'`
  (the number comes from the pod env, so it never appears on your command line):
  1. Captcha: solve https://signalcaptchas.org/registration/generate.html in a desktop browser, copy the
     `signalcaptcha://...` link. Put it in a scratch file and feed it on stdin (`read -r CAP; ... register --captcha "$CAP"`).
  2. That attempt sends an SMS, which a landline ignores, but Signal requires it before voice. Wait ~60-90s, then `register --voice`
     (no captcha needed the second time). The call can be re-requested if missed; it worked twice in a row.
  3. `verify 123456` with the spoken code, then `updateProfile --given-name Hermes`, then
     `touch /var/lib/signal-cli/hermes-registered`; the wrapper starts the daemon within a minute.
- Hermes retries a failed platform with backoff up to 5 min, so it can take ~5 min after the daemon is up to see
  `signal` connect in `kubectl logs deploy/hermes -c hermes`. Check the daemon: `kubectl exec deploy/hermes -c hermes -- curl -s http://127.0.0.1:8080/api/v1/check`.
- Access: no `SIGNAL_ALLOWED_USERS`, so unknown DMs get a pairing code. Approve with
  `kubectl exec deploy/hermes -c hermes -- hermes pairing approve signal CODE` (the exec-shim already drops to uid
  10000; there is no `s6-setuidgid` on PATH). `hermes pairing list` shows approved users (by Signal UUID). Groups are off (no `SIGNAL_GROUP_ALLOWED_USERS`).
- signal-cli logs the account number at daemon start; pod logs are not public, but don't paste them anywhere public.

## Gotcha: unsafe_control_permissions
`hermes chat` failing with `Gateway inaccessible: unsafe_control_permissions (/opt/data has mode 0777 ...)` means
the data dir is group/other-writable (local-path creates 0777). The `perms` initContainer runs `chmod go-w /opt/data`
on every start; if you see this, check that init ran. Smoke test: `kubectl exec deploy/hermes -c hermes -- hermes chat -q "run uname -s"`.

## Next phases (issue 162) — not done yet
- **Gmail**: draft-only means a separate broker holding the tokens (gmail.compose can send at Google); never give Hermes a send-capable token.
- **Bitwarden**: native browser_vault + `bw` CLI (needs a derived image), human unlock; no BW_SESSION/master password in env or PVC.
- Off-node encrypted backup of `/opt/data`.
