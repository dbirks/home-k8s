---
name: hermes-agent
description: Operate the in-cluster NousResearch Hermes Agent (apps/hermes-agent.yaml, issue 162) - s6 process model and security context, tsidp OIDC dashboard login, config.yaml settings we set by hand, the LLM Collective provider and its key, how to exec in and verify, the Signal sidecar and its landline registration, the Gmail read+draft-only broker and its account helper, the Gnotes MCP adapter, and what is next (Microsoft 365 read-only, Bitwarden). Use when changing, upgrading, debugging or extending Hermes.
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
- Access (all values in `apps/hermes-signal.enc.yaml`, never plaintext): `SIGNAL_ALLOWED_USERS` = comma list of Signal
  account UUIDs (numbers also work, but contacts with phone-number privacy only expose a UUID), `SIGNAL_GROUP_ALLOWED_USERS`
  = group IDs. Setting the user allowlist DISABLES pairing, so list everyone, including the owner. Groups need BOTH the
  group ID in the group list AND the sender in the user list. Find IDs over the daemon (the CLI can't run while it holds
  the lock): `kubectl exec deploy/hermes -c hermes -- sh -c 'curl -s http://127.0.0.1:8080/api/v1/rpc -H "Content-Type: application/json" -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"listGroups\",\"params\":{\"account\":\"$SIGNAL_ACCOUNT\"}}"'`
  (`.result[].id` and `.members[].uuid`; write it to a chmod-600 scratch file, not the terminal log). The secret can't
  be decrypted locally (no age private key), so a rebuild must re-supply every key in it.
  Approving pairing without an allowlist: `kubectl exec deploy/hermes -c hermes -- hermes pairing approve signal CODE`
  (the exec-shim drops to uid 10000). Group replies don't require an @mention (`SIGNAL_REQUIRE_MENTION` default false).
- signal-cli logs the account number at daemon start; pod logs are not public, but don't paste them anywhere public.

## Gotcha: unsafe_control_permissions
`hermes chat` failing with `Gateway inaccessible: unsafe_control_permissions (/opt/data has mode 0777 ...)` means
the data dir is group/other-writable (local-path creates 0777). The `perms` initContainer runs `chmod go-w /opt/data`
on every start; if you see this, check that init ran. Smoke test: `kubectl exec deploy/hermes -c hermes -- hermes chat -q "run uname -s"`.

## Gmail = read + draft-only broker (apps/gmail-broker/)
- Why a broker: Google has no draft-only scope (`gmail.compose` and `gmail.modify` both SEND), and Hermes' own
  docs say its deny rules are not a sandbox (anything the agent can read, it can use). Hermes' bundled options all
  fail the no-send rule: Email platform (IMAP/SMTP, always replies), himalaya skill (CLI can send; creds readable
  from the terminal), google-workspace skill (always requests gmail.send/modify, token under ~/.hermes), Nous managed
  connectors (third party holds the grant).
- `broker.py` (stdlib Python, MCP streamable-HTTP with JSON responses) holds the tokens and exposes 8 tools: list_accounts,
  search_messages, get_message, get_thread, list_drafts, get_draft, create_draft (threaded replies keep threadId,
  In-Reply-To, References), update_draft. All outbound Gmail calls pass `_gmail()`'s method+path allowlist (no
  send/delete/import/modify; anything containing "send" is refused and logged `BLOCKED`). Logs never carry
  addresses/subjects/bodies.
- After editing broker.py, regenerate the ConfigMap (the command is in the header of gmail-broker.yaml) and commit both.
- Hermes side: env `GMAIL_BROKER_TOKEN` (same Secret `gmail-broker-token`), and by hand:
  `hermes config set mcp_servers.gmail '{url: "http://gmail-broker.default.svc.cluster.local:8080/mcp", headers: {Authorization: "Bearer ${GMAIL_BROKER_TOKEN}"}}'`
  plus `mcp_servers.gmail.tools.include` = the 8 tool names. Check: `kubectl exec deploy/hermes -c hermes -- hermes mcp test gmail`.
- Google Cloud (once): project, enable Gmail API, consent screen External + test users, then PUBLISH to "In production".
  In Testing mode, refresh tokens for these restricted scopes expire after 7 days. Unverified-in-production just shows a
  warning screen (Advanced, then continue). Create a "Desktop app" OAuth client; keep its JSON out of the repo.
- Add an account: `SOPS=/path/to/sops python3 .agents/skills/hermes-agent/scripts/gmail-auth.py ALIAS [--client FILE]`
  runs the loopback+PKCE consent and writes `apps/gmail-broker/gmail-broker-acct-ALIAS.enc.yaml` (and the client secret
  the first time). Then turn the broker's `accounts` volume into a projected volume with one source per account
  (`items: [{key: refresh_token, path: ALIAS.refresh_token}, {key: email, path: ALIAS.email}]`) and commit. Real
  addresses stay inside the encrypted Secret, and aliases are the only names used anywhere else.
- Token death (`invalid_grant` in broker logs): re-run the helper for that alias.

## Gnotes = notes/tasks via apps/gnotes-mcp/ (issue 163)
- Gnotes (apps/gnotes.yaml) has a REST API but no MCP server and no OIDC. `gnotes_mcp.py` (stdlib MCP, same shape as the
  Gmail broker) wraps `/api/v1` as the dedicated NON-admin Gnotes user `hermes`, which sees only notebooks shared with it
  (share from the Gnotes UI as viewer or editor). There are 13 tools and no trash/delete. `update_note`/`move_note` need the `version`
  from get_note (merge, not clobber), and rewrites that drop `att:` attachment lines are refused. Task `line` is
  0-based (server/src/tasks.rs enumerates body.lines()).
- Secrets: `gnotes-hermes-credentials` (GNOTES_API_KEY, a write-scope gnk_ key named hermes-mcp; GNOTES_HERMES_PASSWORD for
  recovery only, never mounted), `gnotes-mcp-token` (MCP_TOKEN; Hermes env GNOTES_MCP_TOKEN). Created without anyone
  seeing them: a script generated the password, ran `gnotes-server create-user hermes` (password on stdin), logged in at
  `/api/auth/login`, POSTed `/api/me/keys {"name","scope":"write"}`, and piped the values through 0600 temp files into sops.
  To rotate: revoke the key in Gnotes (log in as hermes with the recovery password, Account then API Keys), mint a new
  one the same way, re-encrypt BOTH fields, and roll gnotes-mcp.
- Hermes config (by hand): `mcp_servers.gnotes = {url: "http://gnotes-mcp.default.svc.cluster.local:8080/mcp", headers:
  {Authorization: "Bearer ${GNOTES_MCP_TOKEN}"}}` + `tools.include` with the 13 names. Check with `hermes mcp test gnotes`.
- Gotcha: a `hermes chat` started within seconds of `hermes config set mcp_servers...` may not have the new server yet.
  The model then calls manage_connections and hangs about 4 min. Wait for `MCP servers reconciled ... added=[...]` in
  agent.log, then retry.

## Next phases (issue 162) — not done yet
- **Microsoft 365 work mail/calendar, read-only**: plan is a Graph backend in the same broker (delegated Mail.Read +
  Calendars.Read + offline_access, device-code). Microsoft enforces read-only by scope. IMAP OAuth (IMAP.AccessAsUser.All)
  is read/write and has no calendar, so not that. Likely needs tenant-admin consent; check before building.
- **Bitwarden**: native browser_vault + `bw` CLI (needs a derived image), human unlock; no BW_SESSION/master password in env or PVC.
- Off-node encrypted backup of `/opt/data`.
