---
name: fabro
description: Operate the private Fabro software-factory server (apps/fabro.yaml, issue 161) - config-in-git settings, dev-token login, the Collective model provider and its vault key, the custom fabro-local image, running and gating workflows from the CLI or API, and the known limits (local-only execution, restarts kill in-flight runs). Use when changing, upgrading, debugging or extending Fabro.
---

# Fabro (evaluation deploy, live since 2026-10-27)

- UI/API: `https://fabro.beluga-wyvern.ts.net` (Tailscale Ingress only; `/api/v1/*` is 401 without auth).
- Namespace `fabro`: StatefulSet `fabro` (1 replica, never more), PVC `storage-fabro-0` (local-path, 20Gi,
  on **dave-droid**), ConfigMap `fabro-settings`, Secret `fabro-bootstrap` (apps/fabro-secrets.enc.yaml).
- Image `ghcr.io/dbirks/fabro-local` = upstream `ghcr.io/fabro-sh/fabro` + bash and basic CLI tools
  (docker/fabro-local/Dockerfile, built by .github/workflows/build-fabro-local.yml). Upstream ships no
  `/bin/bash` and Fabro runs every sandbox command through bash, so on the stock image agent nodes fail with
  `sandbox-driver-sentinel: bash: not found` (prompt-only nodes still work).
  Upstream publishes nightlies only (no stable releases). Resolve a tag to a digest anonymously:
  `TOK=$(curl -s "https://ghcr.io/token?scope=repository:fabro-sh/fabro:pull" | jq -r .token)` and then a HEAD on
  `https://ghcr.io/v2/fabro-sh/fabro/manifests/<tag>` with an OCI index Accept header (docker-content-digest).

## Config lives in git, not the wizard
`FABRO_CONFIG=/etc/fabro/settings.toml` (the ConfigMap) makes `fabro server start` skip install mode.
Upstream's browser wizard insists on a GitHub token/App step and writes settings to the PVC. Edit the
ConfigMap, push, then delete the pod (no checksum annotation, so a ConfigMap change alone does not restart it).
The catalog validator is strict: every `[llm.providers.X.models.Y]` needs `api_model`, and a bad
`[llm]` table crash-loops the server with "catalog layer settings [llm] is invalid TOML".

## Auth
`dev-token` only. Token = Secret `fabro-bootstrap` key `FABRO_DEV_TOKEN` (format `fabro_dev_` + 64 hex).
Read it with kubectl when you need it; never commit or paste it. Rotate: regenerate both keys, SOPS-encrypt
over apps/fabro-secrets.enc.yaml, push, restart. No generic OIDC upstream (only dev-token and GitHub OAuth),
no RBAC: anyone logged in is a full admin.

## Model + vault
Provider `collective` (custom lithos provider, OpenAI chat codec) -> `https://llm-collective.e-gineering.com/v1`,
model `qwen3-8-flash-next` (Pennyroyal). Its key is vault secret `COLLECTIVE_API_KEY` (a provider named X reads
`X_API_KEY`). Fabro NEVER reads provider keys from env; set it via the API (value never echoed):
```bash
kubectl port-forward -n fabro svc/fabro 18080:80 &
DEV=$(kubectl get secret -n fabro fabro-bootstrap -o jsonpath='{.data.FABRO_DEV_TOKEN}' | base64 -d)
jq -n --arg v "$NEW_KEY" '{name:"COLLECTIVE_API_KEY",value:$v,type:"token"}' \
  | curl -s -H "Authorization: Bearer $DEV" -H 'content-type: application/json' --data @- http://127.0.0.1:18080/api/v1/secrets
```
Vault values are PLAINTEXT in `/storage/db/fabro.sqlite3`: the PVC is a secret store; back it up encrypted only.
The Collective path honours `response_format` json_schema (verified), so it is declared. Run-title
generation still often fails ("did not return a JSON document") because Pennyroyal thinks by default and
the title call's token budget is small: cosmetic, titles fall back.
`[server.scheduler] max_concurrent_runs = 2` so Fabro can't take most of Pennyroyal's 8 Collective slots.

## Running workflows (CLI inside the pod is quickest)
```bash
kubectl exec -it -n fabro fabro-0 -c fabro -- sh
fabro auth login --server http://127.0.0.1:32276/api/v1 --dev-token "$FABRO_DEV_TOKEN"
cd /tmp/smoke && git init -q . && git -c user.email=a@b -c user.name=a commit -q --allow-empty -m init
fabro model test --model qwen3-8-flash-next --server http://127.0.0.1:32276/api/v1
fabro run hello.fabro --environment local --server http://127.0.0.1:32276/api/v1      # -d to detach
fabro events <run> / fabro logs <run> / fabro inspect <run>   # same --server
```
- `--environment local` is required: server `[run.environment]` is not applied to CLI-created runs,
  and the only environment that exists is the synthetic `local` (CLI warns that settings.toml `run`
  is not transmitted: expected).
- A local run needs a git repo as its folder (the workflow's cwd). /tmp is wiped on pod restart.
- Human gate (`shape=hexagon`): `GET /api/v1/runs/<id>/questions`, then
  `POST /api/v1/runs/<id>/questions/<qid>/answer` with `{"kind":"selected","option_key":"A"}`
  (url-encode `#` in the qid as `%23`). 204 = accepted. Or use the run page's dock in the UI.
- Find a node's real failure in the events: `grep -o '"failure_reason":"[^"]*'` on `fabro events <run>`.

Verified 2026-10-27: prompt node, agent node with shell + write_file tools, plan -> gate -> implement,
answering via the API, run history kept across pod restarts.

## Known limits (do not paper over)
- **A pod restart kills in-flight runs**: they become `failed: terminated`. `fabro resume` says "already
  finished", and `fabro rewind` needs a GitHub-backed run with checkpoint pushes. So don't roll the pod
  while something is running.
- **Execution is `local` only**: tools run inside the Fabro pod (uid 1000, no SA token, caps dropped) with
  full egress and READ ACCESS TO THE VAULT DB. Trusted workflows only. docker/daytona are disabled, and
  Fabro's cron automations require one of them, so automations are unavailable until an isolated executor
  exists. Never mount a node Docker socket. Plugin sandbox drivers exist, but upstream runs still only
  execute on the built-in providers.
- flannel doesn't enforce NetworkPolicy, so none is declared.
- GitHub (PRs, rewind/fork, OAuth login, webhooks) is not configured. The tailnet URL can't receive GitHub
  webhooks; that would need a separate path-restricted public relay.
