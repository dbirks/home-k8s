---
name: omnigent
description: Operate and upgrade the in-cluster Omnigent coding-agent server (apps/omnigent*.yaml) - server, persistent host, on-demand Kubernetes sandbox runners, Postgres. Use when upgrading Omnigent, when a sandbox session will not start, or when checking what changed between releases.
---

# Omnigent

Pieces (namespace `omnigent`, runners in `omnigent-sandboxes`):
- `apps/omnigent.yaml` — server `ghcr.io/omnigent-ai/omnigent-server-kubernetes:<tag>` (sandbox provider SDK bundled),
  Postgres 16 StatefulSet `postgres-0`, ConfigMap `omnigent-config`, private ingress `omnigent.hoam.lan`.
  Tailnet: `apps/omnigent-tailscale-ingress.yaml` (`omnigent.beluga-wyvern.ts.net`). Auth is OFF (single-user);
  the tailnet/private ingress is the perimeter.
- `apps/omnigent-host.yaml` — persistent `omnigent host` (`omnigent-host:<tag>`) with a Pi models.json ConfigMap.
- `apps/omnigent-sandbox.yaml` — RBAC + ConfigMap `omnigent-sandbox-config` (`sandbox.provider: kubernetes`); the
  runner image is pinned BY DIGEST and must match the host tag.
- Secrets (SOPS): `omnigent-secrets` (POSTGRES_PASSWORD etc.), `omnigent-creds` in omnigent-sandboxes (runner env).

## Upgrade recipe (done v0.11.0 -> v0.21.0 on 2026-11-07, no friction)
1. Read the notes: `for t in ...; do gh release view -R omnigent-ai/omnigent $t; done` — grep "Breaking".
   Diff upstream's `deploy/kubernetes/overlays/sandbox-runners/{role,sandbox-config}.yaml` and
   `deploy/kubernetes/base/configmap.yaml` between the old and new tag (`gh api repos/omnigent-ai/omnigent/contents/<path>?ref=<tag>`).
2. Back up Postgres first (custom format, stays on the PVC):
   `kubectl -n omnigent exec postgres-0 -- pg_dump -U omnigent -d omnigent -Fc -f /var/lib/postgresql/data/backups/omnigent-<ver>-<date>.dump`
   then `kubectl cp` a copy off the node. Restore: `pg_restore -U omnigent -d omnigent --clean <file>`.
3. Resolve the runner digest without docker/crane (anonymous GHCR):
   ```
   T=$(curl -s "https://ghcr.io/token?scope=repository:omnigent-ai/omnigent-host:pull" | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
   curl -sI -H "Authorization: Bearer $T" -H "Accept: application/vnd.oci.image.index.v1+json" \
     https://ghcr.io/v2/omnigent-ai/omnigent-host/manifests/vX.Y.Z | grep -i docker-content-digest
   ```
4. Bump the three image refs together, commit, reconcile Flux. The server runs Alembic migrations itself at startup
   (log: "schema is out of date ... attempting automatic migration"); confirm with
   `kubectl -n omnigent exec postgres-0 -- psql -U omnigent -d omnigent -tAc "select version_num from alembic_version"`.
5. The host image is large: a cold pull takes several minutes (PodInitializing is normal). Runner Pods pull the same
   digest, which is why `pod_ready_timeout_s: 300` is set (default 90 s).
6. Verify: `curl /health` in the server pod -> `{"status":"ok"}`; host logs show it registered; start a session in the UI.

## Notes from the release notes (v0.12 - v0.21)
- Bare `omni` now starts a background host (`omni run` launches the default agent); we call `omnigent host` explicitly.
- Canvas is built in (enable with `OMNIGENT_FEATURES=canvas`); `OMNIGENT_FEATURES` now takes `name` / `-name` overrides.
- Managed terminals need tmux >= 3.3 (inside the host image, fine).
- v0.16: only session owners can manage MCP servers. v0.20: uploaded agent bundles are restricted on single-user servers too.
- New optional providers (`agent_sandbox` CRs, microsandbox, Gensee...). Upstream's role grants `agents.x-k8s.io/sandboxes`;
  we mirror it, inert without the CRD.
- The startup WARNING "OMNIGENT_LOCAL_SINGLE_USER is set and the server is bound to 0.0.0.0" is expected (auth off by design).

## Known stale bit
The host's Pi models.json still points at `vllm.default.svc.cluster.local` (Qwen3.6, parked). To use the live model,
point it at the LLM Collective or the running engine's Service (see AGENTS.md "Serving stack").
