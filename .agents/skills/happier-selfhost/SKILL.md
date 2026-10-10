---
name: happier-selfhost
description: Operate the self-hosted Happier relay (happier.beluga-wyvern.ts.net, issue #143) and the Arch daemon + Crush ACP engine on this workstation. Use when upgrading or backing up the relay, when Happier machines show offline, when adding/repairing the crush-acp custom backend, or when the Happier account/key or browser session state needs recovery.
---

# Happier self-host stack (relay on the cluster + daemon on the workstation)

Three moving parts:

1. **Relay** — `apps/happier.yaml` (+ `apps/happier-secrets.enc.yaml`), official
   `ghcr.io/happier-dev/relay-server` **stable** channel pinned by digest, light
   flavor + SQLite on a `local-path` PVC at `/data`, tailnet-only Tailscale
   Ingress. Web UI is embedded (served at `/`). Upgrade:
   `skopeo inspect docker://ghcr.io/happier-dev/relay-server:stable | jq .Digest`
   then update the digest pin, commit, `flux reconcile`.
2. **Daemon on this workstation** — `happier` CLI 0.2.14 (installer lane
   `stable`; payload `~/.happier/cli/current`, shim `~/.local/bin/happier`).
   systemd **user** service `com.happier.cli.daemon.default` (linger is on).
   Manage with `happier daemon status|start|stop|logs`, `happier service install`.
3. **Crush ACP engine (experimental)** — see below.

## Backup set (do this after any change)

- PVC `/data`: SQLite DB + uploaded files (single-writer; Recreate strategy).
- `apps/happier-secrets.enc.yaml` (HANDY_MASTER_SECRET) — a /data restore with a
  different secret can't decrypt server-managed secrets.
- **Account secret key**: `~/.happier/account-backup-key.txt` (600). The E2EE
  account lives ONLY in the headless-browser localStorage state + this key; the
  relay cannot recover it. Copy both into the password manager:
  - key file above
  - `~/.happier/browser-state/happier-web.json` (Playwright storageState holding
    the logged-in web account; importable as a browser profile)
- Phone sign-in later: app → restore → paste secret key (server URL
  `https://happier.beluga-wyvern.ts.net`).

## Machine/account facts

- Relay advertises itself via `GET /v1/features` (canonical URL
  `https://happier.beluga-wyvern.ts.net`, same for the web app; embedded UI).
- Auth policy: anonymous device-key signup ENABLED (documented recommendation
  for tailnet-only servers). Account public id `cmuz5y0k20009lu01rpnpcmup`.
- This machine registers as `delightful-goose`, machine id under
  `~/.happier/settings.json` / `servers/home/`.
- Socket.IO (`/v1/updates`) works through the tailscale-operator ingress;
  `HAPPIER_SERVER_TRUST_PROXY` is intentionally unset (operator proxy doesn't
  rewrite XFF). "Machine registers but stays offline" => check that WebSocket
  upgrade path first.
- **Daemon dead after a workstation reboot** (`happier daemon status` says not
  running): the real unit is `happier-daemon.default.service` (the
  `com.happier.cli.daemon.default` name is only a label). On boot it can start
  before Tailscale MagicDNS resolves (ENOTFOUND), then hit "Machine server
  ownership conflict detected; shutting down" because the relay still holds the
  previous boot's connection. It exits 0, so `Restart=on-failure` never retries.
  Fix: `systemctl --user restart happier-daemon.default` once the stale lease
  has expired, then check `happier daemon status` and look for "Connected to
  server" in the newest `~/.happier/logs/*-daemon.log`.

## Crush as a Happier engine (verified 2026-10-08, experimental)

Upstream Crush has **no released `crush acp`** (PRs #2450/#3295 still open).
We run a pinned personal build of PR #2450 head `df406b14`:

- Source: `~/build/crush-acp` (git checkout of the PR commit), built with
  `go build -o ~/build/crush-acp-bin .` (root main.go at that era).
- Installed at `~/.local/bin/crush-acp` — deliberately NOT named `crush`; the
  system `crush` v0.97.1 stays untouched.
- **Two local deviations from stock PR**, both required, both marked in code:
  1. *auto-auth shim* (`internal/acp/agent.go` Initialize sets
     `authenticated=true`): Happier's custom-ACP launcher never sends
     ACP `authenticate`, and the PR's `local` method only flipped a boolean.
     Without the shim every session dies with "Authentication required"
     (runtime closed before delivery).
  2. *isolated data dir + explicit provider config* (`--data-dir
     ~/.crush-acp`): the PR tree is Crush v0.76-era; it cannot parse the
     current crushrc-generated config, so `~/.crush-acp/crush.json` declares
     the `llmc` provider fully (openai-compat, base_url, static model list —
     that era has no `/models` discovery for custom providers). Without a
     provider the app boots with a nil AgentCoordinator and **panics on the
     first prompt** ("No agent configuration found" in
     `~/.crush-acp/logs/crush.log` is the tell).
- The ACP child gets `LLMC_API_KEY` from the Happier backend definition's
  env var (stored E2EE in account settings; the systemd user daemon env does
  NOT see ~/.bashrc, and busctl set-environment doesn't exist — don't retry
  that). If you re-register the backend, re-add the env var or every prompt
  401s.
- Custom backend registration lives in **account settings** (`acpCatalogSettingsV1`,
  web UI Settings → AI backends → Add ACP backend), id/name must match
  `^[a-z0-9][a-z0-9._-]*$` (a capital letter in `name` fails with the vague
  "ACP catalog settings are invalid."). Command
  `/home/david/.local/bin/crush-acp`, args one-per-line: `acp`,
  `--data-dir`, `/home/david/.crush-acp`. Edit by clicking the backend row
  (deep-link `?backendId=` did not load the existing draft when tried).
- Verified working via `happier session create --backend
  acpBackend:crush-acp --path <dir>` + `session send`: streaming thought/
  message chunks, and the **permission card round-trip** (approve in web UI →
  the write happens on this host). Stop/steer work.
- Known limitation (matches Happier's capability matrix): **Custom ACP resume
  by agent session ID is unsupported.** Killing the ACP child keeps the
  Happier transcript, but the respawned Crush starts with fresh context —
  the mobile "continue later" story is transcript-only for Crush today.
  Claude/Codex built-ins do resume. If true resume is needed, wait for an
  upstream ACP merge (#2450 advertises loadSession but Happier doesn't wire
  Custom-ACP resume yet).
- Debug a failing session: raw ACP frames with `--env HAPPIER_ACP_CAPTURE_IO=1`
  (capture lands in `/tmp/acp.client.messages.raw`, client→agent direction
  only; agent→client requests you must infer), child stderr at
  `~/.happier/cli/logs/subprocess/crush-acp/`, crush internals at
  `~/.crush-acp/logs/crush.log`, runner log path from `happier daemon logs`.

## Headless account/registration flow (repeatable for more machines)

1. `happier server add --name home --server-url https://happier.beluga-wyvern.ts.net --webapp-url https://happier.beluga-wyvern.ts.net --use`
2. `happier auth login --no-open --method web` prints a terminal-connect URL.
3. That URL needs a logged-in Happier session to approve. On this workstation
   it was driven headlessly with Playwright (`/home/david/.cache/ms-playwright/
   chromium-1243`, scratch project `~/build/pw`, storageState persisted to
   `~/.happier/browser-state/happier-web.json`). First visit = "First time
   here" signup (key generated in-browser; then BACK IT UP, see above).
4. `happier service install` (+ `loginctl enable-linger`, already on).
5. Smoke test: `happier session create --backend acpBackend:crush-acp --path /tmp/acp-demo --prompt "Reply with exactly: HAPPY-CRUSH-OK"` then `happier session history <id>`.

## Upgrade notes

- Relay: swap the digest in `apps/happier.yaml` only (GitOps; never kubectl-edit).
- CLI/daemon: `happier self update` (stable lane). The crush-acp shim must be
  rebuilt when you adopt a newer Crush or when PR #2450 merges upstream —
  re-check `gh pr view 2450 --repo charmbracelet/crush --json state,mergedAt`;
  on merge, throw the fork build away and point the backend at `crush acp`.
