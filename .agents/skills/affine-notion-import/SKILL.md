---
name: affine-notion-import
description: Import a Notion export into the self-hosted AFFiNE instance (affine.beluga-wyvern.ts.net), driven headlessly with Playwright over the tailnet. Use when migrating Notion content into AFFiNE, when an AFFiNE import silently produces empty database tables, or when you need to script any logged-in action against AFFiNE.
---

# Importing Notion into self-hosted AFFiNE

AFFiNE's **importers** (Notion/Obsidian/Bear/Docx/Markdown) are client-side only —
there is no server-side import API or CLI, so an import must run in a logged-in
browser. Playwright works well because tsidp auto-authenticates from David's
workstation (see "Signing in").

That is NOT the same as "you can never write to AFFiNE from a script." Since 0.27 the
server ships a Rust markdown<->Yjs pipeline, so creating/reading docs headlessly IS
possible — see "Programmatic access" at the bottom. Use Playwright for the importers
and for anything else that only exists in the client; use the server paths for
ordinary doc read/write.

## 0. Check the kubectl context FIRST

`kubectl config current-context` is often a work EKS cluster, not home. Never assume.
Pass the context explicitly rather than mutating the kubeconfig:

```bash
kubectl --context admin@home get pods -n default | grep affine
```

Beware in zsh: `K="kubectl --context admin@home"; $K get ...` does NOT work
("command not found"). If you then pipe through `grep`, the error is swallowed and you
get a convincing but false "no resources found". Write the command out in full.

## 1. Pick the right export format — HTML, not Markdown

Read it off the running server instead of guessing:

```bash
curl -s https://affine.beluga-wyvern.ts.net/ | grep -oE 'src="/js/i18n[^"]*"'
curl -s https://affine.beluga-wyvern.ts.net/js/i18n.<hash>.js > i18n.js
grep -oE '"com\.affine\.import[^"]*":"[^"]{0,200}"' i18n.js
```

`com.affine.import.notion.tooltip` says: **"Supported import formats: HTML with subpages."**
Export from Notion as **HTML + include subpages**. The Markdown export also drops
embedded images that the HTML export keeps.

The same grep enumerates every importer (Markdown, Markdown+media zip, HTML, Notion,
Obsidian, Bear, OneNote, Docx, Snapshot) — handy for choosing a fallback.

## 2. Create a CLOUD workspace before importing

A fresh browser starts on a browser-local "Demo Workspace". Importing there puts
everything in one browser's storage: not on the PVCs, not visible to the iOS app or a
second user. Confirm server-side that a cloud workspace exists:

```bash
kubectl --context admin@home exec -n default deploy/affine-postgres -- \
  psql -U affine -d affine -c "select id,name from workspaces;"
kubectl --context admin@home exec -n default deploy/affine-postgres -- \
  psql -U affine -d affine -c "select u.email,m.role,m.state from workspace_members m join users u on u.id=m.user_id;"
```

A local workspace never appears in Postgres — that is the reliable test.
In the UI: workspace switcher (top-left) -> Create workspace -> type name ->
"Workspace type: AFFiNE (home)" is the cloud/server option -> Create.

## 3. Back up before importing

```bash
kubectl --context admin@home exec -n default deploy/affine-postgres -- \
  pg_dump -U affine -d affine > affine-pre-import-$(date +%Y%m%d-%H%M).sql
```

## 4. Signing in from a script

Use **python-playwright** (Arch pkg `python-playwright`; `/usr/bin/playwright` is a
Python script — there is no `playwright` node module installed). Use a
`launch_persistent_context` profile dir so the session survives between runs.

Go to `/signIn` and click **Continue with OIDC**: tsidp identifies the caller by its
**tailnet identity**, so from David's workstation it signs in with no password as
`dbirks@github.idp.beluga-wyvern.ts.net`. Verify:

```python
page.evaluate("""async () => (await fetch('/graphql',{method:'POST',
  headers:{'content-type':'application/json'},credentials:'include',
  body:JSON.stringify({query:'query { currentUser { id email name } }'})})).text()""")
```

Note there are two separate accounts on this server (the OIDC one and the
`david@birks.dev` admin). They see different workspaces — "I don't see it in my other
browser" almost always means "signed in as the other account".

## 5. UI automation gotchas

- **Clicks time out / "element outside viewport".** The bottom "Open this doc in AFFiNE
  app" sheet sits below a 900px viewport. Use `viewport={"width":1600,"height":1200}`.
- **`get_by_role(...).click()` often fails** on AFFiNE's custom elements. Real mouse
  clicks at rendered coordinates are the most reliable: `page.mouse.click(x, y)`.
  Screenshot first and read the coordinates off the image.
  At 1600x1200: workspace switcher ~(120, 77), "Create workspace" ~(123, 239),
  sidebar **Import** ~(68, 597).
- **`innerText` is stale/misleading** — it reports the app-sheet text long after the
  sheet is gone. Trust screenshots and server-side counts, not page text.
- **The All-docs list is virtualised.** Scrolling the window collects ~28 rows. Find the
  scroll container and scroll *that*:
  ```python
  page.evaluate("""() => { window.__sc=[...document.querySelectorAll('*')]
    .filter(e=>e.scrollHeight>e.clientHeight+200 && e.clientHeight>300)
    .sort((a,b)=>b.scrollHeight-a.scrollHeight)[0]; }""")
  page.evaluate("() => { window.__sc.scrollTop += 1200; }")
  ```
- **Run the import in the background with `python -u`.** Output through `| tail` is
  buffered and you will see nothing until it exits.

## 6. Block Notion's external images or the import crawls

Notion HTML references stock images on `app.notion.com` (proxied via
`affine-worker.toeverything.workers.dev`). They CORS-fail and stall the import for
minutes. Abort them — they are Notion template art, not user content:

```python
BLOCK = ("app.notion.com", "affine-worker.toeverything.workers.dev",
         "prod-files-secure", "notion.so")
page.route("**/*", lambda r: r.abort() if any(h in r.request.url for h in BLOCK) else r.continue_())
```

With this, a ~180-page export finished in ~255s (vs still running at 4min without).

## 7. DO NOT close the browser until the dialog says "Import completed"

Closing the context mid-import aborts it and **nothing** syncs. Poll the dialog:

```python
d = page.evaluate("() => { const d=document.querySelector('[role=\"dialog\"]'); return d? d.innerText:''; }")
# "Importing your workspace data, please wait patiently." -> keep waiting
# "Import completed" -> then sleep ~120s more to let sync flush to the server
```

## 8. Verify — the success dialog lies

Always check the server, not the dialog:

```bash
kubectl --context admin@home exec -n default deploy/affine-postgres -- psql -U affine -d affine \
  -c "select (select count(*) from workspaces) ws,(select count(*) from snapshots) snaps,(select count(*) from blobs) blobs;"
```

Doc count + quota via GraphQL in the browser context:
`{ workspace(id:"<id>"){ memberCount quota { memberLimit } docs(pagination:{first:400}){ totalCount } } }`

`title` comes back **null** on every doc because this deployment sets
`indexer.enabled: false` — that is expected, not data loss.

## 9. KNOWN DATA LOSS: Notion databases import as empty tables

This is the big one. In 0.27.4 the Notion importer brings over database **structure**
(columns/filters/sorts) but **none of the rows**, and the per-row subpages are not
imported at all. One database (Tasks Tracker) produced no table whatsoever.

Symptom: the doc shows "Table View | New Record | ... | Calculate" and nothing else.

Detect it by diffing titles: extract `<title>` from every HTML file in the export and
compare with the doc titles in AFFiNE. Everything missing will be a database row page.
Cross-check by counting HTML files that live in a folder named after a sibling `.csv`.

**The data is not lost** — Notion exports each database's rows to a `.csv` next to it.
Recover by converting each CSV to a markdown table and importing via
**Import -> Markdown files (.md)** (the importer accepts multiple files at once).
Drop all-empty columns while converting so the table stays readable. Titles will
collide with the empty shells, so either rename or trash the shell afterwards
(Trash is recoverable).

## 10. Sharing with a second person

- Workspace `memberLimit` on this self-hosted server is **10**.
- **No mailer is configured**, but that does not block invites: they arrive as in-app
  notifications ("<user> invited you to join <workspace>" / "Accept & Join").
- Flow: the other person signs in once via **Continue with OIDC** (auto-provisions the
  account), then invite that email from Settings -> Members, then they accept in-app.
- **tsidp identifies by tailnet user**, so a second person needs their own Tailscale
  user (a user invite) — a device merely added to your own tailnet account authenticates
  as *you* and lands in the same AFFiNE account.

## 11. Programmatic access (agents / MCP / scripts)

Verified against the deployed 0.27.4 image on 2026-09-22.

**No general-purpose API token exists.** PATs were removed in 0.27.0 (upstream PR
#15221) and the change is not in the release notes. Two headless auth paths remain,
neither needing a browser (stock self-host has no captcha/OTP):

- Cookie: `POST /api/auth/sign-in {email,password}` -> `affine_session` cookie. Send
  `x-affine-csrf-token` on session-mutating routes.
- Bearer (better for automation): same sign-in but with header
  `x-affine-client-kind: native` returns an `exchangeCode` instead of cookies; then
  `POST /api/auth/session/exchange {code, installationId, platform:"electron"}` ->
  `{accessToken, refreshToken}` (access 900s, refresh idle 30d).

NOTE: the tsidp/OIDC account has NO password, so it cannot use these. Mint a dedicated
service account with the admin `createUser` GraphQL mutation — `auth.allowSignup` is
false, so self-registration is closed by design.

**Reading a doc as markdown — works today, no config change:**

```
GET /api/workspaces/<workspaceId>/docs/<docGuid>     # raw Yjs binary, permission-checked
```

then decode with `@affine/server-native` (already inside the container):
`parseDocToMarkdown()`, plus `readAllDocIdsFromRootDoc()` to enumerate and
`parsePageDoc()` / `parseWorkspaceDoc()` for metadata. The root doc's id == the
workspace id. There is NO HTTP push endpoint; writes go over socket.io
(`space:push-doc-update`, base64 Yjs update) or via MCP below.

**Built-in MCP server** at `GET|POST /api/workspaces/:workspaceId/mcp`, Bearer
`aff_mcp_v1.<id>.<secret>`. Credentials are per-workspace, minted with the
`createMcpCredential(input:{workspaceId,name,accessMode,expirationDays})` GraphQL
mutation (token revealed once; rotate has a grace window).

Two gates:

1. `copilot.enabled` must be true in config.json. `assertCopilotEnabled()` checks ONLY
   that flag — no LLM provider key is required. Until then every MCP call 403s with
   "Copilot is disabled." Enabling it also turns on the AI UI, which will error until a
   provider is configured (see "AI providers / BYOK" below).
2. Write tools (`create_document`, `update_document`, `update_document_meta`) are
   appended only when `accessMode === READ_WRITE && (env.dev || env.namespaces.canary)`.
   On a stable build you get `read_document` + `doc_search` only. `env.namespaces.canary`
   is `AFFINE_ENV === 'dev'`, so setting `AFFINE_ENV=dev` unlocks writes on the stable
   image. Side effects are mostly benign (debug logging, canary client versions); the
   scary-looking y-octo merge codec swap in core/doc/options.ts is double-gated behind
   `doc.experimental.yocto`, which defaults false — leave that off.

`doc_search` needs the indexer, and this deployment sets `indexer.enabled: false`, so
search will be degraded/useless until that changes. `read_document` is unaffected.

**Gotchas for any programmatic write path:**

- Upstream #14582: docs created programmatically exist in Postgres and show up in
  GraphQL, but do NOT appear in All-docs/sidebar until opened once.
- Upstream #15466: table/database cells written programmatically can render empty
  (plain string vs Y.Text) — the same database weak spot as section 9.
- The maintainer's stated reason for canary-gating writes is that y-octo merges differ
  subtly from yjs on CONCURRENT edits. Batch writes to idle docs are the safe case.

**Third-party option:** `DAWNCR0W/affine-mcp-server` (~289 stars, actively released,
106 read+write tools, self-hosted first-class, email/password auth). Mature, but it
drives AFFiNE's explicitly-unstable internal GraphQL/WebSocket APIs and was already
broken once by 0.27.0. Pin versions on both sides if you use it.

## 12. AI providers / BYOK (copilot.enabled)

Authoritative source: `.docker/selfhost/schema.json` at the matching tag. Under
`copilot`, the self-host config keys are FLAT and dotted:

| key | default | meaning |
|---|---|---|
| `enabled` | `false` | Enable AI features. Gates the MCP server too. |
| `byok.enabled` | `true` | Owners/admins may add provider keys via AI BYOK. |
| `byok.allowedProviders` | `["openai","anthropic","gemini","fal"]` | Providers selectable in BYOK. |
| `byok.allowCustomEndpoint` | `false` | Allow a BYOK key to use a CUSTOM provider endpoint. |
| `byok.allowPrivateEndpoint` | `false` | Allow those custom endpoints to resolve to PRIVATE network targets. |

Keys are configured per workspace in the UI: Workspace Settings -> Integrations ->
AI BYOK. The broader provider enum in the backend is `openai`, `anthropic`,
`anthropicVertex`, `cloudflareWorkersAi`, `fal`, `gemini`, `geminiVertex`, but BYOK
only offers the four above by default.

**Pointing AFFiNE at this cluster's own models.** Because `openai` is an allowed type
and custom endpoints are supported, AFFiNE can use the in-cluster vLLM/KServe stack
instead of a paid API. Set in config.json:

```json
"copilot": {
  "enabled": true,
  "byok.allowCustomEndpoint": true,
  "byok.allowPrivateEndpoint": true
}
```

then add an `openai`-type BYOK key in the workspace UI pointing at an OpenAI-compatible
base URL. Prefer the vLLM workload Service DIRECTLY
(`<model>-kserve-workload-svc:8000/v1`) over `llm.birks.dev` — per AGENTS.md the AI
Gateway ext-proc buffers/translates every body and is fragile, and going direct keeps
traffic in-cluster. That is a private-network target, which is exactly why
`byok.allowPrivateEndpoint` must be true.

Caveats for this cluster: KEDA scales idle models to zero, so the first AFFiNE AI call
after idle hits a ~2min cold start and may time out; and only ~2 models fit resident
(46GB host RAM is the binding constraint, not VRAM).

Remember: none of this is needed just to unlock MCP — `enabled: true` alone does that.
