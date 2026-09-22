---
name: affine-notion-import
description: Import a Notion export into the self-hosted AFFiNE instance (affine.beluga-wyvern.ts.net), driven headlessly with Playwright over the tailnet. Use when migrating Notion content into AFFiNE, when an AFFiNE import silently produces empty database tables, or when you need to script any logged-in action against AFFiNE.
---

# Importing Notion into self-hosted AFFiNE

AFFiNE's import is **100% client-side** — there is no server-side import API or CLI.
Anything you want imported must happen in a logged-in browser. Playwright works well
because tsidp auto-authenticates from David's workstation (see "Signing in").

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
