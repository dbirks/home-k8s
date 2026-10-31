---
name: openwebui-web-search
description: Web search in Open WebUI is powered by a self-hosted SearXNG (apps/searxng.yaml, search.beluga-wyvern.ts.net) wired entirely through env vars in apps/openwebui.yaml. Use when web search is off/broken, returns no results or 403, an upstream engine is CAPTCHA'd or rate-limited, the JSON API needs checking, or SearXNG itself must be upgraded or its secret rotated.
---

# Open WebUI web search via self-hosted SearXNG

Two manifests, one wiring. Open WebUI never talks to google directly — it queries
`http://searxng.default.svc.cluster.local:8080/search` and SearXNG fans the query out to
google / duckduckgo / bing / brave / wikipedia and merges the answers.

**Privacy trade-off (tell the user, don't "fix" it silently):** enabling web search sends the
*query text* out of the house. SearXNG logs nothing (limiter off, no valkey, cache dir is an
emptyDir) and hides each browser behind dave-droid's IP, but the engines still see the query and
keep their own logs, and Open WebUI then fetches each result page itself.

## Pieces

| what | where |
|---|---|
| SearXNG Deployment/Service/ConfigMap | `apps/searxng.yaml` (namespace `default`, node `dave-droid`) |
| `server.secret_key` | `apps/searxng-secret.enc.yaml` → env `SEARXNG_SECRET` |
| tailnet UI | Ingress `searxng-ts`, host `search` → `https://search.beluga-wyvern.ts.net` (auto TLS) |
| Open WebUI side | the `--- WEB SEARCH ---` env block in `apps/openwebui.yaml` |
| unrelated twin | `shepherd-searxng` in `apps/shepherd.yaml` (Shepherd's own, unpinned `latest`) — do not conflate |

## Open WebUI settings are env-authoritative — names verified at tag v0.12.4

`ENABLE_PERSISTENT_CONFIG=false`, so the manifest wins on every boot and Admin Panel → Settings →
Web edits vanish on restart. Env names drift between releases; before changing anything, grep the
source at the running tag instead of trusting docs:

```bash
RAW=https://raw.githubusercontent.com/open-webui/open-webui/v0.12.4/backend/open_webui
curl -s $RAW/config.py            | grep -nE "WEB_SEARCH|SEARXNG|WEB_LOADER"
curl -s $RAW/routers/retrieval.py | grep -nE "'searxng'|RETRIEVAL_CONFIG_KEYS|SEARXNG_QUERY_URL"
curl -s $RAW/retrieval/web/searxng.py | head -70     # how the request is actually built
```

What is set (and why):

- `ENABLE_WEB_SEARCH=true`, `WEB_SEARCH_ENGINE=searxng` (engine key is the literal `searxng`).
  The older generation (`ENABLE_RAG_WEB_SEARCH`, `ENABLE_RAG_LOCAL_WEB_FETCH`) is gone in 0.12.x.
- `SEARXNG_QUERY_URL=http://searxng.default.svc.cluster.local:8080/search` — **the endpoint, not a
  `?q=<query>` template**: `search_searxng()` appends `q`, `format=json`, `pageno`, `language`,
  `safesearch`, `theme=simple` itself, and normalises legacy `<query>` URLs by *deleting* their
  query string. `SEARXNG_LANGUAGE=all`.
- `WEB_SEARCH_RESULT_COUNT=5` (default 3), `WEB_SEARCH_CONCURRENT_REQUESTS=3` (0 = unlimited
  parallel queries per turn).
- `BYPASS_WEB_SEARCH_WEB_LOADER=false` + `WEB_LOADER_ENGINE=""` → the built-in `SafeWebBaseLoader`
  (plain HTTP, no Playwright/Firecrawl/Tavily — playwright isn't in this image flavour) fetches each
  result page for real text. `WEB_LOADER_CONCURRENT_REQUESTS=3` is literally requests **per second**;
  `WEB_LOADER_TIMEOUT=15`, `ENABLE_WEB_LOADER_SSL_VERIFICATION=true`, `WEB_SEARCH_TRUST_ENV=false`
  (never route page fetches through an `HTTP(S)_PROXY` env var).
- `BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=true` — no `EMBEDDING_*` is configured, so the
  non-bypass path would embed all five pages with the default MiniLM into a throwaway `web-search-*`
  Chroma collection per search and hand back top-k chunks. Bypass returns the pages inline
  (measured: 5 docs × ~20k chars ≈ 25k tokens, free at 262k context). Flip it to `false` only
  together with an explicit embedding model; set `BYPASS_WEB_SEARCH_WEB_LOADER=true` to send
  snippets only.
- `ENABLE_WEB_SEARCH_CONFIRMATION=false` — no "search first?" prompt per message.
- Non-admin users get the feature because `USER_PERMISSIONS_FEATURES_WEB_SEARCH` defaults to true.

## Turning it on in the chat (for the human)

It is **per message/conversation**, not global: in the chat composer click the **globe / "Web
Search"** toggle (next to the `#` and tools buttons) — it turns blue and Open WebUI searches before
answering; sources appear as numbered citations under the reply and in the "Sources" strip. With
`WEB_SEARCH_ENGINE=searxng` the model also gets the fetched page text, not just snippets. If a
Function/pipeline model overrides `web_search`, the toggle is ignored for that model.

Direct browsing (no Open WebUI): `https://search.beluga-wyvern.ts.net` — tailnet only, same engine
set, `?q=` in the URL works, JSON off by default (`format=json` is allowed but the UI stays HTML).

## Verify it

```bash
# 1. SearXNG answers JSON *from inside the cluster* (this is the exact call Open WebUI makes)
kubectl exec -n default deploy/openwebui -c openwebui -- \
  curl -s -m 25 "http://searxng.default.svc.cluster.local:8080/search?q=kubernetes&format=json" | head -c 300

# 2. which engines answered / which got blocked
kubectl exec -n default deploy/openwebui -c openwebui -- sh -c \
 'curl -s -m 30 "http://searxng:8080/search?q=talos%20linux&format=json"' | \
 python3 -c "import json,sys,collections;d=json.load(sys.stdin);print(len(d['results']),d.get('unresponsive_engines'));print(collections.Counter(r['engine'] for r in d['results']))"

# 3. the tailnet front door
curl -s -o /dev/null -w '%{http_code}\n' https://search.beluga-wyvern.ts.net/    # 200

# 4. the whole Open WebUI path (needs a session JWT; see the open-webui skill for the Playwright
#    login that yields localStorage.token — never echo it). Endpoint: POST /api/v1/retrieval/
#    process/web/search, body {"queries":["..."]} → expect status:true, loaded_count 5, docs inline.
curl -s -H "Authorization: Bearer $TOK" -H 'Content-Type: application/json' \
  -d '{"queries":["Talos Linux 1.13 release notes"]}' \
  https://openwebui.beluga-wyvern.ts.net/api/v1/retrieval/process/web/search | head -c 200

# 5. logs / rollout
kubectl logs -n default deploy/searxng --tail=40
kubectl logs -n default deploy/openwebui -c openwebui --tail=20
```

## Troubleshooting

- **JSON returns 403 / "format not allowed":** `search.formats` in the ConfigMap must list `json`
  (`[html, json]`). SearXNG refuses any format not listed — the single most common cause. Fix in
  `apps/searxng.yaml`, push, reconcile; the pod restarts because the ConfigMap is a mounted volume
  *and* the template changed (edit the ConfigMap data, not the running pod).
- **"Web search failed" / 400 from Open WebUI:** usually the engine list produced zero usable
  results, or the SearXNG Service/pod is down. Run check 1 above — if it fails, look at
  `kubectl describe pod -n default -l app=searxng` and the logs.
- **Engines rate-limited / CAPTCHA (normal here, measured 2026-10-31):** `duckduckgo` answers
  `SearxEngineCaptchaException` and plain `google` gets "access denied" from this IP; results still
  arrive from `bing`, `brave` and `google cse` (which scrapes the CSE endpoint and needs no key).
  SearXNG suspends a failing engine (`ban_time_on_fail: 5s`, `SearxEngineCaptcha: 3600s`,
  `cf_SearxEngineCaptcha: 15d`, `recaptcha_SearxEngineCaptcha: 7d`) — a CAPTCHA therefore blacklists
  that engine for an hour *per pod*, so a restart is not a fix; `kubectl logs deploy/searxng | grep -iE "captcha|denied|Too Many"` is. If one engine is permanently useless, disable it in the
  `engines:` list (`disabled: true`) or trim the whole set with
  `use_default_settings: {engines: {keep_only: [bing, brave, google cse, wikipedia]}}`.
  Also: sending 5 concurrent queries per chat turn is how you earn a ban — keep
  `WEB_SEARCH_CONCURRENT_REQUESTS` at ~3.
- **Log noise that is NOT a problem:** `ERROR:searx.botdetection: X-Forwarded-For nor X-Real-IP
  header is set!` on every in-cluster query (Open WebUI hits the ClusterIP directly; browser traffic
  arrives through the tailscale proxy which does set it); `WARNING: ... missing config file:
  /etc/searxng/limiter.toml` (limiter is off, it still looks); the `ahmia:` / `torch:` "can't register
  engine" errors (engines whose optional deps aren't in the image); and the entrypoint's
  `"..." directory is not owned by "searxng:searxng"` warnings — expected because we run as 977 with
  kubelet-owned volumes (root:searxng, setgid, group-writable).
- **Pod won't start as non-root:** the image has **no** `USER` directive (it starts as root so the
  entrypoint can chown + `update-ca-certificates`). The searxng user is **uid/gid 977**
  (`/etc/passwd` in `searxng/base`). Keep `runAsUser/runAsGroup/fsGroup: 977`, and keep BOTH
  `__SEARXNG_CONFIG_PATH=/etc/searxng` (ConfigMap, read-only, must be a *directory* containing
  `settings.yml` or the entrypoint exits) and `__SEARXNG_DATA_PATH=/var/cache/searxng` mounted, or a
  writable path lands on the read-only rootfs.
- **`granian` memory creeping:** the image defaults to one worker per core; the manifest pins
  `GRANIAN_WORKERS=2` (~120 MiB each) with a 1 Gi limit. Raise it there, not with kubectl.
- **Rotate `secret_key`** (signs SearXNG's session cookies only; a missing one stops the app):
  ```bash
  umask 077; F=$(mktemp); printf '%s' "$(openssl rand -hex 32)" > "$F"; chmod 600 "$F"
  kubectl create secret generic searxng-secret -n default \
    --from-file=SEARXNG_SECRET="$F" --dry-run=client -o yaml \
  | sops encrypt --filename-override apps/searxng-secret.enc.yaml \
      --input-type yaml --output-type yaml /dev/stdin > apps/searxng-secret.enc.yaml
  rm -f "$F"    # never print the value; `sops` needs only the public age recipient in .sops.yaml
  ```
- **Upgrade the image:** SearXNG publishes date-stamped rolling tags (`2026.10.30-<sha>`), not
  semver, and `latest` floats. List tags at
  `https://hub.docker.com/v2/repositories/searxng/searxng/tags/?page_size=20`, then pin tag **plus
  index digest** (`docker buildx imagetools inspect docker.io/searxng/searxng:<tag>` — needs no
  daemon). Read `docs/admin/config_settings.rst` at that commit if settings keys moved.
- **After pushing:** `kubectl annotate -n flux-system gitrepository/flux-system reconcile.fluxcd.io/requestedAt=$(date +%s)`
  and the same for `kustomization/apps`. Deployment is `Recreate`, so SearXNG is briefly down;
  Open WebUI is unaffected until someone searches.
