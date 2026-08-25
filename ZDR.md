# Zero-Data-Retention (ZDR) policy — llm.birks.dev serving path

Tracking issue: **#98**. This file is the human-readable invariant so future vLLM / KServe / Envoy /
Cloudflare upgrades preserve the posture instead of silently regressing to a new default.

## Working definition (customer-content ZDR)

Prompts and model responses are **not retained after a request is processed**, and customer content is
excluded from any retained abuse-monitoring / application-state storage. Aggregate operational telemetry
that contains no customer content **may** be retained (this is not "no observability").

**Customer content** (must never be retained): prompt/message text, completion/output text, input/output
token IDs, image/audio/file contents, user-supplied tool arguments/results, raw request/response bodies,
API keys / Authorization values, user-provided headers, and any directly-identifying per-request value
derived from those.

**Allowed retained telemetry** (aggregate/system only, no customer content as labels/exemplars):
model/pool name, replica/readiness state, aggregate request counts, status/error class, queue depth,
token-count counters/histograms, request/response size histograms, latency/TTFT/TPOT histograms,
GPU/KV-cache utilization, infrastructure metrics.

Request path: `Cloudflare edge -> cloudflared -> llm-public-gw (Envoy Gateway) -> llm-ai-gw (Envoy AI
Gateway) -> InferencePool / EPP -> vLLM`.

## What enforces it (in git)

| Layer | Control | Where |
|-------|---------|-------|
| vLLM (all 6 models) | `--no-enable-log-requests`, `--no-enable-log-outputs`, `--disable-uvicorn-access-log`, `--no-enable-prefix-caching`, `VLLM_LOGGING_LEVEL=INFO` | `kserve/llmisvc-*.yaml` |
| vLLM tracing | no `--otlp-traces-endpoint`, no `--collect-detailed-traces` (asserted, not set) | `kserve/llmisvc-*.yaml` |
| KServe preset (defense-in-depth, only used if a future LLMISVC omits inline command/args) | preset injects the same ZDR flags via `ACCESS_LOG_ARGS` | `kserve/llmisvc-runtime-configs-s2z.yaml` |
| Envoy proxy (both LLM gateways) | `spec.telemetry.accessLog.disable: true` (Prometheus proxy metrics kept for KEDA) | `apps/llm-public-gateway.yaml`, `apps/llm-ai-gateway.yaml` |
| cloudflared | `--loglevel info` (never debug/trace); logs no bodies | `apps/cloudflared.yaml` |
| Prometheus | 15d retention kept; only aggregate LLM telemetry scraped; guardrail test rejects content-bearing labels | `infra/prometheus.yaml` + `scripts/zdr-audit.sh` |

### Why prefix caching is OFF

vLLM defaults prefix caching ON and retains completed-request KV blocks for later matching prefixes.
That is post-request retention of prompt-derived state and a multi-tenant timing surface, so ZDR turns it
off. `cache_salt` is **not** equivalent (it isolates reuse but still retains prompt-derived KV). This has a
throughput cost on multi-turn/shared-prefix traffic; it is an accepted ZDR tradeoff. Ordinary active-request
KV (necessary transient processing) is unaffected — this is only about reusing completed-request prefixes.

## What is NOT provable from git (manual, account-side)

TLS terminates at Cloudflare's edge, so end-to-end ZDR cannot be fully proven from this repo. Re-run the
Cloudflare section of `scripts/zdr-audit.sh` (needs a scoped `CF_API_TOKEN`) after any zone/tunnel change,
and confirm: no request/response body logging, no Worker persisting payloads, no Logpush/Logpull field or
destination capturing customer content or Authorization, no WAF managed-rule `matched_data` payload logging
for the LLM route, and no DLP payload logging on this path. If Cloudflare cannot make a given guarantee,
document the limitation here rather than calling the whole path ZDR unqualified.

## Verifying

Run `scripts/zdr-audit.sh` (see its header for the canary and Cloudflare flags). It covers: effective vLLM
flags per pod (A), a canary-content search across component logs (B), a canary/label search across `/metrics`
and Prometheus (C), and a re-check after ingestion (D).
