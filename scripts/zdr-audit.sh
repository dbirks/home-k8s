#!/usr/bin/env bash
# ZDR audit for the llm.birks.dev serving path — see ZDR.md and issue #98.
#
# Read-only except for ONE optional canary request. Requires kubectl (context admin@home).
#
# Usage:
#   scripts/zdr-audit.sh                       # static checks (A) + metrics/label checks (C/D)
#   ZDR_API_KEY=llm_xxx scripts/zdr-audit.sh   # also sends a canary and does the log/metrics canary hunt (B/C)
#
# Env:
#   ZDR_API_KEY   API key for the public endpoint; enables the canary send (B)
#   ZDR_BASE_URL  default https://llm.birks.dev/v1
#   ZDR_MODEL     default qwen3.8-27b
#   NS            default default          (namespace of the LLMISVC engine pods)
#   CF_API_TOKEN + CF_ZONE_ID + CF_ACCOUNT_ID  enable the Cloudflare account-side checks (8)
set -uo pipefail

NS="${NS:-default}"
BASE_URL="${ZDR_BASE_URL:-https://llm.birks.dev/v1}"
MODEL="${ZDR_MODEL:-qwen3.8-27b}"
CANARY="ZDR_CANARY_$(cat /proc/sys/kernel/random/uuid 2>/dev/null || echo "$RANDOM$RANDOM")"
FAIL=0
red(){ printf '\033[31m%s\033[0m\n' "$*"; }
grn(){ printf '\033[32m%s\033[0m\n' "$*"; }
yel(){ printf '\033[33m%s\033[0m\n' "$*"; }
hdr(){ printf '\n=== %s ===\n' "$*"; }

# Engine pods = the vLLM containers (container name "main"). Router/scheduler (EPP) pods have no "main".
ENGINE_PODS=$(kubectl get pods -n "$NS" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
  | grep -E '\-llm-kserve-' | grep -vE 'router-scheduler' || true)
EPP_PODS=$(kubectl get pods -n "$NS" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
  | grep -E 'router-scheduler' || true)

hdr "A. Effective vLLM configuration (per running engine pod)"
if [ -z "$ENGINE_PODS" ]; then
  yel "no engine pods running (all scaled to zero) — cannot verify effective flags; wake a model and re-run"
else
  for p in $ENGINE_PODS; do
    args=$(kubectl get pod -n "$NS" "$p" -o jsonpath='{.spec.containers[?(@.name=="main")].args}' 2>/dev/null)
    lvl=$(kubectl get pod -n "$NS" "$p" -o jsonpath="{.spec.containers[?(@.name=='main')].env[?(@.name=='VLLM_LOGGING_LEVEL')].value}" 2>/dev/null)
    printf '  %s\n' "$p"
    for want in --no-enable-log-requests --no-enable-log-outputs --disable-uvicorn-access-log --no-enable-prefix-caching; do
      if grep -q -- "$want" <<<"$args"; then grn "    ok   $want"; else red "    MISS $want"; FAIL=1; fi
    done
    for bad in --otlp-traces-endpoint --collect-detailed-traces --enable-log-requests\" --enable-log-outputs\" --enable-prefix-caching\"; do
      if grep -q -- "$bad" <<<"$args"; then red "    BAD  present: $bad"; FAIL=1; fi
    done
    case "$lvl" in
      DEBUG|TRACE) red "    BAD  VLLM_LOGGING_LEVEL=$lvl"; FAIL=1;;
      "")          yel "    warn VLLM_LOGGING_LEVEL unset (preset default INFO expected)";;
      *)           grn "    ok   VLLM_LOGGING_LEVEL=$lvl";;
    esac
  done
fi

hdr "B. Canary-content test"
if [ -n "${ZDR_API_KEY:-}" ]; then
  echo "  sending canary $CANARY to $BASE_URL ($MODEL) ..."
  code=$(curl -s -o /tmp/zdr-canary-resp.json -w '%{http_code}' -m 240 -X POST "$BASE_URL/chat/completions" \
    -H "Authorization: Bearer $ZDR_API_KEY" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK. $CANARY\"}],\"max_tokens\":8}") || true
  echo "  HTTP $code"
  echo "  waiting 20s for logs/metrics to settle ..."; sleep 20
  # Refresh engine pod list (canary may have woken a model).
  ENGINE_PODS=$(kubectl get pods -n "$NS" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep -E '\-llm-kserve-' | grep -vE 'router-scheduler' || true)
  found=0
  for p in $ENGINE_PODS $EPP_PODS; do
    if kubectl logs -n "$NS" "$p" --all-containers --tail=5000 2>/dev/null | grep -qF "$CANARY"; then
      red "  LEAK canary found in pod logs: $p"; FAIL=1; found=1
    fi
  done
  for gwns in envoy-gateway-system; do
    for gp in $(kubectl get pods -n "$gwns" -o name 2>/dev/null | grep -E 'llm-public-gw|llm-ai-gw|kserve-llm-gw'); do
      if kubectl logs -n "$gwns" "$gp" --all-containers --tail=5000 2>/dev/null | grep -qF "$CANARY"; then
        red "  LEAK canary found in gateway logs: $gwns/$gp"; FAIL=1; found=1
      fi
    done
  done
  if kubectl logs -n llm-portal -l app=cloudflared --tail=5000 2>/dev/null | grep -qF "$CANARY"; then
    red "  LEAK canary found in cloudflared logs"; FAIL=1; found=1
  fi
  [ "$found" = 0 ] && grn "  canary absent from all component logs checked"
else
  yel "  ZDR_API_KEY not set — skipping canary send. Set it to exercise B fully."
fi

hdr "C. /metrics canary + content-label scan"
SUSPECT='prompt|completion|message|content|token_ids|request_id|trace_id|span_id|client_id|x_client_id|authorization'
scan_metrics(){ # $1 pod, $2 ns, $3 port, $4 path
  local body; body=$(kubectl exec -n "$2" "$1" -c "${5:-main}" -- sh -c "wget -qO- http://127.0.0.1:$3$4 2>/dev/null || curl -s http://127.0.0.1:$3$4" 2>/dev/null) || return 0
  [ -z "$body" ] && { yel "    (no /metrics from $1)"; return 0; }
  if grep -qF "$CANARY" <<<"$body"; then red "    LEAK canary in $1 metrics"; FAIL=1; fi
  local hits; hits=$(grep -vE '^#' <<<"$body" | grep -iE "\{[^}]*($SUSPECT)=" | head -3 || true)
  if [ -n "$hits" ]; then red "    SUSPECT content-bearing label in $1:"; sed 's/^/      /' <<<"$hits"; FAIL=1
  else grn "    ok   $1 metrics carry no content-bearing labels"; fi
}
for p in $ENGINE_PODS; do scan_metrics "$p" "$NS" 8000 /metrics main; done
# EPP metrics (:9090) and Envoy stats (:19001) exposed on those pods; best-effort.
for p in $EPP_PODS; do scan_metrics "$p" "$NS" 9090 /metrics "" ; done

hdr "D. Prometheus retained-label scan"
PROM="http://prometheus-server.monitoring.svc.cluster.local:80"
labeldump=$(kubectl run zdr-prom-probe --rm -i --restart=Never -n "$NS" --image=curlimages/curl:8.11.0 --quiet -- \
  sh -c "curl -s '$PROM/api/v1/labels'" 2>/dev/null || true)
if [ -n "$labeldump" ]; then
  bad=$(grep -oiE "\"($SUSPECT)\"" <<<"$labeldump" | tr -d '"' | sort -u || true)
  # Known-benign: Envoy Gateway's control-plane metric watchable_depth carries a static `message`
  # enum (gateway-status/infra-ir/provider-resources) naming its internal watch queue — not content.
  bad=$(grep -v '^message$' <<<"$bad" || true)
  if [ -n "$bad" ]; then red "  SUSPECT label names present in Prometheus: $bad"; FAIL=1
  else grn "  no content-bearing label names in Prometheus label set (message= is benign envoy-gw watchable_depth)"; fi
else
  yel "  could not reach Prometheus label API (probe pod failed); re-run when reachable"
fi

hdr "8. Cloudflare account-side (manual, needs CF_API_TOKEN)"
if [ -n "${CF_API_TOKEN:-}" ] && [ -n "${CF_ZONE_ID:-}" ]; then
  CF_API="https://api.cloudflare.com/client/v4"; AUTH=(-H "Authorization: Bearer $CF_API_TOKEN")
  echo "  WAF managed-rule matched_data payload logging (expect none for the LLM route):"
  curl -fsS "$CF_API/zones/$CF_ZONE_ID/rulesets/phases/http_request_firewall_managed/entrypoint" "${AUTH[@]}" \
    | jq '.result.rules[]? | select(.action_parameters.matched_data? != null) | {description,id}' 2>/dev/null || yel "    (query failed)"
  echo "  Logpull retention flag (expect false):"
  curl -fsS "$CF_API/zones/$CF_ZONE_ID/logs/control/retention/flag" "${AUTH[@]}" | jq '.result' 2>/dev/null || yel "    (query failed)"
  if [ -n "${CF_ACCOUNT_ID:-}" ]; then
    echo "  DLP payload logging (non-null public_key = configured):"
    curl -fsS "$CF_API/accounts/$CF_ACCOUNT_ID/dlp/payload_log" "${AUTH[@]}" | jq '.result | {public_key}' 2>/dev/null || yel "    (query failed)"
  fi
  yel "  Also confirm in dashboard: request_body_buffering=none, response_body_buffering=none, /v1* cache bypass, no Logpush of bodies/headers."
else
  yel "  CF_API_TOKEN/CF_ZONE_ID not set — Cloudflare posture NOT audited (see ZDR.md 'not provable from git')."
fi

hdr "Result"
if [ "$FAIL" = 0 ]; then grn "ZDR checks passed (subject to skipped sections above)."; else red "ZDR checks FAILED — see BAD/MISS/LEAK/SUSPECT lines."; fi
exit "$FAIL"
