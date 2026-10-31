#!/usr/bin/env bash
# Store a Jellyfin API key as the SOPS secret `jellyfin-api-key` (key JELLYFIN_API_KEY) without it ever
# touching the screen, shell history, chat or git plaintext. Mint the key first in Jellyfin:
# Dashboard -> API Keys -> + (name it "youtarr"). Then run this from the repo root and paste at the prompt.
# Writes apps/jellyfin-api-key.enc.yaml; commit it and let Flux apply it.
set -euo pipefail
SOPS=${SOPS:-$(command -v sops || echo /tmp/claude-1000/-home-david-dev-home-k8s/fdae59fb-bbf8-43ea-a8dd-d7bad5ef2763/scratchpad/bin/sops)}
OUT=apps/jellyfin-api-key.enc.yaml
[ -x "$SOPS" ] || { echo "sops not found; set SOPS=/path/to/sops"; exit 1; }
umask 077; T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
read -rsp "Paste the Jellyfin API key (input hidden): " K; echo
[[ "$K" =~ ^[0-9a-f]{32}$ ]] || { echo "that doesn't look like a Jellyfin key (32 hex chars)"; exit 1; }
printf %s "$K" > "$T/k"; unset K
code=$(kubectl exec -i deploy/jellyfin -- sh -c 'read -r k; curl -s -o /dev/null -w "%{http_code}" -H "Authorization: MediaBrowser Token=$k" http://localhost:8096/System/Info' < <(cat "$T/k"; echo))
[ "$code" = 200 ] || { echo "Jellyfin rejected the key (HTTP $code)"; exit 1; }
kubectl create secret generic jellyfin-api-key --namespace=default --from-file=JELLYFIN_API_KEY="$T/k" --dry-run=client -o yaml \
  | "$SOPS" encrypt --filename-override "$OUT" --input-type yaml --output-type yaml /dev/stdin > "$OUT"
grep -qF "$(cat "$T/k")" "$OUT" && { rm -f "$OUT"; echo "refusing: plaintext in output"; exit 1; }
echo "key verified against Jellyfin and written to $OUT (encrypted). Tell Claude it's done."
