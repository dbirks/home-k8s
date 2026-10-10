#!/bin/bash
# Offline test harness for the gpu-power-limit enforcer script (issue #146 prep).
# Extracts the DaemonSet's /bin/sh payload, runs it against a FAKE nvidia-smi,
# and asserts exactly which power limits it would apply. No GPU, no cluster.
#   usage: .agents/skills/gpu-power-management/scripts/test-power-map.sh
set -uo pipefail
root=$(cd "$(dirname "$0")/../../../.." && pwd)

MANIFEST=${MANIFEST:-$root/apps/gpu-power-limit.yaml}
export WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

# pull the block-list scalar under "command: [/bin/sh, -c, |]" + the manifest's env defaults
python3 - "$MANIFEST" "$WORK/enforce.sh" "$WORK/env.defaults" <<'PY'
import sys, yaml
docs=[d for d in yaml.safe_load_all(open(sys.argv[1])) if d]
ds=[d for d in docs if d.get('kind')=='DaemonSet'][0]
c=ds['spec']['template']['spec']['containers'][0]
cmd=c['command']; assert cmd[0]=='/bin/sh' and cmd[1]=='-c', cmd
open(sys.argv[2],'w').write(cmd[2])
with open(sys.argv[3],'w') as f:
    for e in c.get('env',[]):
        if 'value' in e: f.write(f"{e['name']}={e['value']}\n")
print(f"extracted {len(cmd[2])} bytes of /bin/sh from {sys.argv[1]}")
PY

# FAKE_GPUS = ';' separated "idx|uuid|name|limit"; honours the requested --query-gpu field list
mkdir -p "$WORK/bin"
cat >"$WORK/bin/nvidia-smi" <<'EOF'
#!/bin/bash
if [[ "$1" == --query-gpu=* ]]; then
  fields=${1#--query-gpu=}
  IFS=';' read -ra G <<< "$FAKE_GPUS"
  for g in "${G[@]}"; do
    IFS='|' read -r i u n p <<< "$g"
    out=""
    IFS=, read -ra F <<< "$fields"
    for f in "${F[@]}"; do
      case "$f" in
        index) v="$i" ;; uuid) v="$u" ;; name) v="$n" ;;
        power.limit) v="$p" ;; power.max_limit) v="600.00" ;;
        power.used) v="180.00" ;; persistence_mode) v="Enabled" ;;
        *) v="n/a" ;;
      esac
      if [[ "$1" == *nounits* && "$f" == power.* ]]; then v=${v%.*}; v="$v.00"; fi
      [ -n "$out" ] && out="$out, $v" || out="$v"
    done
    echo "$out"
  done
  exit 0
fi
case "$*" in
  "-pm 1")      echo "-pm 1" >>"$WORK/pm";;
  "-i "*"-pl "*) echo "$*" >>"$WORK/calls";;
  *)            echo "ALL $*" >>"$WORK/calls";;
esac
exit 0
EOF
chmod +x "$WORK/bin/nvidia-smi"

PRO="0|GPU-9d6cf286-ce1e-96ef-d3ac-1dd0f303bc11|NVIDIA RTX PRO 6000 Blackwell Workstation Edition|400.00"
FIFTY="1|GPU-11111111-2222-3333-4444-555555555555|NVIDIA GeForce RTX 5090|575.00"

fails=0
run() { # run <label> <FAKE_GPUS> <expected -pl calls, space-joined> [ENV=VAL...]
  local label="$1" gpus="$2" expect="$3"; shift 3
  : >"$WORK/calls"; : >"$WORK/pm"
  ( export PATH="$WORK/bin:$PATH" FAKE_GPUS="$gpus" HEARTBEAT_PASSES=1 INTERVAL=1
    while IFS= read -r kv; do [ -n "$kv" ] && export "$kv"; done <"$WORK/env.defaults"
    while [ $# -gt 0 ]; do export "$1"; shift; done
    timeout 3 sh "$WORK/enforce.sh" >"$WORK/log.$label" 2>&1 )
  local got; got=$(sort "$WORK/calls" | tr '\n' ' ' | sed -e 's/ *$//')
  if [ "$got" = "$expect" ]; then
    echo "PASS  $label  →  ${got:-<no power-limit writes>}"
  else
    echo "FAIL  $label"; echo "        expected: [$expect]"; echo "        got:      [$got]"
    grep -E "gpu-power-limit" "$WORK/log.$label" | tail -6 | sed -e 's/^/        | /'
    fails=$((fails+1))
  fi
  grep -E "WARN|FAILED" "$WORK/log.$label" | sed -e 's/^/        note: /' || true
}

echo "== gpu-power-limit enforcer =="
# today's reality: one card, empty map, already at its 400W limit -> writes NOTHING
run "single-card-noop"       "$PRO" ""                 GPU_POWER_MAP=
# name substring routes the PRO card to 350
run "name-substring"         "$PRO" "-i 0 -pl 350"     GPU_POWER_MAP="PRO 6000=350"
# UUID prefix (the stable matcher)
run "uuid-prefix"            "$PRO" "-i 0 -pl 380"     GPU_POWER_MAP="GPU-9d6cf286=380"
# exact nvidia-smi index
run "index-exact"            "$PRO" "-i 0 -pl 360"     GPU_POWER_MAP="0=360"
# documented trap: a loose 'RTX' matcher also hits the PRO 6000 (first match wins)
run "trap-rtx-matches-both"  "$PRO;$FIFTY" "-i 0 -pl 300 -i 1 -pl 300"  GPU_POWER_MAP="RTX=300"
# heterogeneous: PRO stays at 400 (untouched), the 5090 gets capped to 300
run "two-cards"              "$PRO;$FIFTY" "-i 1 -pl 300"  GPU_POWER_MAP="PRO 6000=400,RTX 5090=300"
# renumbered so the 5090 is index 0: the matcher follows the CARD, not the index
FIFTY0="0|GPU-11111111-2222-3333-4444-555555555555|NVIDIA GeForce RTX 5090|575.00"
PRO1="1|GPU-9d6cf286-ce1e-96ef-d3ac-1dd0f303bc11|NVIDIA RTX PRO 6000 Blackwell Workstation Edition|400.00"
run "two-cards-rev-order"    "$FIFTY0;$PRO1" "-i 0 -pl 320"  GPU_POWER_MAP="RTX 5090=320"
# matcher selects nothing -> warn + fall back to TARGET_WATTS (400 == already there)
run "unmatched-matcher"      "$PRO" ""                 GPU_POWER_MAP="5090=300"
# unmatched matcher on the fallback-sensitive card -> falls back to 400 and writes it
run "fallback-writes"        "$FIFTY" "-i 1 -pl 400"   GPU_POWER_MAP="PRO 6000=400"
# DRY_RUN must never write a power limit
run "dry-run"                "$PRO" ""                 GPU_POWER_MAP="PRO 6000=300" DRY_RUN=1

[ $fails -eq 0 ] && { echo; echo "ALL PASS"; } || { echo; echo "$fails FAILED"; exit 1; }
