#!/usr/bin/env bash
# gpu-wedge-watchdog.sh — canonical tracked copy (home-k8s; skill: gpu-wedge-recovery)
#
# Installs at ~/.local/share/home-k8s-auto/gpu-wedge-watchdog.sh and fires every 2 min
# from the user systemd timer gpu-wedge-watchdog.timer on David's workstation.
#
# Detect an NVIDIA FULLCHIP_RESET wedge (RTX PRO 6000 Blackwell GSP crash, home-k8s #46)
# and recover it with a BMC/IPMI power-cycle. A warm reboot does NOT clear a
# FULLCHIP_RESET; only `talosctl reboot --mode powercycle` re-enumerates the card.
#
# MULTI-GPU (issue #146 prep): the original trigger was `capacity nvidia.com/gpu == 0`.
# On a 2-GPU node one wedged card still leaves capacity > 0 (HAMi keeps advertising the
# survivor), so a single-card wedge would have gone unrecovered forever. Wedge detection
# is now PER GPU UUID: the expected set lives in $UUIDFILE and a UUID that disappears
# from `nvidia-smi --query-gpu=uuid` is a wedge signal. The hard requirement — FULLCHIP_RESET
# in the node's dmesg — is unchanged, so this can only ever ADD precision, never loosen
# the guard. Re-learn the expected set after any intentional hardware change:
#   gpu-wedge-watchdog.sh --learn     # capture current GPUs as the expected set
#   gpu-wedge-watchdog.sh --status    # what the watchdog sees right now
#   gpu-wedge-watchdog.sh --check     # evaluate + log, never act (safe to run by hand)
set -uo pipefail

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
TC=${WATCHDOG_TALOSCONFIG:-/home/david/dev/home-k8s/_newconfig/talosconfig}
NODE=${WATCHDOG_NODE:-talos-210-73x}
DIR=/home/david/.local/share/home-k8s-auto
LOG="$DIR/gpu-wedge-watchdog.log"
STATE="$DIR/gpu-wedge-watchdog.last"   # unix ts of last power-cycle
LOCK="$DIR/gpu-wedge-watchdog.lock"
UUIDFILE="${GPU_WEDGE_UUID_FILE:-$DIR/gpu-uuids.list}"   # expected GPU UUIDs (one per line)

CONFIRM_SECONDS=120   # wedge must persist this long before we act (skip transients)
COOLDOWN=1800         # min 30 min between power-cycles (reboot-loop guard)
POD_NS=${GPU_WEDGE_POD_NS:-default}                  # where the always-on GPU pod lives
POD=${GPU_WEDGE_POD:-ds/gpu-power-limit}   # privileged + NVIDIA_VISIBLE_DEVICES=all

log(){ echo "[$(date -Is)] $*" >>"$LOG"; }

node_ip(){ kubectl get node "$NODE" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null; }
gpu_cap(){ kubectl get node "$NODE" -o jsonpath='{.status.capacity.nvidia\.com/gpu}' 2>/dev/null; }

# Live GPU UUIDs, read through the always-on, privileged, GPU-visible enforcer pod.
# Prints nothing and returns non-zero if the query itself failed (node down, pod gone,
# driver dead) — callers MUST NOT read that as "GPUs missing", that is a different fault.
live_uuids(){
  local out
  out=$(timeout 25 kubectl -n "$POD_NS" exec "$POD" -- \
        nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' \r' | sed '/^$/d') || return 1
  [ -n "$out" ] || return 1          # empty output = unusable answer, not zero GPUs
  echo "$out"
}

expected_uuids(){ [ -f "$UUIDFILE" ] && sed -e 's/[[:space:]]//g' "$UUIDFILE" | grep -v '^#' | sed '/^$/d'; }

learn_uuids(){
  local l; if ! l=$(live_uuids); then log "cannot learn GPU set: enforcer pod query failed"; return 1; fi
  { echo "# expected GPU UUIDs on $NODE — captured $(date -Is)"; echo "$l"; } >"$UUIDFILE"
  log "learned GPU set ($(echo "$l" | wc -l) device(s)) into $UUIDFILE"
}

# UUIDs we expect but cannot see. Only meaningful when a live query succeeded.
missing_uuids(){
  local exp got
  exp=$(expected_uuids); [ -n "$exp" ] || return 0        # no expectation recorded -> nothing to compare
  got=$(live_uuids) || return 0                           # live query failed: not evidence of a wedge
  comm -23 <(echo "$exp" | sort) <(echo "$got" | sort)
}

# The GPU is not serviceable if EITHER the node advertises 0 GPUs, OR the HAMi
# device-plugin can't run, OR an expected UUID has fallen off the bus. The
# device-plugin StartError ("error getting device handle for index '0': Unknown
# Error") is the RELIABLE EARLY signal — node capacity lags and can still read 10
# (stale) for a while after a wedge, so keying on cap=0 alone misses it for minutes.
gpu_unserviceable(){
  [ "$(gpu_cap)" = "0" ] && return 0
  local line; line=$(kubectl get pods -n kube-system --no-headers 2>/dev/null | grep -E '^hami-device-plugin')
  [ -n "$line" ] || return 1
  echo "$line" | grep -qE 'RunContainerError|CrashLoopBackOff|StartError|Error' && return 0
  [ "$(echo "$line" | awk '{print $2}')" = "2/2" ] || return 0
  return 1
}

# Wedged == kernel shows FULLCHIP_RESET (the hard proof the card is in reset) AND at
# least one device is unserviceable. dmesg clears on the recovery reboot, so stale
# assertions can't linger to cause a double power-cycle.
is_wedged(){
  local ip; ip=$(node_ip); [ -n "$ip" ] || return 1
  timeout 20 talosctl --talosconfig "$TC" -e "$ip" -n "$ip" dmesg 2>/dev/null \
    | grep -q 'GPU_IN_FULLCHIP_RESET' || return 1
  gpu_unserviceable && return 0
  local miss; miss=$(missing_uuids)
  if [ -n "$miss" ]; then
    log "GPU(s) absent from the live bus: $(echo "$miss" | tr '\n' ' ')"
    return 0
  fi
  return 1
}

status(){
  echo "node            : $NODE  ip=$(node_ip)  capacity nvidia.com/gpu=$(gpu_cap)"
  echo "expected GPUs   : $(expected_uuids | tr '\n' ' ')"
  echo "live GPUs       : $(live_uuids | tr '\n' ' ' || echo '<query failed>')"
  echo "missing         : $(missing_uuids | tr '\n' ' ')"
  echo "device-plugin   : $(kubectl get pods -n kube-system --no-headers 2>/dev/null | grep -E '^hami-device-plugin')"
  echo "applied caps    :"
  timeout 25 kubectl -n "$POD_NS" exec "$POD" -- nvidia-smi \
    --query-gpu=index,uuid,name,power.limit,persistence_mode --format=csv,noheader 2>&1 | sed -e 's/^/  /'
}

case "${1:-}" in
  --learn)  learn_uuids; exit $?;;
  --status) status; exit 0;;
esac

# Single instance only; a stale run holding the confirm window must not stack.
exec 9>"$LOCK"
flock -n 9 || exit 0

# First run on a fresh install: record the GPU set so future runs can notice a
# device disappearing. Never blocks detection — missing_uuids() no-ops until then.
[ -s "$UUIDFILE" ] || learn_uuids || true

ACT=1; [ "${1:-}" = "--check" ] && ACT=0

is_wedged || exit 0
log "WEDGE suspected (FULLCHIP_RESET + unserviceable GPU). Confirming over ${CONFIRM_SECONDS}s..."

waited=0
while [ "$waited" -lt "$CONFIRM_SECONDS" ]; do
  sleep 30; waited=$((waited+30))
  is_wedged || { log "cleared during confirm window (${waited}s) — no action."; exit 0; }
done

if [ "$ACT" = 0 ]; then
  log "--check: STILL WEDGED after ${CONFIRM_SECONDS}s but acting is disabled — no power-cycle."
  exit 2
fi

now=$(date +%s)
last=$(cat "$STATE" 2>/dev/null || echo 0)
if [ $((now - last)) -lt "$COOLDOWN" ]; then
  log "still wedged but within cooldown ($((now-last))s < ${COOLDOWN}s) — NOT power-cycling. Needs a human look."
  exit 0
fi

ip=$(node_ip)
log "CONFIRMED wedge. Triggering BMC power-cycle: talosctl reboot --mode powercycle (node $ip)"
echo "$now" >"$STATE"
timeout 60 talosctl --talosconfig "$TC" -e "$ip" -n "$ip" reboot --mode powercycle >>"$LOG" 2>&1
log "power-cycle command returned rc=$? (node cycling; GPU should re-enumerate in ~2-3 min)"
