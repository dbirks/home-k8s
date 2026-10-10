#!/usr/bin/env bash
# Safety tests for gpu-wedge-watchdog.sh (issue #146 prep). Drives the REAL script
# against the live cluster but ONLY in --check mode, which cannot power-cycle, and
# mocks `talosctl dmesg` so the wedge gate can be exercised without crashing a node.
#   usage: .agents/skills/gpu-wedge-recovery/scripts/test-watchdog.sh
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
WD="$here/gpu-wedge-watchdog.sh"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
mkdir -p "$W/bin"
export GPU_WEDGE_UUID_FILE="$W/uuids.list"
export WATCHDOG_TALOSCONFIG="$W/no-such-talosconfig"     # the real one is never needed: no reboot in these tests
LIVE=$(bash "$WD" --status 2>/dev/null | sed -n 's/^live GPUs  *: //p' | tr -d ' ')
[ -n "$LIVE" ] || { echo "SKIP: cannot read live GPUs from the cluster (offline?)"; exit 0; }
echo "live GPU UUID: $LIVE"

mock_dmesg(){ printf '#!/bin/bash\ncase "$*" in *dmesg*) echo "%s";; esac\nexit 0\n' "$1" >"$W/bin/talosctl"; chmod +x "$W/bin/talosctl"; }

fails=0
check(){ # check <label> <expect-exit> <grep-pattern-file> ...
  local label=$1 want=$2; shift 2
  local rc=0; ( cd "$W" && timeout 200 env PATH="$W/bin:$PATH" WATCHDOG_LOG_DIR="$W" bash "$WD" --check ) >"$W/out.$label" 2>&1 || rc=$?
  if [ "$rc" = "$want" ]; then echo "PASS  $label (rc=$rc)"; else
    echo "FAIL  $label (rc=$rc, wanted $want)"; sed -e 's/^/        | /' "$W/out.$label" | tail -5
    grep -h . "$W"/gpu-wedge-watchdog.log 2>/dev/null | tail -4 | sed -e 's/^/        ! /'
    fails=$((fails+1)); fi
}

echo "== gate 1: no FULLCHIP_RESET in dmesg => never a wedge, even with a GPU 'missing' =="
printf '%s\n%s\n' "$LIVE" "GPU-00000000-0000-0000-0000-000000000000" >"$GPU_WEDGE_UUID_FILE"
mock_dmesg "nvrm: (PCI:0000:06:00:0) Permissions : via GSP firmware"
check "dmesg-gate-blocks-action" 0

echo "== gate 2: FULLCHIP_RESET + an expected UUID off the bus => confirmed wedge, --check must still refuse to act =="
: >"$W"/$(basename "$(bash -c 'echo /home/david/.local/share/home-k8s-auto/gpu-wedge-watchdog.log')") 2>/dev/null || true
mock_dmesg "NVRM: NV_ERR_GPU_IN_FULLCHIP_RESET assertion failed"
# --check cannot power-cycle by construction; rc=2 means it reached the confirmed-wedge branch.
check "missing-uuid-detected" 2

echo "== gate 3: FULLCHIP_RESET but every expected GPU is present and the plugin is healthy => no wedge =="
printf '%s\n' "$LIVE" >"$GPU_WEDGE_UUID_FILE"
check "healthy-set-no-action" 0

[ $fails -eq 0 ] && { echo; echo "ALL PASS"; } || { echo; echo "$fails FAILED"; exit 1; }
