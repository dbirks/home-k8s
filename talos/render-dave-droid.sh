#!/usr/bin/env bash
# Render the dave-droid worker config. Output holds cluster secrets: write it OUTSIDE the repo.
#   talos/render-dave-droid.sh /path/outside/repo/dave-droid.yaml
set -euo pipefail
out=${1:?output path (outside the repo)}
cd "$(dirname "$0")/.."
umask 077
talosctl machineconfig patch _newconfig/worker.yaml --patch @talos/patches/dave-droid-worker.yaml -o "$out"
yq -i '(select(.machine) | .machine.install) |= del(.disk)' "$out"
yq -i '(select(.kind == "HostnameConfig")) |= (del(.auto) | .hostname = "dave-droid")' "$out"
talosctl validate --mode metal --config "$out"
