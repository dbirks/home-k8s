#!/usr/bin/env bash
# hami-rollout-guard.sh — the single-node HAMi rollout trap (#146/#147).
#
# The chart requires one hami-scheduler pod per hostname; on a one-node cluster a
# RollingUpdate surge pod can never be placed, so any HAMi value change times out after
# 5 min, gets rolled back by Flux, and wedges the `prereqs` Kustomization — which blocks
# `infra` and `apps` too, i.e. NOTHING in the repo reconciles while it loops.
#
#   --check     (default) read-only: strategy, wedged-rollout detection, chain health
#   --enforce    set Deployment/hami-scheduler strategy.type=Recreate (durable: the chart
#                renders no spec.strategy, so Helm never rewrites it)
#   --post       post-rollout GPU invariants: enforcer pod + caps + scheduler flags + serving
#   --unstick    do the documented recovery for a CONFIRMED wedge (refuses otherwise)
set -uo pipefail
NS=kube-system
ACT=${1:---check}

hr_cond(){ kubectl -n flux-system get helmrelease hami -n $NS -o jsonpath='{range .status.conditions[*]}{.type}={.status} {.reason}: {.message}{"\n"}{end}' 2>/dev/null; }
strategy(){ kubectl -n $NS get deploy hami-scheduler -o jsonpath='{.spec.strategy.type}' 2>/dev/null; }
pending(){ kubectl -n $NS get pods --no-headers 2>/dev/null | awk '/hami-scheduler/ && /Pending/ {print $1, $3, $6}'; }
blocked(){ flux -n flux-system get kustomization 2>/dev/null | awk 'NR>1{printf "  %-12s ready=%-6s %s\n",$1,$4,substr($0,index($0,$5))}' | grep -E "prereqs|infra|apps|flux-system "; }

wedge_confirmed(){
  [ -n "$(pending)" ] && hr_cond | grep -qE "UpgradeFailed|timeout waiting|Progressing"
}

case "$ACT" in
  --check)
    echo "hami-scheduler strategy : $(strategy)   (want: Recreate)"
    echo "pending scheduler pod   : $(pending | tr '\n' ' ')"
    echo "helmrelease:"; hr_cond | sed -e 's/^/  /'
    echo "kustomization chain:"; blocked
    if [ "$(strategy)" != "Recreate" ]; then
      echo "!! strategy is NOT Recreate — the next HAMi value change will wedge prereqs/infra/apps."
      echo "   fix: $0 --enforce"
    elif [ -n "$(pending)" ]; then
      echo "!! a scheduler pod is Pending — rollout is wedged NOW. fix: $0 --unstick"
    else
      echo "ok: single-node-safe, no wedge."
    fi
    ;;

  --enforce)
    before=$(strategy)
    kubectl -n $NS patch deploy hami-scheduler --type=strategic \
      -p '{"spec":{"strategy":{"type":"Recreate","rollingUpdate":null}}}'
    echo "strategy: $before -> $(strategy)"
    ;;

  --post)
    echo "enforcer pod   : $(kubectl get pod -l app=gpu-power-limit --no-headers 2>/dev/null | awk '{print $1,$3,$5}')"
    kubectl logs -l app=gpu-power-limit --tail=3 2>&1 | sed -e 's/^/    /'
    echo "applied caps   :"
    kubectl exec ds/gpu-power-limit -- nvidia-smi --query-gpu=index,uuid,name,power.limit,persistence_mode --format=csv,noheader 2>&1 | sed -e 's/^/    /'
    echo "rogue enforcers   : $(kubectl -n $NS get ds --no-headers 2>/dev/null | grep -c nvidia-power-cap) (want 0)"
    echo "scheduler flags :"
    kubectl -n $NS get deploy hami-scheduler -o yaml 2>/dev/null | grep -E "scheduler-policy" | tr -d ' ' | sed -e 's/^/    /'
    N=$(kubectl get node -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)   # a bare `get node` is a LIST: jsonpath needs .items[0] or a name
    reg=$(kubectl get node "$N" -o 'jsonpath={.metadata.annotations.hami\.io/node-nvidia-register}' 2>/dev/null)
    cap=$(kubectl get node "$N" -o 'jsonpath={.status.capacity.nvidia\.com/gpu}' 2>/dev/null)
    echo "register        : $reg"
    echo "gpu capacity    : ${cap:-?} units"
    echo "pennyroyal      : $(kubectl get pod -l app=pennyroyal-flashnext --no-headers 2>/dev/null | awk '{print $1,$2,$3,$5}')"
    kubectl exec ds/gpu-power-limit -- curl -s -m 6 http://pennyroyal-flashnext.default.svc.cluster.local:8001/v1/models 2>/dev/null \
      | head -c 160 | sed -e 's/^/    /' ; echo
    ;;

  --unstick)
    if ! wedge_confirmed; then
      echo "refusing: no confirmed wedge (no Pending hami-scheduler pod / no UpgradeFailed)."
      echo "run $0 --check first. If prereqs is stuck for another reason, do NOT use this."
      exit 1
    fi
    echo "confirmed wedge. 1) forcing Recreate so the new pod can be placed"
    "$0" --enforce
    echo "2) resuming the HelmRelease if it was suspended"
    suspended=$(kubectl -n flux-system get helmrelease hami -n $NS -o jsonpath='{.spec.suspend}' 2>/dev/null)
    [ "$suspended" = "true" ] && flux -n flux-system resume hr hami -n $NS
    echo "3) driving the upgrade, then the blocked chain in dependency order"
    flux -n flux-system reconcile hr hami -n $NS --timeout 6m 2>&1 | tail -2
    for k in prereqs infra apps; do flux -n flux-system reconcile kustomization $k --timeout 6m 2>&1 | tail -1; done
    "$0" --check; "$0" --post
    ;;

  *) echo "usage: $0 [--check|--enforce|--post|--unstick]"; exit 2;;
esac
