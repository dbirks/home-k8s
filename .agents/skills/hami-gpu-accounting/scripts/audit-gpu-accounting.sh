#!/usr/bin/env bash
# Audit HAMi GPU accounting: which workloads claim the GPU, and which of them forgot the
# mandatory nvidia.com/gpumem cap (the defaultMemory:0 trap = they seize the WHOLE card).
#   usage: audit-gpu-accounting.sh [--live-only]
# Manifest scan is offline; the live scan needs the cluster.
set -uo pipefail
root=$(cd "$(dirname "$0")/../../../.." && pwd)
LIVE_ONLY=0; [ "${1:-}" = "--live-only" ] && LIVE_ONLY=1

if [ "$LIVE_ONLY" = 0 ]; then
python3 - "$root" <<'PY'
import os,sys,yaml,collections
root=sys.argv[1]
rows=[]
for sub in ("apps","infra","prereqs"):
    for dirpath,_,files in os.walk(os.path.join(root,sub)):
        for fn in sorted(files):
            if not fn.endswith((".yaml",".yaml.hold")): continue
            p=os.path.join(dirpath,fn)
            try: docs=[d for d in yaml.safe_load_all(open(p)) if isinstance(d,dict)]
            except Exception as e:
                rows.append(("PARSE",f"{os.path.relpath(p,root)}",str(e)[:60],"","","")); continue
            for d in docs:
                kind=d.get("kind",""); name=(d.get("metadata") or {}).get("name","")
                spec=(d.get("spec") or {})
                tpl=spec.get("template") or {}
                pod=(tpl.get("spec") or {}) if isinstance(tpl,dict) else {}
                replicas=spec.get("replicas", 1 if kind in ("Deployment","StatefulSet") else None)
                parked = (str(fn).endswith(".hold") or replicas==0)
                for c in (pod.get("containers") or [])+(pod.get("initContainers") or []):
                    res=(c.get("resources") or {})
                    allr={**(res.get("requests") or {}),**(res.get("limits") or {})}
                    gk=[k for k in allr if k.startswith("nvidia.com/") and not k.endswith("gpumem") and not k.endswith("gpucores")]
                    if not gk: continue
                    mem=allr.get("nvidia.com/gpumem"); cores=allr.get("nvidia.com/gpucores")
                    rows.append(("GPU", f"{os.path.relpath(p,root)}", f"{kind}/{name} ctr={c.get('name')}",
                                 f"gpu={ {k:allr[k] for k in gk} }", f"gpumem={mem}", f"gpucores={cores} parked={parked} replicas={replicas}"))
bad=[r for r in rows if r[0]=="GPU" and r[4]=="gpumem=None"]
verbose = os.environ.get("AUDIT_VERBOSE","")=="1"
print("== HAMi GPU accounting audit ==")
print(f"  {len([r for r in rows if r[0]=='GPU'])} container(s) across the repo claim an nvidia.com/ device resource")
for r in [x for x in rows if x[0]=="PARSE"]:
    print(f"  !! parse error {r[1]}: {r[2]}")
livebad=[r for r in bad if "parked=False" in r[5]]
latent=[r for r in bad if "parked=True" in r[5]]
if not livebad:
    print("  OK: every APPLIED manifest sets an explicit nvidia.com/gpumem (no whole-card grabs)")
else:
    print(f"  !! {len(livebad)} APPLIED container(s) with NO nvidia.com/gpumem -> each claims the ENTIRE card:")
    for r in livebad: print(f"     {r[1]}\n        {r[2]}  {r[3]}")
if latent:
    files=collections.Counter(os.path.basename(r[1]) for r in latent)
    print(f"  {len(latent)} parked/replicas:0 container(s) also lack it — latent, fires the moment someone revives them:")
    for f,n in files.most_common(None if verbose else 12):
        print(f"     {n:3d}  {f}")
    if not verbose and len(files)>12: print(f"     ... +{len(files)-12} more files (AUDIT_VERBOSE=1 to list)")
if verbose:
    print("  compliant (gpumem set):")
    for r in [x for x in rows if x[0]=="GPU" and x not in bad][:200]:
        print(f"     {r[1]}  {r[2]}  {r[4]}")
PY
fi

echo
echo "== live: node register + per-pod reservations =="
kubectl get node -o jsonpath='{.metadata.annotations.hami\.io/node-nvidia-register}' 2>/dev/null | sed 's/^/  register: /' || echo "  (cluster unreachable)"
echo
kubectl get pods -A -o json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
for p in d['items']:
    sp=p['spec']; st=p.get('status',{})
    if st.get('phase') not in ('Running','Pending'): continue
    hits=[]
    for c in sp['containers']:
        r={**(c.get('resources',{}).get('requests') or {}),**(c.get('resources',{}).get('limits') or {})}
        g={k:v for k,v in r.items() if k.startswith('nvidia.com/')}
        if g: hits.append((c['name'],g))
    if not hits: continue
    ann={k:v for k,v in (p['metadata'].get('annotations') or {}).items() if 'hami' in k.lower() or 'vgpu' in k.lower()}
    print(f\"  {p['metadata']['namespace']}/{p['metadata']['name']}  [{st['phase']}]\")
    for n,g in hits: print(f'     {n}: {g}')
    if ann: print(f'     hami-ann: {ann}')
" | head -60
