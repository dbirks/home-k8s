# Summarise NInfer request/throughput log lines for a speculative-decoding A/B.
# Usage: kubectl logs deploy/ninfer > run.log; python3 spec-decode-stats.py run.log
# Logs vanish when the pod restarts: snapshot them BEFORE switching --spec.
import sys,re,statistics as st
reqs=[];thr={}
for l in open(sys.argv[1]):
    m=re.search(r"\| (\S+) \|.*?prompt ([\d,]+) \| output ([\d,]+) .*?TTFT ([\d.]+) (ms|s) .*?decode ([\d.]+) tok/s \| (\w+) accepted ([\d,]+)/([\d,]+)",l)
    if m:
        n=lambda s:int(s.replace(",",""))
        out,acc=n(m.group(3)),n(m.group(8))
        ttft=float(m.group(4))/(1000 if m.group(5)=="ms" else 1)
        reqs.append(dict(ts=l[:19],out=out,acc=acc,prop=n(m.group(9)),dec=float(m.group(6)),ttft=ttft,spec=m.group(7)))
        continue
    m=re.search(r"throughput \| 5.0s \| decode ([\d.]+) tok/s.*\| batch ([\d.]+)",l)
    if m: thr.setdefault(min(8,round(float(m.group(2)))),[]).append(float(m.group(1)))
if not reqs: sys.exit("no requests")
A=sum(r["acc"] for r in reqs);P=sum(r["prop"] for r in reqs);O=sum(r["out"] for r in reqs)
L=[r["out"]/(r["out"]-r["acc"]) for r in reqs if r["out"]>=50 and r["out"]>r["acc"]]
big=[r for r in reqs if r["out"]>=200]
print(f'{reqs[0]["spec"]}: {len(reqs)} requests {reqs[0]["ts"]} .. {reqs[-1]["ts"]}')
print(f'  acceptance {A/P:.1%}  mean accept length {st.mean(L):.2f}')
print(f'  per-request decode tok/s (output>=200, n={len(big)}): median {st.median([r["dec"] for r in big]):.0f}  p10 {st.quantiles([r["dec"] for r in big],n=10)[0]:.0f}  p90 {st.quantiles([r["dec"] for r in big],n=10)[-1]:.0f}')
print(f'  TTFT median {st.median([r["ttft"] for r in reqs]):.2f}s')
for k in sorted(thr):
    if len(thr[k])>=10: print(f'  engine decode at batch~{k}: median {st.median(thr[k]):.0f} tok/s (n={len(thr[k])} 5-s windows)')
