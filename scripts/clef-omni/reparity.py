"""Recompute every export's BF16 parity against a SAME-PATH BF16 reference (issue #152).

The original reference came from eager per-expert BF16; exports are graded through serve_clef.py's grouped
experts, and eager-vs-grouped drift alone flips near-tie answers. eval_export.py on the BF16 release writes
rows whose "quant" field is grouped-path BF16; this script uses those as the reference and rescores each
export's saved parity_export_rows.jsonl ("quant" field), no model needed.

    python reparity.py --reference /work/bf16/results/export/parity_export_rows.jsonl \
        int4=/work/int4/results/fast/parity_export_rows.jsonl fp8=/work/fp8/results/export/parity_export_rows.jsonl ...
"""
import argparse
import json

import numpy as np


def rows(path):
    return {(r["record"], r["question"]): r["quant"] for r in map(json.loads, open(path))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("exports", nargs="+", help="name=path/to/parity_export_rows.jsonl")
    args = ap.parse_args()
    ref = rows(args.reference)
    report = {}
    for item in args.exports:
        name, path = item.split("=", 1)
        q = rows(path)
        keys = [k for k in q if k in ref]
        tv = [0.5 * sum(abs(ref[k][o] - q[k][o]) for o in ref[k]) for k in keys]
        agree = sum(max(ref[k], key=ref[k].get) == max(q[k], key=q[k].get) for k in keys)
        report[name] = {"questions": len(keys), "top1_agreement": round(agree / len(keys), 4),
                        "mean_tv": round(float(np.mean(tv)), 5), "p95_tv": round(float(np.percentile(tv, 95)), 5)}
        print(f"{name:14s} top1 {report[name]['top1_agreement']:.4f}  mean_tv {report[name]['mean_tv']:.5f}  (n={len(keys)})", flush=True)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
