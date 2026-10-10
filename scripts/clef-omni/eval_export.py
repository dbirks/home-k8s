"""Grade an EXPORTED Clef-Omni checkpoint against the BF16 reference (issue #152).

Loads the export exactly as users will (serve_clef.load), scores the 128 held-out records from
clef_records.build(), and compares with the BF16 probabilities saved by the Stage 2a ModelOpt job
(parity_fakequant_rows.jsonl, "bf16" field). Same records, same seeds, same head.

    python eval_export.py --export /work/int4/export --out /work/int4/results [--simulate-fp4-activations]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clef_records  # noqa: E402
import serve_clef  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default="/work/nvfp4-experts/results/parity_fakequant_rows.jsonl")
    ap.add_argument("--simulate-fp4-activations", action=argparse.BooleanOptionalAction, default=None,
                    help="NVFP4 W4A4: reproduce FP4 activation rounding (default off, see serve_clef.load)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fast-moe", action=argparse.BooleanOptionalAction, default=True,
                    help="AutoRound INT4: grouped int4 MoE fast path (see serve_clef.fast_int4_moe)")
    ap.add_argument("--fakequant-rows", default=None,
                    help="a quant job's parity_fakequant_rows.jsonl: also report how closely the export reproduces it")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ref = {}
    for line in open(args.reference):
        r = json.loads(line)
        ref[(r["record"], r["question"])] = r["bf16"]
    _, evals = clef_records.build()
    evals = [r for r in evals if any((r["id"], q) in ref for q in r["questions"])][: args.limit]

    t0 = time.time()
    model, processor, jsm, kind = serve_clef.load(Path(args.export), "cuda",
                                                  simulate_fp4_activations=args.simulate_fp4_activations,
                                                  fast_moe=args.fast_moe)
    load_s = time.time() - t0
    clef_records.log(f"loaded {args.export} ({kind}) in {load_s:.0f}s; "
                     f"gpu {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    rows, agree, total, tvs, flips, lat = [], 0, 0, [], [], []
    torch.cuda.reset_peak_memory_stats()
    for rec in evals:
        enc = jsm.encode_record(processor.tokenizer, rec, processor=processor)
        batch = jsm.collate_records([enc], processor.tokenizer.pad_token_id, torch.device("cuda"))
        torch.cuda.synchronize(); ts = time.time()
        with torch.inference_mode():
            logits = model(batch)[0]
        torch.cuda.synchronize(); lat.append(time.time() - ts)
        for q, lg in zip(enc.questions, logits):
            key = (rec["id"], q.question_id)
            if key not in ref:
                continue
            bd = ref[key]
            qd = dict(zip(q.option_ids, lg.float().softmax(-1).tolist()))
            tv = 0.5 * sum(abs(bd[o] - qd[o]) for o in bd)
            b_top, q_top = max(bd, key=bd.get), max(qd, key=qd.get)
            srt = sorted(bd.values(), reverse=True)
            total += 1; agree += b_top == q_top; tvs.append(tv)
            if b_top != q_top:
                flips.append({"record": rec["id"], "question": q.question_id, "bf16": b_top, "quant": q_top,
                              "bf16_margin": round(srt[0] - (srt[1] if len(srt) > 1 else 0), 4)})
            rows.append({"record": rec["id"], "question": q.question_id, "tv": round(tv, 6), "bf16": bd, "quant": qd})
    parity = {"export": args.export, "kind": kind, "fast_moe": args.fast_moe, "simulate_fp4_activations": args.simulate_fp4_activations,
              "questions": total, "top1_agreement": round(agree / total, 4),
              "mean_tv": round(float(np.mean(tvs)), 5), "p95_tv": round(float(np.percentile(tvs, 95)), 5),
              "max_tv": round(float(np.max(tvs)), 5), "flips": flips,
              "gate_top1_ge_0.98": agree / total >= .98, "gate_mean_tv_le_0.03": float(np.mean(tvs)) <= .03,
              "load_s": round(load_s, 1), "forward_ms_median": round(1000 * float(np.median(lat)), 1),
              "forward_ms_p95": round(1000 * float(np.percentile(lat, 95)), 1),
              "weights_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
              "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    if args.fakequant_rows:
        fq = {(r["record"], r["question"]): r["nvfp4"] for r in map(json.loads, open(args.fakequant_rows))}
        diffs, same = [], 0
        for r in rows:
            f = fq.get((r["record"], r["question"]))
            if f is None:
                continue
            diffs.append(max(abs(f[o] - r["quant"][o]) for o in f))
            same += max(f, key=f.get) == max(r["quant"], key=r["quant"].get)
        parity["vs_fakequant"] = {"questions": len(diffs), "top1_same": round(same / max(len(diffs), 1), 4),
                                  "max_abs_dp": round(max(diffs), 5) if diffs else None,
                                  "mean_abs_dp": round(float(np.mean(diffs)), 6) if diffs else None}
    json.dump(parity, open(out / "parity_export.json", "w"), indent=1)
    with open(out / "parity_export_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    clef_records.log("EXPORT PARITY:", json.dumps({k: v for k, v in parity.items() if k != "flips"}), f"flips={len(flips)}")


if __name__ == "__main__":
    main()
