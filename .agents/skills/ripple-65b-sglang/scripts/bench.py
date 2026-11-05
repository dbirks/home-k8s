"""Concurrency + prefix-reuse bench for an OpenAI-compatible endpoint (stdlib only).

usage: python3 -I bench.py BASE_URL MODEL "1,4,8,16" [PREFIX_TOKENS] [MAX_TOKENS]   (API key from env BENCH_API_KEY)
Each client sends: shared prefix (~PREFIX_TOKENS of code-like text, identical across clients; BENCH_DISTINCT=1
gives each client its own) + a per-client divergent tail, streams, records TTFT and decode tok/s; then every
client re-sends its own conversation plus one more turn, BENCH_TURNS times in all (default 2) = prefix reuse. Prints per-level p50/p95 TTFT, per-stream and aggregate decode tok/s.
Pair with /metrics deltas (sglang:cache_hit_rate, sglang:token_usage, num_running/queue) on the engine.
"""
import json, os, statistics as st, sys, threading, time, urllib.request

BASE, MODEL = sys.argv[1].rstrip("/"), sys.argv[2]
LEVELS = [int(x) for x in sys.argv[3].split(",")]
PREFIX = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
MAXT = int(sys.argv[5]) if len(sys.argv) > 5 else 256
KEY = sys.argv[6] if len(sys.argv) > 6 else os.environ.get("BENCH_API_KEY")
HDR = {"Content-Type": "application/json", **({"Authorization": "Bearer " + KEY} if KEY else {})}

# ~10 tokens per line of fake code; deterministic so the prefix is shared byte-for-byte
SHARED = "\n".join(f"def helper_{i}(x):\n    return (x * {i} + {i * 7}) % 1009" for i in range(PREFIX // 20))


def one(msgs, out):
    body = {"model": MODEL, "messages": msgs, "max_tokens": MAXT, "temperature": 0.7, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.time(); first = None; n = 0; usage = None; text = ""
    try:
        r = urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), HDR), timeout=1800)
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ch = json.loads(line[5:])
            if ch.get("usage"):
                usage = ch["usage"]
            if ch.get("choices"):
                dl = ch["choices"][0].get("delta", {})
                piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
                if piece:
                    first = first or time.time(); n += 1; text += dl.get("content") or ""
        t1 = time.time()
        ct = (usage or {}).get("completion_tokens", n)
        cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        out.append({"ttft": (first or t1) - t0, "dec": ct / max(t1 - (first or t1), 1e-6), "ct": ct, "wall": t1 - t0,
                    "pt": (usage or {}).get("prompt_tokens"), "cached": cached, "text": text})
    except Exception as e:  # noqa: BLE001
        out.append({"err": repr(e)[:200]})


def level(c):
    res1, res2 = [], []
    distinct = os.environ.get("BENCH_DISTINCT") == "1"   # each client its own repo (no cross-client sharing)
    convs = [[{"role": "system", "content": "You are a coding agent. Repository context:\n"
               + (SHARED.replace("helper_", f"c{i}_helper_") if distinct else SHARED)},
              {"role": "user", "content": f"Task {c}-{i}: explain helper_{i * 3} briefly, then write a test for it."}] for i in range(c)]
    turns = int(os.environ.get("BENCH_TURNS", "2"))
    allres = [[] for _ in range(turns)]
    res1 = allres[0]
    for phase, res in enumerate(allres, 1):
        if phase > 1:
            prev = allres[phase - 2]
            for i, cv in enumerate(convs):
                cv += [{"role": "assistant", "content": (prev[i].get("text") or "ok") if i < len(prev) else "ok"},
                       {"role": "user", "content": f"Turn {phase}: now refactor it differently. Code only."}]
        t0 = time.time()
        th = [threading.Thread(target=one, args=(cv, res)) for cv in convs]
        [t.start() for t in th]; [t.join() for t in th]
        wall = time.time() - t0
        ok = [r for r in res if "err" not in r]
        errs = [r["err"] for r in res if "err" in r]
        if ok:
            tt = sorted(r["ttft"] for r in ok)
            p95 = tt[min(len(tt) - 1, int(0.95 * len(tt)))]
            agg = sum(r["ct"] for r in ok) / wall
            hit = [r["cached"] / r["pt"] for r in ok if r.get("cached") is not None and r.get("pt")]
            print(f"c={c:>2} turn{phase} ok={len(ok)}/{c} ttft p50={st.median(tt):.2f}s p95={p95:.2f}s "
                  f"per-stream={st.median(r['dec'] for r in ok):.1f} tok/s agg={agg:.0f} tok/s "
                  f"prompt~{st.median(r['pt'] or 0 for r in ok):.0f} tok cached={('%.0f%%' % (100 * st.mean(hit))) if hit else 'n/a'} "
                  f"wall={wall:.1f}s", flush=True)
        if errs:
            print(f"   errors: {errs[:3]}", flush=True)


for c in LEVELS:
    level(c)
