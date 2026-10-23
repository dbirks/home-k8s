import json, sys, time, urllib.request, concurrent.futures as cf

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:18080"
MODE = sys.argv[2] if len(sys.argv) > 2 else "all"


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t


def tools():
    body = {"model": "flint", "max_tokens": 4096, "messages": [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                       "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]}
    d, dt = post("/v1/chat/completions", body)
    m = d["choices"][0]["message"]
    print("TOOLS finish=%s tool_calls=%s content=%r (%.1fs)" % (d["choices"][0]["finish_reason"], json.dumps(m.get("tool_calls")), (m.get("content") or "")[:80], dt))
    if not m.get("tool_calls"):
        return
    tc = m["tool_calls"][0]
    body["messages"] += [m, {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"temp_c": 14, "sky": "light rain"})}]
    d, dt = post("/v1/chat/completions", body)
    print("TOOLS turn2:", (d["choices"][0]["message"].get("content") or "")[:200].replace("\n", " "))


def anthropic():
    d, dt = post("/v1/messages", {"model": "flint", "max_tokens": 2048, "messages": [{"role": "user", "content": "Name three prime numbers above 100. One line."}]})
    print("ANTHROPIC types=%s text=%r (%.1fs)" % ([c["type"] for c in d.get("content", [])], [c.get("text") for c in d.get("content", []) if c["type"] == "text"], dt))


def one(i, max_tokens=600):
    d, dt = post("/v1/chat/completions", {"model": "flint", "max_tokens": max_tokens,
                 "messages": [{"role": "user", "content": f"Write a short Python function #{i} that returns the {i+2}th Fibonacci number, with a docstring. Then explain it in two sentences."}]})
    t = d.get("timings") or {}
    return i, d["choices"][0]["finish_reason"], d["usage"]["completion_tokens"], t.get("predicted_per_second") or d["usage"]["completion_tokens"] / dt, dt, "def " in (d["choices"][0]["message"].get("content") or "")


def parallel(n):
    t = time.time()
    with cf.ThreadPoolExecutor(n) as ex:
        res = list(ex.map(one, range(n)))
    wall = time.time() - t
    toks = sum(r[2] for r in res)
    print("PARALLEL %d: wall %.1fs, %d tok total = %.0f tok/s aggregate; per-req tok/s min/med/max %.0f/%.0f/%.0f; has code %d/%d; finish %s" % (
        n, wall, toks, toks / wall, min(r[3] for r in res), sorted(r[3] for r in res)[n // 2], max(r[3] for r in res),
        sum(r[5] for r in res), n, sorted(set(r[1] for r in res))))


def longctx(n_words):
    filler = " ".join(f"line{i}: the quick brown fox jumps over the lazy dog {i % 97}." for i in range(n_words // 12))
    secret = "The vault code is 4471-ZETA."
    mid = len(filler) // 2
    text = filler[:mid] + " " + secret + " " + filler[mid:]
    d, dt = post("/v1/chat/completions", {"model": "flint", "max_tokens": 2048,
                 "messages": [{"role": "user", "content": text + "\n\nWhat is the vault code mentioned above? Reply with just the code."}]})
    t = d.get("timings")
    if t:
        print("LONGCTX prompt %d tok: prefill %.0f tok/s (%.1fs), decode %.0f tok/s, answer=%r" % (
            t["prompt_n"], t["prompt_per_second"], t["prompt_ms"] / 1000, t["predicted_per_second"], (d["choices"][0]["message"].get("content") or "").strip()[:60]))
    else:
        u = d["usage"]
        print("LONGCTX prompt %d tok, %d out: total %.1fs (prefill+decode), answer=%r" % (
            u["prompt_tokens"], u["completion_tokens"], dt, (d["choices"][0]["message"].get("content") or "").strip()[:60]))

if MODE in ("all", "tools"): tools()
if MODE in ("all", "anthropic"): anthropic()
if MODE in ("all", "par"):
    for n in (1, 4, 8, 16):
        parallel(n)
if MODE in ("all", "long"):
    for w in (30000, 120000):
        longctx(w)
