#!/usr/bin/env python3
"""One repeatable evaluation harness for the Inkling 2 Mini endpoint (Basalt).

Run this before and after every Basalt change (image bump, --spec/drafting, conversation-cache flags,
KV dtype, --parallel/--max-context) so numbers compare like for like, and keep the JSON as the record.

usage:
  # port-forward first: kubectl port-forward svc/inkling-2-mini 18100:8000
  python3 -I inkling_eval.py --base http://127.0.0.1:18100 --suite quick \
      --out ~/.cache/pi-orch/home-k8s/scratch/inkling-eval/baseline-0.5.5.json --label 0.5.5
  python3 -I inkling_eval.py --suite full --needle-tokens 120000 --out full-before.json
  python3 -I inkling_eval.py --compare baseline-0.5.5.json candidate-0.5.6.json

Stdlib only, Python 3.11+. At most TWO concurrent requests, ever (Basalt serves 2 slots and real users
share this endpoint): a global semaphore guards every HTTP call and the thread pool is 2 wide.
Requests are streamed where TTFT matters (speed/prefix/needle) and plain elsewhere.
Every request carries model "inkling-2-mini" and an x-session-id header (distinct per logical
conversation - that is how Basalt pins a conversation to one slot). 429/503 (and transport errors)
retry 3x with 10 s backoff. Per-request timeout 900 s.

Tests (each prints one summary line; --only speed,tools selects a subset):
  speed     single/2-stream decode tok/s (TTFT excluded) + TTFT, code & prose, temp 1.0 and temp 0
  tools     edit_file tool-call validity (temp 1.0 + temp 0) and a 2-tool "both calls" check
  prefix    23K-token prompt, same-session append (prefix hit) and a different-session reuse
  park      3 x ~20K conversations vs 2 slots: does an evicted conversation come back from the conversation cache?
  needle    ~--needle-tokens prompt, needle at 50 %: prefill/decode tok/s (full suite only)
  quality   20 seeded 3x3-digit multiplications at reasoning_effort low
  effort    one math question at reasoning_effort none/low/medium/high -> reasoning size + correctness
  vision    generated 64x64 solid-red PNG as a data URL
  anthropic /v1/messages text + tool_use
  metrics   /health and /props snapshots (truncated to 2 KB)
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import re
import statistics
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor

MODEL = "inkling-2-mini"
TIMEOUT = 900.0
RETRIES = 3
RETRY_BACKOFF = 10.0
RETRY_CODES = {429, 503}
MAX_CONCURRENCY = 2
SUITE_QUICK = ["speed", "tools", "prefix", "park", "quality", "vision", "anthropic", "metrics"]
SUITE_FULL = ["speed", "tools", "prefix", "needle", "quality", "effort", "vision", "anthropic", "metrics"]

_slots = threading.Semaphore(MAX_CONCURRENCY)
NONCE = uuid.uuid4().hex[:6]
LABEL = "run"

# ---------------------------------------------------------------- HTTP plumbing


def sid(name: str) -> str:
    """Stable-but-unique session id per logical conversation (Basalt slot affinity)."""
    return f"inkling-eval-{LABEL}-{NONCE}-{name}"[:120]


def _open(base, path, body, session, timeout, extra_headers=None):
    headers = {"Content-Type": "application/json", "x-session-id": session}
    headers.update(extra_headers or {})
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def _retry(fn, log):
    """3 attempts, 10 s backoff, on 429/503 and transport errors."""
    last = None
    for attempt in range(RETRIES):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_CODES:
                raise
            last = e
            log.append(f"http {e.code} on attempt {attempt + 1}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
            log.append(f"transport error on attempt {attempt + 1}: {e.__class__.__name__}")
        if attempt < RETRIES - 1:
            time.sleep(RETRY_BACKOFF)
    raise last


def _err_dict(e):
    out = {"error": f"{e.__class__.__name__}: {str(e)[:200]}"}
    if isinstance(e, urllib.error.HTTPError):
        out["http_status"] = e.code
        try:
            out["body"] = e.read().decode("utf-8", "replace")[:400]
        except Exception:  # noqa: BLE001
            pass
    return out


def read_sse(resp):
    """Parse OpenAI-style SSE: 'data: {...}' lines, terminated by 'data: [DONE]'."""
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        if not payload:
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def chat(base, body, session, timeout=TIMEOUT, path="/v1/chat/completions", extra_headers=None):
    """Non-streaming POST -> (json dict, wall seconds)."""

    def go():
        with _slots:
            t0 = time.time()
            resp = _open(base, path, body, session, timeout, extra_headers)
            with resp:
                return json.load(resp), time.time() - t0

    return _retry(go, [])


def chat_stream(base, body, session, timeout=TIMEOUT):
    """Streaming POST -> timing dict. Decode rate excludes TTFT: first..last token chunk."""

    def go():
        payload = dict(body, stream=True, stream_options={"include_usage": True})
        with _slots:
            t0 = time.time()
            resp = _open(base, "/v1/chat/completions", payload, session, timeout)
            first = last = None
            pieces = 0
            content, reasoning = [], []
            usage = timings = None
            finish = None
            with resp:
                for chunk in read_sse(resp):
                    now = time.time()
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    if chunk.get("timings"):
                        timings = chunk["timings"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = delta.get("content") or ""
                        reacon = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                        if piece or reacon:
                            if first is None:
                                first = now
                            last = now
                            pieces += 1
                            content.append(piece)
                            reasoning.append(reacon)
            end = time.time()
        pt = (usage or {}).get("prompt_tokens")
        ct = (usage or {}).get("completion_tokens") or pieces
        window = (last - first) if (first and last and last > first) else 0.0
        out = {
            "ttft": round((first - t0) if first else (end - t0), 3),
            "wall": round(end - t0, 3),
            "completion_tokens": ct,
            "prompt_tokens": pt,
            "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "stream_pieces": pieces,
            "finish_reason": finish,
            "text": "".join(content),
            "reasoning": "".join(reasoning),
            "reasoning_chars": len("".join(reasoning)),
        }
        out["decode_tps"] = round(ct / window, 1) if window else None
        if timings:
            out["engine_prompt_tps"] = round(timings.get("prompt_per_second") or 0, 1) or None
            out["engine_decode_tps"] = round(timings.get("predicted_per_second") or 0, 1) or None
            if timings.get("draft_n"):
                out["draft_accept_ratio"] = round(timings.get("draft_n_accepted", 0) / timings["draft_n"], 3)
        return out

    return _retry(go, [])


# ---------------------------------------------------------------- prompt blobs

def filler_line(i: int) -> str:
    """Deterministic code-ish filler, numbered lines; ~9-10 tokens per line (calibrated at runtime)."""
    return (
        f"{i:05d} def helper_{i}(ctx):\n"
        f"           return (ctx * {i} + {i * 7}) % 1009  # cached path {i}\n"
    )


_calib = {"tokens_per_line": None}


def calibrate_tokens_per_line(base):
    """One cheap probe so a '23K token' prompt really is ~23K tokens."""
    if _calib["tokens_per_line"]:
        return _calib["tokens_per_line"]
    probe = "\n".join(filler_line(i) for i in range(1, 301))
    d, _ = chat(base, {"model": MODEL, "max_tokens": 1, "temperature": 0, "reasoning_effort": "none",
                       "messages": [{"role": "user", "content": probe}]}, sid("calib"))
    pt = (d.get("usage") or {}).get("prompt_tokens") or 3000
    _calib["tokens_per_line"] = pt / 300.0
    return _calib["tokens_per_line"]


# The planted fact is a plain build number, not a "secret": phrased as a vault code / credential the model
# sometimes refuses to reveal, which reads as a fact miss (facts_ok=False / needle FAIL) that is not a cache miss.
def long_prompt(base, target_tokens, build_no, question):
    per = calibrate_tokens_per_line(base)
    lines = max(50, int(target_tokens / per))
    blob = "\n".join(filler_line(i) for i in range(1, lines + 1)).split("\n")
    mid = len(blob) // 2
    build_no = f"{build_no:06d}"
    blob.insert(mid, f"{mid:05d} # IMPORTANT: The build number of this file is {build_no}. Record it exactly.")
    return (f"You are a coding agent. Below is a numbered repository dump (two source lines per entry).\n"
            f"It contains exactly one IMPORTANT line with a build number.\n\n" + "\n".join(blob) + f"\n\n{question}"), lines


# ---------------------------------------------------------------- 1. speed

SPEED_PROMPTS = {
    "code": ("Write a thread-safe LRU cache module in Python: class LRUCache with get/put, capacity, "
             "threading.Lock, and a unittest.TestCase covering eviction and concurrent access. "
             "Reply with only one python code block, no prose."),
    "prose": ("Write a 600-word short story about a lighthouse keeper who discovers that the sea keeps "
              "returning lost objects. Begin it in medias res. Prose only."),
}


def run_speed(base, args):
    out = {"params": {"reasoning_effort": "none", "temperature": 1.0, "max_tokens": 800}}

    def meas(sessions):
        """Run 1 session single-stream (sequential); 2 sessions concurrently."""
        if len(sessions) == 1:
            return [chat_stream(base, body, s) for s, body in sessions]
        with ThreadPoolExecutor(len(sessions)) as ex:
            return list(ex.map(lambda sb: chat_stream(base, sb[1], sb[0]), sessions))

    for name, prompt in SPEED_PROMPTS.items():
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800, "temperature": 1.0, "reasoning_effort": "none"}
        per = {}
        single = meas([(sid(f"speed-{name}-{i}"), body) for i in range(3)])
        ok = [r for r in single if r.get("decode_tps")]
        per["one_stream"] = {
            "runs": len(single),
            "median_decode_tps": round(statistics.median(r["decode_tps"] for r in ok), 1) if ok else None,
            "all_decode_tps": [r["decode_tps"] for r in single],
            "median_ttft": round(statistics.median(r["ttft"] for r in single), 3) if single else None,
            "median_completion_tokens": round(statistics.median(r["completion_tokens"] for r in single)) if single else None,
            "engine_decode_tps": [r.get("engine_decode_tps") for r in single],
            "errors": [r.get("error") for r in single if r.get("error")],
        }
        pairs = meas([(sid(f"speed-{name}-p1"), body), (sid(f"speed-{name}-p2"), body)])
        okp = [r for r in pairs if r.get("decode_tps")]
        wall = max((r["wall"] for r in pairs), default=0.0)
        per["two_streams"] = {
            "per_stream_median_decode_tps": round(statistics.median(r["decode_tps"] for r in okp), 1) if okp else None,
            "per_stream_decode_tps": [r["decode_tps"] for r in pairs],
            "aggregate_decode_tps": round(sum(r["completion_tokens"] for r in okp) / wall, 1) if okp and wall else None,
            "median_ttft": round(statistics.median(r["ttft"] for r in pairs), 3) if pairs else None,
            "errors": [r.get("error") for r in pairs if r.get("error")],
        }
        body0 = dict(body, temperature=0.0)
        t0run = chat_stream(base, body0, sid(f"speed-{name}-t0"))
        per["temp0_one_stream"] = {"decode_tps": t0run.get("decode_tps"), "ttft": t0run.get("ttft"),
                                   "completion_tokens": t0run.get("completion_tokens"),
                                   "error": t0run.get("error")}
        out[name] = per
    return out


# ---------------------------------------------------------------- 2. tools

EDIT_TOOL = {"type": "function", "function": {
    "name": "edit_file", "description": "Replace exact text in a file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                 "old_string": {"type": "string"}, "new_string": {"type": "string"}},
                 "required": ["path", "old_string", "new_string"]}}}


def _edit_case(base, i, temp, session):
    args = {"path": f"src/module_{i}.py", "old_string": f'value_{i} = "{i * 7}"',
            "new_string": f'value_{i} = "{i * 7}"  # edited {i}'}
    body = {"model": MODEL, "temperature": temp, "max_tokens": 8192, "tools": [EDIT_TOOL],
            "messages": [{"role": "user", "content": "Call the edit_file tool exactly once with these "
                         "arguments and do not answer in text. Copy each string character for character.\n"
                         + json.dumps(args)}]}
    try:
        d, _ = chat(base, body, session)
    except Exception as e:  # noqa: BLE001
        return i, "error", str(e)[:100]
    choice = (d.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    tc = msg.get("tool_calls") or []
    if tc:
        try:
            ok = json.loads(tc[0]["function"]["arguments"]) == args
        except Exception:  # noqa: BLE001
            ok = False
        return i, "valid" if ok else "wrong-args", json.dumps(tc[0]["function"])[:160]
    if choice.get("finish_reason") == "length":
        return i, "truncated", (msg.get("reasoning_content") or "")[:80]
    return i, "text", (msg.get("content") or "")[:120]


def _edit_batch(base, n, temp):
    counts = {"valid": 0, "wrong-args": 0, "text": 0, "truncated": 0, "error": 0}
    fails = []
    t0 = time.time()
    with ThreadPoolExecutor(MAX_CONCURRENCY) as ex:
        # two rotating session ids, as the original toolprobe did
        res = list(ex.map(lambda i: _edit_case(base, i, temp, sid(f"tool-{i % 2}")), range(1, n + 1)))
    for i, kind, info in res:
        counts[kind] += 1
        if kind != "valid" and len(fails) < 3:
            fails.append({"case": i, "kind": kind, "info": info})
    return {"n": n, "temperature": temp, "counts": counts,
            "valid_pct": round(100.0 * counts["valid"] / n, 1), "seconds": round(time.time() - t0, 1),
            "sample_failures": fails}


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
TIME_TOOL = {"type": "function", "function": {"name": "get_time", "description": "Current time in a timezone",
             "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}}


def run_tools(base, args):
    out = {"edit_file_temp1": _edit_batch(base, args.tool_n_temp1, 1.0),
           "edit_file_temp0": _edit_batch(base, args.tool_n_temp0, 0.0)}
    body = {"model": MODEL, "temperature": 1.0, "max_tokens": 4096, "tools": [WEATHER, TIME_TOOL],
            "messages": [{"role": "user", "content": "What is the weather in Paris and what time is it there? "
                         "Call both get_weather and get_time as tool calls."}]}

    def one(i):
        try:
            d, _ = chat(base, body, sid(f"multi-{i % 2}"))
        except Exception as e:  # noqa: BLE001
            return "error", str(e)[:100]
        msg = (d.get("choices") or [{}])[0].get("message") or {}
        names = sorted({(tc.get("function") or {}).get("name") for tc in msg.get("tool_calls") or []})
        return ("both" if {"get_weather", "get_time"} <= set(names) else
                "one" if names else "text"), names

    with ThreadPoolExecutor(MAX_CONCURRENCY) as ex:
        res = list(ex.map(one, range(args.multi_tool_n)))
    kinds = {}
    for kind, _info in res:
        kinds[kind] = kinds.get(kind, 0) + 1
    out["multi_tool"] = {"n": args.multi_tool_n, "counts": kinds,
                         "both_pct": round(100.0 * kinds.get("both", 0) / args.multi_tool_n, 1),
                         "samples": [r[1] for r in res if r[0] != "both"][:3]}
    return out


# ---------------------------------------------------------------- 3. prefix

def run_prefix(base, args):
    target = args.prefix_tokens
    build_no = random.Random(NONCE).randint(0, 999999)
    q1 = "Question 1: What is the build number of this file? Reply with just the six digits."
    p1, lines = long_prompt(base, target, build_no, q1)
    msgs = [{"role": "user", "content": p1}]
    conv = sid("prefix-main")
    req = {"model": MODEL, "max_tokens": 1500, "temperature": 0, "reasoning_effort": "none"}
    r1 = chat_stream(base, dict(req, messages=msgs), conv)
    ans1 = r1.get("text") or ""
    build_str = f"{build_no:06d}"
    in_ans1 = build_str in ans1 or build_str in ans1.replace(",", "")
    body2 = {"model": MODEL, "max_tokens": 1500, "temperature": 0, "reasoning_effort": "none"}
    msgs2 = msgs + [{"role": "assistant", "content": ans1 or "(no answer)"},
                    {"role": "user", "content": "Question 2: How many helper_ functions are defined in the dump? "
                     "Reply with just the number."}]
    r2 = chat_stream(base, dict(body2, messages=msgs2), conv)
    ans2 = r2.get("text") or ""
    msgs3 = msgs2 + [{"role": "assistant", "content": ans2 or "(no answer)"},
                     {"role": "user", "content": "Question 3: Restate the build number and the helper count on one line."}]
    r3 = chat_stream(base, dict(body2, messages=msgs3), conv)
    r3alt = chat_stream(base, dict(body2, messages=msgs3), sid("prefix-alt"))
    out = {
        "target_tokens": target, "filler_entries": lines,
        "turn1": {k: r1.get(k) for k in ("ttft", "prompt_tokens", "cached_tokens", "decode_tps", "engine_prompt_tps", "wall")},
        "turn2_same_session": {k: r2.get(k) for k in ("ttft", "prompt_tokens", "cached_tokens", "decode_tps", "engine_prompt_tps", "wall")},
        "turn3_same_session": {k: r3.get(k) for k in ("ttft", "prompt_tokens", "cached_tokens", "decode_tps", "engine_prompt_tps", "wall")},
        "turn3_different_session": {k: r3alt.get(k) for k in ("ttft", "prompt_tokens", "cached_tokens", "decode_tps", "engine_prompt_tps", "wall")},
        "turn2_ttft_ratio_vs_turn1": (round(r1["ttft"] / r2["ttft"], 2) if r1.get("ttft") and r2.get("ttft") else None),
        "build_number": build_str,
        "answer_contains_fact": in_ans1,
        "answer_contains_fact_in_reasoning": (not in_ans1) and (
            build_str in (r1.get("reasoning") or "") or build_str in (r1.get("reasoning") or "").replace(",", "")),
        "answer1_excerpt": ans1[:60].replace("\n", " ") or (r1.get("reasoning") or "")[:60].replace("\n", " "),
        "answer3_excerpt": (r3.get("text") or "")[:80].replace("\n", " "),
        "cross_session_reuse": bool((r3alt.get("cached_tokens") or 0) > 0),
        "errors": [r.get("error") for r in (r1, r2, r3, r3alt) if r.get("error")],
    }
    return out


# ---------------------------------------------------------------- 3b. park


def park_line(i: int, tag: str) -> str:
    """Code-ish filler unique per conversation tag: A/B/C must NOT share a prefix or the prompt cache serves
    the later ones and nothing is ever evicted."""
    return (f"{i:05d} def helper_{tag}_{i}(ctx):\n"
            f"           return (ctx * {i} + {i * 7}) % 1009  # {tag} cached path {i}\n")


def park_prompt(base, target_tokens, tag, build_no, offset):
    """~target_tokens of filler tagged `tag` with its own build number buried at the midpoint."""
    per = calibrate_tokens_per_line(base)
    lines = max(50, int(target_tokens / per))
    build_str = f"{build_no:06d}"
    blob = "\n".join(park_line(i, tag) for i in range(offset, offset + lines)).split("\n")
    mid = len(blob) // 2
    blob.insert(mid, f"{offset + mid:05d} # IMPORTANT for conversation {tag}: "
                     f"The build number of this file is {build_str}. Record it exactly.")
    text = (f"You are a coding agent. Below is a numbered repository dump, tag {tag} "
            f"(two source lines per entry), seen by no other conversation. It contains exactly one IMPORTANT "
            f"line with a build number.\n\n" + "\n".join(blob))
    return text, lines, build_str


def _find_key(obj, key):
    """First value stored under `key` anywhere in a JSON tree (breadth-first), else None."""
    queue = [obj]
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, dict):
            if key in cur:
                return cur[key]
            queue.extend(cur.values())
        elif isinstance(cur, list):
            queue.extend(cur)
    return None


def _cache_stats(base):
    """The engine's conversation-cache figures from /metrics, verbatim, truncated to 1 KB ('' if absent)."""
    try:
        with _slots:
            resp = _open(base, "/metrics", None, sid("park-metrics"), 60)
            with resp:
                d = json.load(resp)
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e.__class__.__name__}"
    found = _find_key(d, "conversation")
    return "" if found is None else json.dumps(found)[:1024]


def _has_fact(result, build_str):
    """Did this reply (content or reasoning) carry the conversation's own build number back?
    A restore that grabbed the wrong conversation's K/V answers with someone else's fact."""
    blob = ((result.get("text") or "") + " " + (result.get("reasoning") or "")).replace(",", "")
    return build_str in blob


def run_park(base, args):
    """Basalt serves --parallel 2 slots. Send three distinct ~20K-token conversations one after another, so the
    third must evict a slot's K/V, then ask two of them a follow-up. With the conversation cache on (V3) a return
    is restored from RAM/disk in a few hundred ms; with it off the whole prompt is re-read (~cold TTFT)."""
    target = args.park_tokens
    req = {"model": MODEL, "max_tokens": 1200, "temperature": 0, "reasoning_effort": "none"}
    rnd = random.Random(f"{NONCE}-park")
    convs = {}
    out = {"target_tokens": target, "slots": 2, "turn1": {}, "cold_ttft": [], "return_ttft": {},
           "return_cached": {}, "facts_ok": True, "errors": []}
    for n, tag in enumerate(("A", "B", "C")):
        prompt, lines, build_str = park_prompt(base, target, tag, rnd.randint(0, 999999), 1 + n * 100000)
        session = sid(f"park-{tag}")
        msgs = [{"role": "user", "content": prompt + "\n\nQuestion 1: What is the build number of this file? "
                                                     "Reply with just the six digits."}]
        r = chat_stream(base, dict(req, messages=msgs), session)
        ans = r.get("text") or ""
        convs[tag] = {"msgs": msgs, "ans": ans, "build": build_str}
        out["turn1"][tag] = {k: r.get(k) for k in ("ttft", "prompt_tokens", "cached_tokens", "wall")}
        out["cold_ttft"].append(r.get("ttft"))
        out[f"turn1_{tag}_answer"] = ans[:70].replace("\n", " ")
        out["filler_lines"] = out.get("filler_lines", 0) + lines
        out["errors"].extend([r["error"]] if r.get("error") else [])
        if not _has_fact(r, build_str):
            out["facts_ok"] = False
            out.setdefault("fact_misses", []).append(f"turn1-{tag}")
        time.sleep(1.0)   # let the slot settle before the next conversation arrives
    for tag in ("A", "B"):
        c = convs[tag]
        msgs2 = c["msgs"] + [{"role": "assistant", "content": c["ans"] or "(no answer)"},
                             {"role": "user", "content": f"Question 2: What is the build number of this file, and how "
                              "many helper_ functions does it define? Reply with the six digits and the count on one line."}]
        r = chat_stream(base, dict(req, messages=msgs2), sid(f"park-{tag}"))
        ans2 = r.get("text") or ""
        out["return_ttft"][tag] = r.get("ttft")
        out["return_cached"][tag] = r.get("cached_tokens")
        out[f"return_{tag}"] = {k: r.get(k) for k in ("ttft", "prompt_tokens", "cached_tokens", "wall")}
        out[f"return_{tag}_answer"] = ans2[:70].replace("\n", " ")
        if not _has_fact(r, c["build"]):
            out["facts_ok"] = False
            out.setdefault("fact_misses", []).append(f"return-{tag}")
        out["errors"].extend([r["error"]] if r.get("error") else [])
    cold = max((t for t in out["cold_ttft"] if t), default=None)
    out["cache_stats"] = _cache_stats(base)
    out["return_speedup_vs_cold"] = {
        tag: (round(cold / r, 2) if cold and r else None) for tag, r in out["return_ttft"].items()}
    out["restored"] = any((r or 0) < 0.6 for r in out["return_ttft"].values())
    return out


# ---------------------------------------------------------------- 4. needle

def run_needle(base, args):
    target = args.needle_tokens
    build_no = random.Random(NONCE + "-needle").randint(0, 999999)
    build_str = f"{build_no:06d}"
    q = ("You were given a numbered repository dump with exactly one IMPORTANT line. "
         "What is the build number of that file? Reply with just the six digits.")
    p, lines = long_prompt(base, target, build_no, q)
    out = {"target_tokens": target, "filler_entries": lines, "attempts": []}
    # The model-card sampler (temperature 1.0) sometimes spends the whole budget on its thought and returns
    # empty content (known gotcha in SKILL.md) -> rerun once with more room before calling it a miss.
    for attempt, extra in enumerate(({"max_tokens": 600, "reasoning_effort": "none"},
                                    {"max_tokens": 1500, "reasoning_effort": "low"})):
        r = chat_stream(base, dict({"model": MODEL, "temperature": 1.0,
                                    "messages": [{"role": "user", "content": p}]}, **extra),
                        sid(f"needle-{attempt + 1}"), timeout=TIMEOUT)
        pt, ttft = (r.get("prompt_tokens") or 0), (r.get("ttft") or 0)
        found = build_str in (r.get("text") or "") or build_str in (r.get("text") or "").replace(",", "")
        att = {"max_tokens": extra["max_tokens"], "reasoning_effort": extra["reasoning_effort"],
               "prompt_tokens": pt, "ttft_seconds": round(ttft, 2),
               "prefill_tps": round(pt / ttft, 1) if ttft else None,
               "decode_tps": r.get("decode_tps"), "completion_tokens": r.get("completion_tokens"),
               "wall": r.get("wall"), "engine_prompt_tps": r.get("engine_prompt_tps"),
               "answer_excerpt": (r.get("text") or "")[:60].replace("\n", " "),
               "pass": found, "error": r.get("error")}
        out["attempts"].append(att)
        for k in ("prompt_tokens", "ttft_seconds", "prefill_tps", "decode_tps", "completion_tokens",
                  "wall", "engine_prompt_tps"):
            out.setdefault(k, att[k])
        out["pass"] = out.get("pass", False) or found
        if found or att["error"]:
            break
    out["build_number"] = build_str
    out["pass"] = bool(out.get("pass"))
    return out


# ---------------------------------------------------------------- 5. quality

def run_quality(base, args):
    rnd = random.Random(args.seed)
    cases = [(a, b) for _ in range(args.quality_n)
             for a, b in [(rnd.randint(100, 999), rnd.randint(100, 999))]]

    def one(i):
        a, b = cases[i]
        want = a * b
        try:
            d, _ = chat(base, {"model": MODEL, "max_tokens": 4000, "temperature": 0, "reasoning_effort": "low",
                               "messages": [{"role": "user", "content": f"What is {a} * {b}? "
                                             "Think it through, then reply with the number."}]},
                        sid(f"quality-{i % 2}"))
        except Exception as e:  # noqa: BLE001
            return {"case": f"{a}*{b}", "want": want, "ok": False, "why": str(e)[:80]}
        msg = (d.get("choices") or [{}])[0].get("message") or {}
        txt = msg.get("content") or ""
        nums = _numbers(txt)
        return {"case": f"{a}*{b}", "want": want, "ok": str(want) in nums,
                "why": None if str(want) in nums else (txt[-40:] or "empty").replace("\n", " ")}

    with ThreadPoolExecutor(MAX_CONCURRENCY) as ex:
        res = list(ex.map(one, range(len(cases))))
    out = {"n": len(res), "correct": sum(r["ok"] for r in res),
           "seed": args.seed, "failures": [r for r in res if not r["ok"]][:5]}
    return out


# ---------------------------------------------------------------- 6. effort

EFFORT_QUESTION = ("A farmer has 17 sheep. All but 9 run away. Then the farmer buys 15 more sheep and "
                   "loses a third of them to wolves. How many sheep are left? Reply with just the number.")
EFFORT_ANSWER = "16"


def run_effort(base, args):
    out = {}
    for eff in ("none", "low", "medium", "high"):
        body = {"model": MODEL, "max_tokens": args.effort_max_tokens, "temperature": 0,
                "reasoning_effort": eff, "messages": [{"role": "user", "content": EFFORT_QUESTION}]}
        try:
            d, wall = chat(base, body, sid(f"effort-{eff}"))
        except Exception as e:  # noqa: BLE001
            out[eff] = _err_dict(e)
            continue
        msg = (d.get("choices") or [{}])[0].get("message") or {}
        usage = d.get("usage") or {}
        rtoks = ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
        rchars = len(msg.get("reasoning_content") or msg.get("reasoning") or "")
        nums = _numbers(msg.get("content") or "")
        out[eff] = {"reasoning_tokens": rtoks, "reasoning_chars": rchars,
                    "estimated_reasoning_tokens": (None if rtoks is not None else round(rchars / 4.0)),
                    "completion_tokens": usage.get("completion_tokens"),
                    "answer": (msg.get("content") or "").strip()[:30],
                    "correct": EFFORT_ANSWER in nums, "wall": round(wall, 2),
                    "finish_reason": (d.get("choices") or [{}])[0].get("finish_reason")}
    return out


# ---------------------------------------------------------------- 7. vision

def _numbers(text):
    """Digit runs, thousands separators folded ('493,696' -> '493696'); models format big numbers both ways."""
    return [n.replace(",", "") for n in re.findall(r"\d[\d,]*\d|\d", text)]


def red_png(width=64, height=64):
    """Solid red PNG built with zlib+struct (no PIL)."""
    row = b"\x00" + b"\xff\x00\x00" * width
    raw = row * height

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def run_vision(base, args):
    url = "data:image/png;base64," + base64.b64encode(red_png()).decode()
    body = {"model": MODEL, "max_tokens": 1500, "temperature": 0, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "What colour is this image? Reply with one word."}]}]}
    try:
        d, wall = chat(base, body, sid("vision-1"))
    except Exception as e:  # noqa: BLE001
        return _err_dict(e)
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    txt = (msg.get("content") or "").strip()
    return {"pass": "red" in txt.lower(), "answer": txt[:60], "wall": round(wall, 2),
            "prompt_tokens": (d.get("usage") or {}).get("prompt_tokens")}


# ---------------------------------------------------------------- 8. anthropic

def run_anthropic(base, args):
    hdr = {"x-api-key": "dummy", "anthropic-version": "2023-06-01"}
    out = {}
    d, wall = chat(base, {"model": MODEL, "max_tokens": 1500,
                          "messages": [{"role": "user", "content": "What is 6*7? Reply with the number."}]},
                   sid("anth-text"), extra_headers=hdr, path="/v1/messages")
    blocks = d.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    out["text"] = {"pass": "42" in text, "answer": text[:40], "block_types": [b.get("type") for b in blocks],
                   "usage": d.get("usage"), "wall": round(wall, 2)}
    d2, wall2 = chat(base, {"model": MODEL, "max_tokens": 1500,
                            "messages": [{"role": "user", "content": "What is the weather in Paris? "
                                          "Use the get_weather tool."}],
                            "tools": [{"name": "get_weather", "description": "Current weather for a city",
                                       "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                                                        "required": ["city"]}}]},
                     sid("anth-tool"), extra_headers=hdr, path="/v1/messages")
    blocks2 = d2.get("content") or []
    uses = [b for b in blocks2 if b.get("type") == "tool_use"]
    out["tool_use"] = {"pass": bool(uses), "calls": [{"name": b.get("name"), "input": b.get("input")} for b in uses],
                       "block_types": [b.get("type") for b in blocks2], "stop_reason": d2.get("stop_reason"),
                       "wall": round(wall2, 2)}
    return out


# ---------------------------------------------------------------- 9. metrics

def run_metrics(base, args):
    out = {}
    for name, path in (("health", "/health"), ("props", "/props")):
        try:
            with _slots:
                resp = _open(base, path, None, sid("metrics"), 60)
                with resp:
                    raw = resp.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            out[name] = {"status": "ok", "truncated_json": raw[:2048], "bytes": len(raw)}
            if name == "health" and isinstance(parsed, dict):
                out[name].update({k: parsed.get(k) for k in ("status", "loaded", "max_context", "images", "service")})
            if name == "props" and isinstance(parsed, dict):
                out[name]["keys"] = sorted(parsed.keys())[:20]
                out[name]["total_slots"] = parsed.get("total_slots")
        except Exception as e:  # noqa: BLE001
            out[name] = _err_dict(e)
    return out


TESTS = {"speed": run_speed, "tools": run_tools, "prefix": run_prefix, "park": run_park, "needle": run_needle,
         "quality": run_quality, "effort": run_effort, "vision": run_vision, "anthropic": run_anthropic,
         "metrics": run_metrics}


def summarize_line(name, r):
    """One-line summary per test."""
    def g(*keys, default="-"):
        cur = r
        for k in keys:
            if not isinstance(cur, dict) or cur.get(k) is None:
                return default
            cur = cur[k]
        return cur
    if "error" in r and len(r) <= 2:
        return f"{name:<10} ERROR {r['error'][:100]}"
    if name == "speed":
        parts = []
        for kind in ("code", "prose"):
            a = g(kind, "one_stream", "median_decode_tps")
            t = g(kind, "one_stream", "median_ttft")
            b = g(kind, "two_streams", "per_stream_median_decode_tps")
            agg = g(kind, "two_streams", "aggregate_decode_tps")
            c = g(kind, "temp0_one_stream", "decode_tps")
            parts.append(f"{kind} 1x={a} (ttft {t}) 2x={b}/agg{agg} t0={c}")
        return f"{name:<10} " + " | ".join(parts)
    if name == "tools":
        e1, e0 = r.get("edit_file_temp1", {}), r.get("edit_file_temp0", {})
        return (f"{name:<10} edit_file t1.0 {e1.get('valid_pct')}% of {e1.get('n')} "
                f"{e1.get('counts')} in {e1.get('seconds')}s | t0 {e0.get('valid_pct')}% of {e0.get('n')} "
                f"| multi-tool both {r.get('multi_tool', {}).get('both_pct')}% of {r.get('multi_tool', {}).get('n')}")
    if name == "prefix":
        return (f"{name:<10} prompt={g('turn1', 'prompt_tokens')} cached1={g('turn1', 'cached_tokens')} "
                f"ttft t1={g('turn1', 'ttft')} t2={g('turn2_same_session', 'ttft')} "
                f"t3={g('turn3_same_session', 'ttft')} new-session={g('turn3_different_session', 'ttft')} "
                f"speedup={r.get('turn2_ttft_ratio_vs_turn1')}x fact_ok={r.get('answer_contains_fact')} "
                f"cross_session_reuse={r.get('cross_session_reuse')}")
    if name == "park":
        return (f"{name:<10} cold ttft A/B/C={r.get('cold_ttft')} -> return A={g('return_ttft', 'A')} "
                f"B={g('return_ttft', 'B')} speedup={r.get('return_speedup_vs_cold')} "
                f"restored={r.get('restored')} facts_ok={r.get('facts_ok')} cache={r.get('cache_stats') or '-'}")
    if name == "needle":
        last = (r.get("attempts") or [{}])[-1]
        return (f"{name:<10} {'PASS' if r.get('pass') else 'FAIL'} prompt={r.get('prompt_tokens')} "
                f"prefill={r.get('prefill_tps')} tok/s in {r.get('ttft_seconds')}s, "
                f"decode={r.get('decode_tps')} tok/s answer={last.get('answer_excerpt')!r} "
                f"attempts={len(r.get('attempts') or [])}")
    if name == "quality":
        return f"{name:<10} {r.get('correct')}/{r.get('n')} correct (seed {r.get('seed')}, effort low, temp 0)"
    if name == "effort":
        parts = []
        for k in ("none", "low", "medium", "high"):
            e = r.get(k) if isinstance(r.get(k), dict) else {}
            tok = e.get("reasoning_tokens")
            tok = (f"{tok}tok" if tok is not None else f"~{e.get('estimated_reasoning_tokens')}tok")
            flag = "ok" if e.get("correct") else ("LEN" if e.get("finish_reason") == "length" else "WRONG")
            parts.append(f"{k}:{tok}/{e.get('reasoning_chars')}ch/{e.get('answer') or '-'}:{flag}")
        return f"{name:<10} " + " ".join(parts)
    if name == "vision":
        return f"{name:<10} {'PASS' if r.get('pass') else 'FAIL'} answer={r.get('answer')!r} ({r.get('wall')}s)"
    if name == "anthropic":
        return (f"{name:<10} text {'PASS' if g('text', 'pass') else 'FAIL'} {g('text', 'answer')!r} | "
                f"tool_use {'PASS' if g('tool_use', 'pass') else 'FAIL'} {g('tool_use', 'calls')}")
    if name == "metrics":
        return f"{name:<10} health={g('health', 'status')} service={g('health', 'service')} " f"ctx={g('health', 'max_context')} loaded={g('health', 'loaded')} props={g('props', 'status')}"
    return f"{name:<10} {json.dumps(r)[:140]}"


def headline_rows(r):
    """(label, value) pairs used by the stdout summary table and --compare."""
    rows = []

    def val(*keys):
        cur = r
        for k in keys:
            if not isinstance(cur, dict) or k not in cur or cur[k] is None:
                return None
            cur = cur[k]
        return cur
    for kind in ("code", "prose"):
        rows.append((f"speed {kind} tok/s 1 stream", val("speed", kind, "one_stream", "median_decode_tps")))
        rows.append((f"speed {kind} tok/s 2 streams", val("speed", kind, "two_streams", "per_stream_median_decode_tps")))
        rows.append((f"speed {kind} agg tok/s 2 streams", val("speed", kind, "two_streams", "aggregate_decode_tps")))
        rows.append((f"speed {kind} TTFT s", val("speed", kind, "one_stream", "median_ttft")))
    rows.append(("tools edit_file valid % t1.0", val("tools", "edit_file_temp1", "valid_pct")))
    rows.append(("tools edit_file valid % t0", val("tools", "edit_file_temp0", "valid_pct")))
    rows.append(("tools multi-tool both %", val("tools", "multi_tool", "both_pct")))
    rows.append(("prefix prompt tokens", val("prefix", "turn1", "prompt_tokens")))
    rows.append(("prefix TTFT turn1 s", val("prefix", "turn1", "ttft")))
    rows.append(("prefix TTFT turn2 s (same session)", val("prefix", "turn2_same_session", "ttft")))
    rows.append(("prefix TTFT turn3 new session s", val("prefix", "turn3_different_session", "ttft")))
    rows.append(("prefix cached turn2", val("prefix", "turn2_same_session", "cached_tokens")))
    rows.append(("park cold TTFT worst s", max([t for t in (val("park", "cold_ttft") or []) if t], default=None)
                 if val("park", "cold_ttft") else None))
    rows.append(("park return TTFT A s", val("park", "return_ttft", "A")))
    rows.append(("park return TTFT B s", val("park", "return_ttft", "B")))
    rows.append(("park facts ok", val("park", "facts_ok")))
    rows.append(("needle prefill tok/s", val("needle", "prefill_tps")))
    rows.append(("needle decode tok/s", val("needle", "decode_tps")))
    rows.append(("needle", "PASS" if val("needle", "pass") else ("FAIL" if "needle" in r else None)))
    rows.append(("quality correct", val("quality", "correct")))
    if val("quality", "n") is not None:
        rows[-1] = (f"quality correct of {val('quality', 'n')}", val("quality", "correct"))
    for k in ("none", "low", "medium", "high"):
        rows.append((f"effort {k} reasoning tokens",
                     val("effort", k, "reasoning_tokens") or val("effort", k, "estimated_reasoning_tokens")))
    rows.append(("vision", "PASS" if val("vision", "pass") else ("FAIL" if "vision" in r else None)))
    rows.append(("anthropic text", "PASS" if val("anthropic", "text", "pass") else None))
    rows.append(("anthropic tool_use", "PASS" if val("anthropic", "tool_use", "pass") else None))
    rows.append(("total wall seconds", val("_wall")))
    return [(k, v) for k, v in rows if v is not None]

def fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def do_compare(a_path, b_path):
    a, b = (json.load(open(p)) for p in (a_path, b_path))
    ra, rb = a.get("results", {}), b.get("results", {})
    ra["_wall"] = a.get("wall_seconds")
    rb["_wall"] = b.get("wall_seconds")
    la, lb = a.get("label", a_path), b.get("label", b_path)
    rows_a, rows_b = dict(headline_rows(ra)), dict(headline_rows(rb))
    keys = list(dict.fromkeys(list(rows_a) + list(rows_b)))
    print(f"{'metric':<36} {la[:22]:>22} {lb[:22]:>22} {'delta B-A':>14} {'%':>8}")
    print("-" * 106)
    for k in keys:
        va, vb = rows_a.get(k), rows_b.get(k)
        delta = pct = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            delta = f"{vb - va:+.2f}" if abs(vb - va) < 10 else f"{vb - va:+.1f}"
            if va:
                pct = f"{100.0 * (vb - va) / va:+.1f}"
        print(f"{k:<36} {(fmt(va) if va is not None else '-'):>22} "
              f"{(fmt(vb) if vb is not None else '-'):>22} {delta:>14} {pct:>8}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Inkling 2 Mini (Basalt) evaluation harness")
    ap.add_argument("--base", default="http://127.0.0.1:18100")
    ap.add_argument("--suite", choices=["quick", "full"], default="quick")
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument("--label", default=None)
    ap.add_argument("--only", help="comma list: " + ",".join(TESTS))
    ap.add_argument("--needle-tokens", type=int, default=120000)
    ap.add_argument("--prefix-tokens", type=int, default=23000)
    ap.add_argument("--park-tokens", type=int, default=20000, help="filler per parked conversation (park test)")
    ap.add_argument("--tool-n-temp1", type=int, default=30, help="edit_file cases at temperature 1.0")
    ap.add_argument("--tool-n-temp0", type=int, default=20, help="edit_file cases at temperature 0")
    ap.add_argument("--multi-tool-n", type=int, default=10)
    ap.add_argument("--quality-n", type=int, default=20)
    ap.add_argument("--effort-max-tokens", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--budget-min", type=float, default=0.0, help="skip remaining tests past N minutes (0 = off)")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args(argv)

    if args.compare:
        do_compare(*args.compare)
        return 0

    global LABEL
    args.base = args.base.rstrip("/")
    LABEL = re.sub(r"[^A-Za-z0-9._-]", "-", args.label or time.strftime("%Y%m%d-%H%M%S"))[:24]
    names = ([s.strip() for s in args.only.split(",") if s.strip()] if args.only
             else (SUITE_QUICK if args.suite == "quick" else SUITE_FULL))
    bad = [n for n in names if n not in TESTS]
    if bad:
        ap.error(f"unknown test(s): {bad}; choose from {','.join(TESTS)}")

    print(f"inkling_eval label={LABEL} base={args.base} suite={args.suite} tests={','.join(names)}", flush=True)
    started = time.time()
    results = {}
    for name in names:
        if args.budget_min and (time.time() - started) > args.budget_min * 60:
            results[name] = {"skipped": f"over budget {args.budget_min} min"}
            print(f"{name:<10} SKIPPED (over budget {args.budget_min} min)", flush=True)
            continue
        t0 = time.time()
        try:
            r = TESTS[name](args.base, args)
        except Exception as e:  # noqa: BLE001
            r = _err_dict(e)
        r["_seconds"] = round(time.time() - t0, 1)
        results[name] = r
        print(f"{summarize_line(name, r)}  [{r['_seconds']}s]", flush=True)

    wall = round(time.time() - started, 1)
    payload = {"label": args.label or LABEL, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "base": args.base, "suite": args.suite, "model": MODEL, "only": names,
               "args": {k: getattr(args, k) for k in ("needle_tokens", "prefix_tokens", "park_tokens",
                                                      "tool_n_temp1", "tool_n_temp0", "multi_tool_n", "quality_n", "seed")},
               "wall_seconds": wall, "results": results}
    if args.out:
        import os
        os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
        with open(os.path.expanduser(args.out), "w") as f:
            json.dump(payload, f, indent=1)
        print(f"wrote {os.path.expanduser(args.out)}", flush=True)

    print(f"\n{'=' * 78}\nheadline numbers ({payload['label']})\n{'=' * 78}")
    results["_wall"] = wall
    for k, v in headline_rows(results):
        print(f"  {k:<36} {fmt(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
