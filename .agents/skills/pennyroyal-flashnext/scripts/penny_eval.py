#!/usr/bin/env python3
"""Before/after harness for the Pennyroyal v2.5.3 -> v3.1.7 upgrade.

Stdlib only, Python 3.11+. Served endpoint: Pennyroyal (SGLang fork) serving
Qwen3.8-Flash-Next, served model name "pennyroyal" (port-forward of the Service).

usage:
  python3 penny_eval.py --base http://127.0.0.1:18200 --suite quick \
      --label penny-2.5.3 --out base-2.5.3.json
  python3 penny_eval.py --only hicache --hicache-trials 2 --out base-2.5.3-hicache.json
  python3 penny_eval.py --compare base-2.5.3.json new-3.1.7.json

At most 4 concurrent requests, EXCEPT the explicit C6 speed step (6 concurrent,
full suite only). Requests are streamed where TTFT matters. 429/503 and transport
errors retry 3x with 10 s backoff. Per-request timeout 900 s.

Tests:
  speed       streaming decode tok/s (TTFT excluded), code + prose prompts, temp 0.7,
              max_tokens 800: C1 x3 (median), C4 (per-stream + aggregate), C6 (aggregate, full only)
  warm-ttft   ~30K-token system prompt + short user turn; send twice; TTFT cold/warm + cached_tokens
  prefill-cached  ~200K-token prefix warm, then prefix + ~4K new tokens; fresh-input prefill tok/s
  hicache     issue #74 procedure (full suite or --only hicache): unique ~200K-token docs, cold ->
              warm -> flush x N -> return; per-step TTFT + /metrics cached-token delta; PASS = code in reply
  tools       qwen3_coder edit_file probe (N=20, temp 0.7) + a 2-tool parallel call
  compat      C1-C8 (r3 §4): status + PASS/FAIL per item; never crash on one failure
  info        /get_server_info (truncated 3 KB) + /health
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

MODEL = "pennyroyal"
TIMEOUT = 900.0
RETRIES = 3
RETRY_BACKOFF = 10.0
RETRY_CODES = {429, 503}
MAX_CONCURRENCY = 4
C6_CONCURRENCY = 6
SUITE_QUICK = ["speed", "warm-ttft", "prefill-cached", "tools", "compat", "info"]
SUITE_FULL = ["speed", "warm-ttft", "prefill-cached", "hicache", "tools", "compat", "info"]

_slots = threading.Semaphore(MAX_CONCURRENCY)
_slots_c6 = threading.Semaphore(C6_CONCURRENCY)
NONCE = uuid.uuid4().hex[:6]
LABEL = "run"

# ------------------------------------------------------------- HTTP plumbing

def sid(name: str) -> str:
    return f"penny-eval-{LABEL}-{NONCE}-{name}"[:120]


def _open(base, path, body, session, timeout, extra_headers=None):
    headers = {"Content-Type": "application/json", "x-session-id": session}
    headers.update(extra_headers or {})
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def _retry(fn, log, slot=None):
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


def _retry_capture(fn):
    """Like _retry but never raises: returns (result|None, err_dict)."""
    try:
        return fn(), None
    except Exception as e:  # noqa: BLE001
        return None, _err_dict(e)


def chat_stream(base, body, session, timeout=TIMEOUT, slot=None):
    """Streaming POST -> timing dict. Decode rate excludes TTFT."""
    def go():
        sem = slot if slot is not None else _slots
        payload = dict(body, stream=True, stream_options={"include_usage": True})
        with sem:
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
        return out

    return _retry(go, [], slot=slot)


def post_path(base, path, body, session, timeout=TIMEOUT, extra_headers=None):
    """Plain POST returning (dict|None, http_status|None, error_dict|None, wall)."""
    def go():
        with _slots:
            t0 = time.time()
            try:
                resp = _open(base, path, body, session, timeout, extra_headers)
                with resp:
                    raw = resp.read().decode("utf-8", "replace")
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"_raw": raw[:2000]}
                return parsed, resp.status, None, time.time() - t0
            except urllib.error.HTTPError as e:
                try:
                    raw = e.read().decode("utf-8", "replace")[:400]
                except Exception:  # noqa: BLE001
                    raw = ""
                return None, e.code, {"http_status": e.code, "body": raw}, time.time() - t0
    return _retry(go, [])


def get_json(base, path, timeout=60):
    def go():
        with _slots:
            resp = _open(base, path, None, sid("get"), timeout)
            with resp:
                raw = resp.read().decode("utf-8", "replace")
            return raw, resp.status
    return _retry(go, [])


def metrics_text(base, timeout=60):
    def go():
        with _slots:
            resp = _open(base, "/metrics", None, sid("metrics"), timeout)
            with resp:
                return resp.read().decode("utf-8", "replace")
    return _retry(go, [])


def cached_by_source(base):
    """{cache_source: total} from sglang:cached_tokens_total{...cache_source=...}."""
    try:
        text = metrics_text(base)
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("sglang:cached_tokens_total"):
            continue
        m = re.search(r'cache_source="([^"]+)"', line)
        if not m:
            continue
        try:
            out[m.group(1)] = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
    return out


# ------------------------------------------------------------- prompt blobs

WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()


def doc_line(i):
    return f"Line {i}: " + " ".join(WORDS[(i + j) % len(WORDS)] for j in range(6)) + "."


def build_doc(nonce, secret, needle_line, lines):
    out = [f"Document id {nonce}."]
    for i in range(lines):
        if i == needle_line:
            out.append(f"Line {i}: the secret code is {secret} -- remember this exactly.")
        else:
            out.append(doc_line(i))
    return "\n".join(out)


_calib = {"tok_per_line": None}


def calibrate_tokens_per_line(base):
    """Calibrate filler tokens/line with one cheap probe (~300 lines)."""
    if _calib["tok_per_line"]:
        return _calib["tok_per_line"]
    probe = "\n".join(doc_line(i) for i in range(300))
    try:
        d, _ = chat(base, {"model": MODEL, "max_tokens": 1, "temperature": 0,
                           "messages": [{"role": "user", "content": probe}]}, sid("calib"))
        pt = (d.get("usage") or {}).get("prompt_tokens")
        _calib["tok_per_line"] = (pt / 300.0) if pt else 16.0
    except Exception:  # noqa: BLE001
        _calib["tok_per_line"] = 16.0
    return _calib["tok_per_line"]


def filler_block(base, target_tokens):
    per = calibrate_tokens_per_line(base)
    lines = max(50, int(target_tokens / per))
    # run-unique first line so the whole prefix is genuinely cold for this run
    head = f"run id {NONCE} start."
    return head + "\n" + "\n".join(doc_line(i) for i in range(lines)), lines + 1


# ------------------------------------------------------------- reasoning probe

def detect_reasoning_off(base, log):
    """Check what the server honours for turning thinking off.

    Probe: chat_template_kwargs {"enable_thinking": false}. If the model still
    emits reasoning_content (or reasoning usage), that form is not honoured;
    fall back to reasoning_effort="low".
    Returns (body_extra, mode_str).
    """
    body = {"model": MODEL, "max_tokens": 300, "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}]}
    try:
        d, _ = chat(base, body, sid("probe-kwargs"))
    except Exception as e:  # noqa: BLE001
        log.append(f"probe kwargs failed: {e.__class__.__name__}: {str(e)[:120]}")
        return {"reasoning_effort": "low"}, "reasoning_effort=low (kwargs probe errored)"
    msg = ((d.get("choices") or [{}])[0].get("message") or {})
    rtoks = ((d.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
    has_reacon = bool(msg.get("reasoning_content") or msg.get("reasoning"))
    if has_reacon or (rtoks or 0) > 0:
        # kwargs not honoured; try reasoning_effort=low
        body2 = dict(body)
        body2.pop("chat_template_kwargs")
        body2["reasoning_effort"] = "low"
        try:
            d2, _ = chat(base, body2, sid("probe-low"))
            msg2 = ((d2.get("choices") or [{}])[0].get("message") or {})
            rtoks2 = ((d2.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
            has_r2 = bool(msg2.get("reasoning_content") or msg2.get("reasoning")) or (rtoks2 or 0) > 0
            if has_r2:
                return {"reasoning_effort": "low"}, "reasoning_effort=low (kwargs ignored; low still emits reasoning)"
            return {"reasoning_effort": "low"}, "reasoning_effort=low (kwargs ignored)"
        except Exception as e:  # noqa: BLE001
            log.append(f"probe low failed: {e.__class__.__name__}")
            return {"reasoning_effort": "low"}, "reasoning_effort=low (probe errored)"
    return {"chat_template_kwargs": {"enable_thinking": False}}, "chat_template_kwargs enable_thinking=false"


# ------------------------------------------------------------- 1. speed

SPEED_PROMPTS = {
    "code": ("Write a thread-safe LRU cache module in Python: class LRUCache with get/put, capacity, "
             "threading.Lock, and a unittest.TestCase covering eviction and concurrent access. "
             "Reply with only one python code block, no prose."),
    "prose": ("Write a 600-word short story about a lighthouse keeper who discovers that the sea keeps "
              "returning lost objects. Begin it in medias res. Prose only."),
}


def run_speed(base, args):
    log = []
    reasoning_extra, reasoning_mode = detect_reasoning_off(base, log)
    out = {"params": {"reasoning": reasoning_mode, "temperature": 0.7, "max_tokens": 800}, "log": log}

    for name, prompt in SPEED_PROMPTS.items():
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800, "temperature": 0.7, **reasoning_extra}
        # C1 x3 (sequential, median)
        single = []
        for i in range(3):
            r, err = _retry_capture(lambda i=i: chat_stream(base, body, sid(f"speed-{name}-{i}")))
            if err:
                single.append({"error": err.get("error"), "http_status": err.get("http_status")})
            else:
                single.append(r)
        ok = [r for r in single if r.get("decode_tps")]
        out[name] = {
            "one_stream": {
                "runs": len(single),
                "median_decode_tps": round(statistics.median(r["decode_tps"] for r in ok), 1) if ok else None,
                "all_decode_tps": [r.get("decode_tps") for r in single],
                "median_ttft": round(statistics.median(r["ttft"] for r in single if r.get("ttft")), 3) if single else None,
                "median_completion_tokens": round(statistics.median(r["completion_tokens"] for r in single if r.get("completion_tokens"))) if single else None,
                "engine_decode_tps": [r.get("engine_decode_tps") for r in single],
                "errors": [r.get("error") for r in single if r.get("error")],
            },
        }
        # C4: 4 concurrent
        with ThreadPoolExecutor(MAX_CONCURRENCY) as ex:
            pairs = list(ex.map(
                lambda i: _retry_capture(lambda i=i: chat_stream(base, body, sid(f"speed-{name}-c4-{i}"))),
                range(4)))
        pairs = [p[0] if p[0] else {"error": (p[1] or {}).get("error"), "http_status": (p[1] or {}).get("http_status")} for p in pairs]
        okp = [r for r in pairs if r.get("decode_tps")]
        wall = max((r["wall"] for r in pairs if r.get("wall")), default=0.0)
        out[name]["four_streams"] = {
            "per_stream_median_decode_tps": round(statistics.median(r["decode_tps"] for r in okp), 1) if okp else None,
            "per_stream_decode_tps": [r.get("decode_tps") for r in pairs],
            "aggregate_decode_tps": round(sum(r["completion_tokens"] for r in okp) / wall, 1) if okp and wall else None,
            "median_ttft": round(statistics.median(r["ttft"] for r in pairs if r.get("ttft")), 3) if pairs else None,
            "wall": round(wall, 3),
            "errors": [r.get("error") for r in pairs if r.get("error")],
        }
        # C6: 6 concurrent (full suite only)
        if args.suite == "full":
            with ThreadPoolExecutor(C6_CONCURRENCY) as ex:
                six = list(ex.map(
                    lambda i: _retry_capture(lambda i=i: chat_stream(base, body, sid(f"speed-{name}-c6-{i}"), slot=_slots_c6)),
                    range(6)))
            six = [s[0] if s[0] else {"error": (s[1] or {}).get("error"), "http_status": (s[1] or {}).get("http_status")} for s in six]
            ok6 = [r for r in six if r.get("decode_tps")]
            wall6 = max((r["wall"] for r in six if r.get("wall")), default=0.0)
            out[name]["six_streams"] = {
                "aggregate_decode_tps": round(sum(r["completion_tokens"] for r in ok6) / wall6, 1) if ok6 and wall6 else None,
                "per_stream_decode_tps": [r.get("decode_tps") for r in six],
                "median_ttft": round(statistics.median(r["ttft"] for r in six if r.get("ttft")), 3) if six else None,
                "wall": round(wall6, 3),
                "errors": [r.get("error") for r in six if r.get("error")],
            }
    return out


# ------------------------------------------------------------- 2. warm-ttft

def run_warm_ttft(base, args):
    target = 30000
    blob, lines = filler_block(base, target)
    system = ("You are a coding agent. Below is a numbered repository dump.\n\n" + blob)
    body = {"model": MODEL, "temperature": 0, "max_tokens": 300, "reasoning_effort": "low",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": "Say OK."}]}
    session = sid("warm-ttft")
    r1, err1 = _retry_capture(lambda: chat_stream(base, body, session))
    time.sleep(1.0)
    r2, err2 = _retry_capture(lambda: chat_stream(base, body, session))
    succ = lambda r: r if isinstance(r, dict) and not r.get("error") else None
    r1, r2 = succ(r1), succ(r2)
    return {
        "target_tokens": target, "filler_lines": lines,
        "cold": {k: r1.get(k) for k in ("ttft", "prompt_tokens", "cached_tokens", "wall", "finish_reason")} if r1 else {"error": (err1 or {}).get("error")},
        "warm": {k: r2.get(k) for k in ("ttft", "prompt_tokens", "cached_tokens", "wall", "finish_reason")} if r2 else {"error": (err2 or {}).get("error")},
        "errors": [e.get("error") for e in (err1, err2) if e],
    }


# ------------------------------------------------------------- 3. prefill-cached

def run_prefill_cached(base, args):
    target = args.prefix_tokens
    blob, lines = filler_block(base, target)
    q1 = "Question 1: Read the dump and say OK."
    body = {"model": MODEL, "temperature": 0, "max_tokens": 10, "reasoning_effort": "low",
            "messages": [{"role": "user", "content": blob + "\n\n" + q1}]}
    session = sid("prefill-cached")
    warm, err_w = _retry_capture(lambda: chat_stream(base, body, session))
    warm_ok = warm and not warm.get("error")
    # new user turn with ~4K NEW tokens appended to the same prefix
    new_blob, _ = filler_block(base, 4000)
    # make the new segment unique to this run so it is genuinely new tokens
    new_blob = f"Fresh tail {NONCE} run marker.\n" + new_blob
    body2 = {"model": MODEL, "temperature": 0, "max_tokens": 10, "reasoning_effort": "low",
             "messages": [{"role": "user", "content": blob + "\n\n" + new_blob + "\n\nQuestion 2: Answer with OK."}]}
    r2, err2 = _retry_capture(lambda: chat_stream(base, body2, session))
    r2_ok = r2 and not r2.get("error")
    out = {
        "target_tokens": target, "filler_lines": lines,
        "warm_prefix": {k: warm.get(k) for k in ("ttft", "prompt_tokens", "cached_tokens", "wall")} if warm_ok else {"error": (err_w or {}).get("error")},
        "fresh_turn": {k: r2.get(k) for k in ("ttft", "prompt_tokens", "cached_tokens", "wall", "completion_tokens")} if r2_ok else {"error": (err2 or {}).get("error")},
        "errors": [e.get("error") for e in (err_w, err2) if e],
    }
    if warm_ok and r2_ok:
        new_tokens = max(0, (r2.get("prompt_tokens") or 0) - (warm.get("prompt_tokens") or 0))
        out["new_tokens"] = new_tokens
        out["fresh_prefill_tps"] = round(new_tokens / r2["ttft"], 1) if r2.get("ttft") else None
        out["cached_delta"] = (r2.get("cached_tokens") or 0) - (warm.get("cached_tokens") or 0)
    else:
        out["new_tokens"] = None
        out["fresh_prefill_tps"] = None
    # full suite: C4 variant (4 concurrent identical fresh turns)
    if args.suite == "full":
        with ThreadPoolExecutor(MAX_CONCURRENCY) as ex:
            res = list(ex.map(lambda i: _retry_capture(lambda i=i: chat_stream(base, body2, sid(f"prefill-cached-c4-{i}"))), range(4)))
        res = [r[0] if r[0] else {"error": (r[1] or {}).get("error")} for r in res]
        ok4 = [r for r in res if r.get("decode_tps") or r.get("ttft")]
        # fresh-input prefill tok/s aggregate over wall time
        wall = max((r["wall"] for r in ok4), default=0.0)
        total_new = sum(max(0, (r.get("prompt_tokens") or 0) - (warm.get("prompt_tokens") or 0)) for r in ok4) if warm_ok else 0
        out["c4"] = {
            "wall": round(wall, 3) if wall else None,
            "total_new_tokens": total_new,
            "aggregate_fresh_prefill_tps": round(total_new / wall, 1) if wall and total_new else None,
            "median_ttft": round(statistics.median(r["ttft"] for r in ok4 if r.get("ttft")), 3) if ok4 else None,
            "errors": [r.get("error") for r in res if r.get("error")],
        }
    return out


# ------------------------------------------------------------- 4. hicache

def run_hicache(base, args):
    trials = args.hicache_trials
    doc_tokens = args.hicache_doc_tokens
    flush_docs = args.flush_docs
    per = calibrate_tokens_per_line(base)
    lines = max(50, int(doc_tokens / per))
    rng = random.Random(NONCE)

    def ask(doc, secret, session):
        body = {"model": MODEL, "max_tokens": 32, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": doc + "\n\nWhat is the secret code? Reply with the code only."}]}
        before = cached_by_source(base) or {}
        t0 = time.time()
        r, err = _retry_capture(lambda: chat(base, body, session))
        if isinstance(r, tuple):
            r = r[0]
        lat = time.time() - t0
        after = cached_by_source(base)
        delta = None
        if after is not None:
            delta = {k: int(after.get(k, 0) - before.get(k, 0)) for k in after}
        if r is None:
            return {"lat": round(lat, 2), "prompt_tokens": None, "delta": delta, "pass": False,
                    "reply": None, "error": (err or {}).get("error")}
        choice = (r.get("choices") or [{}])[0]
        reply = (choice.get("message") or {}).get("content") or ""
        usage = r.get("usage") or {}
        return {"lat": round(lat, 2),
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
                "delta": delta,
                "pass": secret in reply,
                "reply": reply.strip()[:40]}

    out = {"target_doc_tokens": doc_tokens, "lines": lines, "flush_docs": flush_docs,
           "trials": [], "n": trials, "n_pass": 0}
    for t in range(trials):
        nonce = rng.randrange(10**9)
        secret = f"TEAL-{rng.randrange(1000, 9999)}-{t}"
        needle_line = rng.randrange(lines // 10, 9 * lines // 10)
        doc = build_doc(nonce, secret, needle_line, lines)
        sess = sid(f"hicache-{t}")
        steps = {}
        steps["cold"] = ask(doc, secret, sid(f"hicache-{t}-cold"))
        steps["warm"] = ask(doc, secret, sid(f"hicache-{t}-warm"))
        flushes = []
        for f in range(flush_docs):
            s = f"GREY-{rng.randrange(1000, 9999)}-f{f}"
            fdoc = build_doc(rng.randrange(10**9), s, lines // 2, lines)
            flushes.append(ask(fdoc, s, sid(f"hicache-{t}-flush{f}")))
        steps["flush"] = flushes
        steps["return"] = ask(doc, secret, sid(f"hicache-{t}-return"))
        ret = steps["return"]
        host_hit = bool(ret["delta"] and any(v > 0 for k, v in ret["delta"].items() if k not in ("none",)))
        host_only_hit = bool(ret["delta"] and (ret["delta"].get("host") or 0) > 0 and not (ret["delta"].get("device") or 0) > 0)
        cold_lat = steps["cold"].get("lat")
        return_lat = ret.get("lat")
        host_tier_looks_like_hit = bool(cold_lat and return_lat and return_lat < 0.5 * cold_lat)
        out["trials"].append({
            "trial": t, "doc": nonce, "secret": secret, "needle_line": needle_line,
            "cold": steps["cold"], "warm": steps["warm"], "flush": flushes, "return": ret,
            "return_host_tier_hit": host_tier_looks_like_hit,
            "return_host_delta": ret["delta"],
            "pass": bool(ret["pass"]),
        })
        if ret["pass"]:
            out["n_pass"] += 1
    out["pass_rate"] = f"{out['n_pass']}/{out['n']}"
    out["note"] = "PASS = needle code in reply on host-tier return (issue #74 procedure)"
    return out


# ------------------------------------------------------------- 5. tools

EDIT_TOOL = {"type": "function", "function": {
    "name": "edit_file", "description": "Replace exact text in a file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                    "old_string": {"type": "string"}, "new_string": {"type": "string"}},
                    "required": ["path", "old_string", "new_string"]}}}

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
TIME_TOOL = {"type": "function", "function": {"name": "get_time", "description": "Current time in a timezone",
            "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}}


def _edit_case(base, i, temp, session):
    args = {"path": f"src/module_{i}.py", "old_string": f'value_{i} = "{i * 7}"',
            "new_string": f'value_{i} = "{i * 7}"  # edited {i}'}
    body = {"model": MODEL, "temperature": temp, "max_tokens": 2048, "tools": [EDIT_TOOL],
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


def run_tools(base, args):
    n = 20
    counts = {"valid": 0, "wrong-args": 0, "text": 0, "truncated": 0, "error": 0}
    fails = []
    t0 = time.time()
    with ThreadPoolExecutor(MAX_CONCURRENCY) as ex:
        res = list(ex.map(lambda i: _edit_case(base, i, 0.7, sid(f"tool-{i % 2}")), range(1, n + 1)))
    for i, kind, info in res:
        counts[kind] += 1
        if kind != "valid" and len(fails) < 5:
            fails.append({"case": i, "kind": kind, "info": info})
    out = {"edit_file_temp07": {"n": n, "temperature": 0.7, "counts": counts,
                                "valid_pct": round(100.0 * counts["valid"] / n, 1),
                                "seconds": round(time.time() - t0, 1),
                                "sample_failures": fails}}
    # 2-tool parallel call
    body = {"model": MODEL, "temperature": 0.7, "max_tokens": 2048, "tools": [WEATHER, TIME_TOOL],
            "messages": [{"role": "user", "content": "What is the weather in Paris and what time is it there? "
                         "Call both get_weather and get_time as tool calls."}]}
    r, err = _retry_capture(lambda: chat(base, body, sid("multi")))
    if r is None:
        out["multi_tool"] = {"error": (err or {}).get("error"), "http_status": (err or {}).get("http_status")}
    elif isinstance(r, tuple):
        r = r[0]
        msg = (r.get("choices") or [{}])[0].get("message") or {}
        names = [(tc.get("function") or {}).get("name") for tc in msg.get("tool_calls") or []]
        out["multi_tool"] = {"names": names, "both": {"get_weather", "get_time"} <= set(names)}
    return out


# ------------------------------------------------------------- 6. compat

def _anthropic_headers():
    return {"x-api-key": "dummy", "anthropic-version": "2023-06-01"}


def _red_png(width=64, height=64):
    row = b"\x00" + b"\xff\x00\x00" * width
    raw = row * height

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _responses_body(reasoning_effort=None):
    body = {"model": MODEL, "input": "What is 2+2? Answer with just the number.",
            "tools": [{"type": "function", "name": "math.add", "description": "Add two numbers",
                       "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                                      "required": ["a", "b"]}}]}
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def run_compat(base, args):
    hdr = _anthropic_headers()
    out = {}

    def anthropic(body, extra_headers=None, path="/v1/messages"):
        return post_path(base, path, body, sid("compat"), extra_headers=dict(hdr, **(extra_headers or {})))

    # C1: Anthropic thinking enabled, no budget
    r, status, err, wall = anthropic({"model": MODEL, "max_tokens": 256,
                                      "thinking": {"type": "enabled"},
                                      "messages": [{"role": "user", "content": "say ok"}]})
    out["C1_anthropic_thinking_no_budget"] = {"http_status": status, "wall": round(wall, 2),
                                              "pass": status == 200,
                                              "error": (err or {}).get("error") if status != 200 else None,
                                              "blocks": [b.get("type") for b in (r or {}).get("content", [])] if r else None}

    # C2: thinking budget 1024 + display updates
    r, status, err, wall = anthropic({"model": MODEL, "max_tokens": 512,
                                      "thinking": {"type": "enabled", "budget_tokens": 1024, "display": "updates"},
                                      "messages": [{"role": "user", "content": "say ok"}]})
    out["C2_anthropic_thinking_updates"] = {"http_status": status, "wall": round(wall, 2),
                                            "pass": status == 200,
                                            "error": (err or {}).get("error") if status != 200 else None,
                                            "blocks": [b.get("type") for b in (r or {}).get("content", [])] if r else None}

    # C3: nested tool_reference inside tool_result content, beta=true + mcp-client-*
    c3_body = {"model": MODEL, "max_tokens": 256,
               "messages": [
                   {"role": "user", "content": "Use the lookup tool."},
                   {"role": "assistant", "content": [
                       {"type": "text", "text": "I'll look that up."},
                       {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"key": "foo"}}]},
                   {"role": "user", "content": [
                       {"type": "tool_result", "tool_use_id": "toolu_1",
                        "content": [{"type": "tool_reference",
                                     "tool_name": "lookup",
                                     "tool_id": "toolu_1",
                                     "type": "tool_reference"}]}]}]}
    c3_headers = {"anthropic-beta": "mcp-client-2025-11-20,beta"}
    r, status, err, wall = anthropic(c3_body, extra_headers=c3_headers)
    out["C3_anthropic_nested_tool_reference"] = {"http_status": status, "wall": round(wall, 2),
                                                 "pass": status == 200,
                                                 "error": (err or {}).get("error") if status != 200 else None,
                                                 "blocks": [b.get("type") for b in (r or {}).get("content", [])] if r else None}

    # C4: /v1/responses namespaced function + function_call_output replay
    r, status, err, wall = post_path(base, "/v1/responses", _responses_body(), sid("compat-resp"))
    out["C4_responses_namespaced_tool"] = {"http_status": status, "wall": round(wall, 2),
                                           "pass": status == 200,
                                           "error": (err or {}).get("error") if status != 200 else None,
                                           "keys": sorted((r or {}).keys())[:12] if r else None}
    # replay: function_call_output for the model's call, if any
    replay = None
    if status == 200 and isinstance(r, dict):
        fc = None
        for item in r.get("output") or r.get("output_items") or []:
            if isinstance(item, dict) and item.get("type") == "function_call":
                fc = item
                break
        if fc:
            body2 = {"model": MODEL, "input": [
                {"type": "function_call_output", "call_id": fc.get("call_id") or fc.get("id"),
                 "output": "{\"value\": 4}"}],
                "tools": [{"type": "function", "name": "math.add", "description": "Add two numbers",
                           "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                                          "required": ["a", "b"]}}]}
            r2, status2, err2, wall2 = post_path(base, "/v1/responses", body2, sid("compat-resp-replay"))
            replay = {"http_status": status2, "pass": status2 == 200,
                      "error": (err2 or {}).get("error") if status2 != 200 else None}
        else:
            replay = {"http_status": None, "pass": None, "note": "no function_call in first response; replay skipped"}
    out["C4_replay"] = replay

    # C5: Responses API reasoning.effort low, record reasoning token count vs medium
    def responses_reasoning(effort):
        r, status, err, wall = post_path(base, "/v1/responses", _responses_body(effort), sid(f"compat-resp-{effort}"))
        if status != 200:
            return {"http_status": status, "reasoning_tokens": None, "error": (err or {}).get("error")}
        # reasoning tokens: look across usage and output text_token counts
        usage = (r or {}).get("usage") or {}
        rtok = ((usage.get("output_tokens_details") or {}).get("reasoning_tokens"))
        if rtok is None:
            acc = 0
            for item in (r or {}).get("output") or []:
                if isinstance(item, dict) and item.get("type") in ("reasoning", "thought"):
                    txt = item.get("text") or json.dumps(item.get("summary") or "")
                    acc += len(txt)
            rtok = acc if acc else None
            rtok = {"raw_chars": rtok} if rtok else None
        return {"http_status": status, "reasoning_tokens": rtok, "pass": True,
                "keys": sorted((r or {}).keys())[:12]}
    c5_low = responses_reasoning("low")
    c5_med = responses_reasoning("medium")
    out["C5_responses_reasoning_low_vs_medium"] = {
        "low": c5_low, "medium": c5_med,
        "pass": c5_low.get("http_status") == 200 and c5_med.get("http_status") == 200,
    }

    # C6: OpenAI reasoning_effort medium + temperature 0
    r, status, err, wall = post_path(base, "/v1/chat/completions",
                                     {"model": MODEL, "temperature": 0, "reasoning_effort": "medium",
                                      "max_tokens": 300,
                                      "messages": [{"role": "user", "content": "What is 6*7? Answer with just the number."}]},
                                     sid("compat-openai-medium"))
    out["C6_openai_reasoning_medium"] = {"http_status": status, "wall": round(wall, 2),
                                         "pass": status == 200,
                                         "error": (err or {}).get("error") if status != 200 else None,
                                         "answer": ((r or {}).get("choices", [{}])[0].get("message", {}).get("content") or "")[:40] if r else None}

    # C7: image -> "red"
    url = "data:image/png;base64," + base64.b64encode(_red_png()).decode()
    r, status, err, wall = post_path(base, "/v1/chat/completions",
                                     {"model": MODEL, "max_tokens": 300, "temperature": 0, "reasoning_effort": "low",
                                      "messages": [{"role": "user", "content": [
                                          {"type": "image_url", "image_url": {"url": url}},
                                          {"type": "text", "text": "What colour is this image? Reply with one word."}]}]},
                                     sid("compat-vision"))
    txt = ((r or {}).get("choices", [{}])[0].get("message", {}).get("content") or "") if r else ""
    out["C7_image_png_red"] = {"http_status": status,
                               "pass": status == 200 and "red" in txt.lower(),
                               "answer": txt[:40],
                               "error": (err or {}).get("error") if status != 200 else None,
                               "wall": round(wall, 2)}

    # C8: corrupt image data URL -> expect 4xx, not 5xx
    r, status, err, wall = post_path(base, "/v1/chat/completions",
                                     {"model": MODEL, "max_tokens": 32, "temperature": 0,
                                      "messages": [{"role": "user", "content": [
                                          {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg"}},
                                          {"type": "text", "text": "What colour is this image?"}]}]},
                                     sid("compat-vision-corrupt"))
    out["C8_corrupt_image"] = {"http_status": status,
                               "pass": status is not None and 400 <= status < 500,
                               "error": (err or {}).get("error") if status is None else None,
                               "wall": round(wall, 2)}

    # replay count for C4: combined pass
    out["pass_count"] = sum(1 for k, v in out.items() if isinstance(v, dict) and v.get("pass") is True)
    out["pass_total"] = sum(1 for k, v in out.items() if isinstance(v, dict) and "pass" in v and v.get("pass") is not None)
    return out


# ------------------------------------------------------------- 7. info

def run_info(base, args):
    out = {}
    try:
        raw, status = get_json(base, "/get_server_info", timeout=60)
        try:
            info = json.loads(raw)
        except json.JSONDecodeError:
            info = {"_raw": raw[:3000]}
        out["get_server_info"] = {"http_status": status, "raw_excerpt": raw[:3000]}
        # pull the load-bearing fields out
        def find(obj, key, depth=0):
            if depth > 6:
                return None
            if isinstance(obj, dict):
                if key in obj:
                    return obj[key]
                for v in obj.values():
                    r = find(v, key, depth + 1)
                    if r is not None:
                        return r
            elif isinstance(obj, list):
                for v in obj:
                    r = find(v, key, depth + 1)
                    if r is not None:
                        return r
            return None
        out["get_server_info"]["fields"] = {
            "max_running_requests": find(info, "max_running_requests"),
            "mamba_cache_size": find(info, "max_mamba_cache_size"),
            "max_total_num_tokens": find(info, "max_total_num_tokens"),
            "max_total_tokens": find(info, "max_total_tokens"),
            "version": find(info, "version") or find(info, "git_version") or find(info, "sglang_version"),
        }
    except Exception as e:  # noqa: BLE001
        out["get_server_info"] = _err_dict(e)
    try:
        raw, status = get_json(base, "/health", timeout=60)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        out["health"] = {"http_status": status, "status_ok": status == 200,
                         "raw_excerpt": (raw[:500] if isinstance(parsed, dict) is False else json.dumps(parsed)[:500])}
    except Exception as e:  # noqa: BLE001
        out["health"] = _err_dict(e)
    return out


TESTS = {"speed": run_speed, "warm-ttft": run_warm_ttft, "prefill-cached": run_prefill_cached,
         "hicache": run_hicache, "tools": run_tools, "compat": run_compat, "info": run_info}


# ------------------------------------------------------------- summaries

def summarize_line(name, r):
    if not isinstance(r, dict):
        return f"{name:<14} {json.dumps(r)[:140]}"
    if "error" in r and len(r) <= 2:
        return f"{name:<14} ERROR {r['error'][:100]}"
    if name == "speed":
        parts = []
        for kind in ("code", "prose"):
            a = (r.get(kind, {}).get("one_stream") or {}).get("median_decode_tps")
            t = (r.get(kind, {}).get("one_stream") or {}).get("median_ttft")
            c4 = (r.get(kind, {}).get("four_streams") or {}).get("aggregate_decode_tps")
            line = f"{kind} 1x={a} (ttft {t}) C4agg={c4}"
            c6 = (r.get(kind, {}).get("six_streams") or {}).get("aggregate_decode_tps")
            if c6 is not None:
                line += f" C6agg={c6}"
            parts.append(line)
        return f"{name:<14} " + " | ".join(parts) + f"  [{r.get('params', {}).get('reasoning', '?')}]"
    if name == "warm-ttft":
        cold = (r.get("cold") or {}).get("ttft")
        warm = (r.get("warm") or {}).get("ttft")
        wc = (r.get("warm") or {}).get("cached_tokens")
        return f"{name:<14} cold_ttft={cold} warm_ttft={warm} cached_tokens={wc}"
    if name == "prefill-cached":
        fresh = r.get("fresh_turn") or {}
        return (f"{name:<14} prefix={r.get('target_tokens')} new_tokens={r.get('new_tokens')} "
                f"fresh_prefill_tps={r.get('fresh_prefill_tps')} cached_delta={r.get('cached_delta')} "
                + (f"c4_agg={r.get('c4', {}).get('aggregate_fresh_prefill_tps')} " if r.get("c4") else ""))
    if name == "hicache":
        passes = r.get("n_pass")
        n = r.get("n")
        trials_str = ",".join(f"t{t['trial']}:{'PASS' if t['pass'] else 'FAIL'}" for t in (r.get("trials") or []))
        return f"{name:<14} {passes}/{n} return PASS [{trials_str}]"
    if name == "tools":
        e = r.get("edit_file_temp07") or {}
        m = r.get("multi_tool") or {}
        return (f"{name:<14} edit_file {e.get('valid_pct')}% of {e.get('n')} {e.get('counts')} | "
                f"multi both={m.get('both')} names={m.get('names')}")
    if name == "compat":
        items = [k for k in r.keys() if k.startswith("C") and isinstance(r.get(k), dict)]
        parts = [f"{k}:{'PASS' if r[k].get('pass') else ('FAIL' if r[k].get('pass') is False else '?')}"
                 for k in items]
        return f"{name:<14} " + " ".join(parts) + f" | {r.get('pass_count')}/{r.get('pass_total')}"
    if name == "info":
        f = ((r.get("get_server_info") or {}).get("fields")) or {}
        return (f"{name:<14} health={(r.get('health') or {}).get('status_ok')} "
                f"max_running={f.get('max_running_requests')} mamba={f.get('mamba_cache_size')} "
                f"max_total={f.get('max_total_num_tokens') or f.get('max_total_tokens')} v={f.get('version')}")
    return f"{name:<14} {json.dumps(r)[:140]}"


def headline_rows(r):
    rows = []

    def val(*keys):
        cur = r
        for k in keys:
            if not isinstance(cur, dict) or k not in cur or cur[k] is None:
                return None
            cur = cur[k]
        return cur

    for kind in ("code", "prose"):
        rows.append((f"speed {kind} C1 tok/s", val("speed", kind, "one_stream", "median_decode_tps")))
        rows.append((f"speed {kind} C4 agg tok/s", val("speed", kind, "four_streams", "aggregate_decode_tps")))
        rows.append((f"speed {kind} C1 TTFT s", val("speed", kind, "one_stream", "median_ttft")))
        c6 = val("speed", kind, "six_streams", "aggregate_decode_tps")
        if c6 is not None:
            rows.append((f"speed {kind} C6 agg tok/s", c6))
    rows.append(("speed C6 agg tok/s (code)", val("speed", "code", "six_streams", "aggregate_decode_tps")))
    rows.append(("speed C6 agg tok/s (prose)", val("speed", "prose", "six_streams", "aggregate_decode_tps")))
    rows.append(("warm-ttft cold s", val("warm-ttft", "cold", "ttft")))
    rows.append(("warm-ttft warm s", val("warm-ttft", "warm", "ttft")))
    rows.append(("warm-ttft warm cached tokens", val("warm-ttft", "warm", "cached_tokens")))
    rows.append(("prefill-cached fresh tok/s", val("prefill-cached", "fresh_prefill_tps")))
    rows.append(("prefill-cached new tokens", val("prefill-cached", "new_tokens")))
    rows.append(("prefill-cached C4 agg fresh tok/s", val("prefill-cached", "c4", "aggregate_fresh_prefill_tps")))
    rows.append(("hicache return pass", val("hicache", "pass_rate")))
    rows.append(("hicache n_pass/n", val("hicache", "n_pass")))
    rows.append(("tools edit_file valid %", val("tools", "edit_file_temp07", "valid_pct")))
    rows.append(("tools multi both", val("tools", "multi_tool", "both")))
    rows.append(("compat pass", val("compat", "pass_count")))
    rows.append(("compat total", val("compat", "pass_total")))
    rows.append(("total wall seconds", val("_wall")))
    return [(k, v) for k, v in rows if v is not None]


def do_compare(a_path, b_path):
    a, b = (json.load(open(p)) for p in (a_path, b_path))
    ra, rb = a.get("results", {}), b.get("results", {})
    ra["_wall"] = a.get("wall_seconds")
    rb["_wall"] = b.get("wall_seconds")
    la, lb = a.get("label", a_path), b.get("label", b_path)
    rows_a, rows_b = dict(headline_rows(ra)), dict(headline_rows(rb))
    keys = list(dict.fromkeys(list(rows_a) + list(rows_b)))
    print(f"{'metric':<40} {la[:24]:>24} {lb[:24]:>24} {'delta B-A':>14} {'%':>8}")
    print("-" * 116)
    for k in keys:
        va, vb = rows_a.get(k), rows_b.get(k)
        delta = pct = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            delta = f"{vb - va:+.2f}" if abs(vb - va) < 10 else f"{vb - va:+.1f}"
            if va:
                pct = f"{100.0 * (vb - va) / va:+.1f}"
        print(f"{k:<40} {(fmt(va) if va is not None else '-'):>24} "
              f"{(fmt(vb) if vb is not None else '-'):>24} {delta:>14} {pct:>8}")


def fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pennyroyal v2.5.3 -> v3.1.7 evaluation harness")
    ap.add_argument("--base", default="http://127.0.0.1:18200")
    ap.add_argument("--suite", choices=["quick", "full"], default="quick")
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument("--label", default=None)
    ap.add_argument("--only", help="comma list: " + ",".join(TESTS))
    ap.add_argument("--prefix-tokens", type=int, default=200000, help="prefill-cached prefix size")
    ap.add_argument("--hicache-trials", type=int, default=3)
    ap.add_argument("--hicache-doc-tokens", type=int, default=200000)
    ap.add_argument("--flush-docs", type=int, default=5)
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

    print(f"penny_eval label={LABEL} base={args.base} suite={args.suite} tests={','.join(names)}", flush=True)
    started = time.time()
    results = {}
    for name in names:
        t0 = time.time()
        try:
            r = TESTS[name](args.base, args)
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{e.__class__.__name__}: {str(e)[:300]}"}
        results[name] = r
        print(f"{summarize_line(name, r)}  ({time.time() - t0:.1f}s)", flush=True)

    wall = time.time() - started
    results["_wall"] = round(wall, 1)
    out_doc = {
        "label": LABEL,
        "base": args.base,
        "suite": args.suite,
        "tests": names,
        "wall_seconds": round(wall, 1),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": {"prefix_tokens": args.prefix_tokens, "hicache_trials": args.hicache_trials,
                 "hicache_doc_tokens": args.hicache_doc_tokens, "flush_docs": args.flush_docs},
        "results": results,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out_doc, f, indent=2)
        print(f"wrote {args.out}", flush=True)

    print("\n=== summary ===", flush=True)
    rows = headline_rows(results)
    for k, v in rows:
        print(f"  {k:<40} {fmt(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
