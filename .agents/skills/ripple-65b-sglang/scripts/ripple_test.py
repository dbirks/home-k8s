"""Correctness probes for an OpenAI-compatible Ripple endpoint (stdlib only).

usage: python3 -I ripple_test.py BASE_URL [MODEL] [API_KEY]
  BASE_URL e.g. http://localhost:18000 (port-forward to svc/ripple-65b) or the Collective base.
Checks: chat, streaming (reasoning_content vs content deltas), reasoning separation, a two-turn tool
round trip with structured tool_call ids, tool_choice=required with a strict tool, /v1/messages if served.
"""
import json, sys, time, urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "ripple-65b"
KEY = sys.argv[3] if len(sys.argv) > 3 else None
HDR = {"Content-Type": "application/json", **({"Authorization": "Bearer " + KEY} if KEY else {})}
OK = []


def post(path, body, stream=False):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), HDR)
    r = urllib.request.urlopen(req, timeout=900)
    return r if stream else json.load(r)


def check(name, cond, info=""):
    OK.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, ("- " + str(info)[:300]) if info else "")


ADD = {"type": "function", "function": {
    "name": "add_numbers", "description": "Add two integers.", "strict": True,
    "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                   "required": ["a", "b"], "additionalProperties": False}}}

# 1. plain chat + reasoning separation
t = time.time()
d = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 4096, "temperature": 0,
                                   "messages": [{"role": "user", "content": "What is 37 * 43? Reply with only the number."}]})
m = d["choices"][0]["message"]
check("chat answer 1591", "1591" in (m.get("content") or ""), m.get("content"))
check("reasoning separated (no <think> in content)", "<think>" not in (m.get("content") or "") and "</think>" not in (m.get("content") or ""),
      f"reasoning_len={len(m.get('reasoning_content') or '')}")
print(f"     usage={d.get('usage')} {time.time()-t:.1f}s")

# 2. streaming
t = time.time(); first = None; rc = c = 0; text = ""
r = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 4096, "temperature": 0, "stream": True,
                                   "messages": [{"role": "user", "content": "Name three primary colors, comma separated."}]}, stream=True)
for line in r:
    line = line.decode().strip()
    if not line.startswith("data:") or line == "data: [DONE]":
        continue
    ch = json.loads(line[5:])
    if not ch.get("choices"):
        continue
    dl = ch["choices"][0].get("delta", {})
    if dl.get("reasoning_content"):
        rc += 1; first = first or time.time()
    if dl.get("content"):
        c += 1; text += dl["content"]; first = first or time.time()
check("stream deltas (reasoning + content)", c > 0, f"reasoning_deltas={rc} content_deltas={c} ttft={(first or t)-t:.2f}s text={text!r}")

# 3. tool round trip
msgs = [{"role": "user", "content": "Use add_numbers to calculate 23 + 19."}]
d = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 4096, "temperature": 0, "messages": msgs,
                                   "tools": [ADD], "tool_choice": "auto"})
m = d["choices"][0]["message"]; tc = m.get("tool_calls") or []
args = json.loads(tc[0]["function"]["arguments"]) if tc else {}
check("tool call structured", tc and tc[0]["function"]["name"] == "add_numbers" and args == {"a": 23, "b": 19} and tc[0].get("id"),
      tc)
if tc:
    msgs += [m, {"role": "tool", "tool_call_id": tc[0]["id"], "content": "42"}]
    d = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 4096, "temperature": 0, "messages": msgs, "tools": [ADD]})
    m2 = d["choices"][0]["message"]
    check("tool result used in answer", "42" in (m2.get("content") or ""), m2.get("content"))

# 4. tool_choice required (strict)
d = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 4096, "temperature": 0,
                                   "messages": [{"role": "user", "content": "Add 2 and 5."}], "tools": [ADD], "tool_choice": "required"})
tc = d["choices"][0]["message"].get("tool_calls") or []
check("tool_choice=required", tc and json.loads(tc[0]["function"]["arguments"]) == {"a": 2, "b": 5}, tc)

# 5. Anthropic messages door (SGLang serves /v1/messages)
try:
    d = post("/v1/messages", {"model": MODEL, "max_tokens": 2048, "messages": [{"role": "user", "content": "Say hi in one word."}]})
    check("/v1/messages", any(b.get("type") == "text" for b in d.get("content", [])), [b.get("type") for b in d.get("content", [])])
except Exception as e:  # noqa: BLE001
    check("/v1/messages", False, e)

print(f"{sum(OK)}/{len(OK)} passed")
sys.exit(0 if all(OK) else 1)
