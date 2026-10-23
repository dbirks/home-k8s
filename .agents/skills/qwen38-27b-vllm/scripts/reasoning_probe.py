import json, sys, time, urllib.request, urllib.error, base64, struct, zlib
BASE = sys.argv[1]
def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r: return json.load(r), time.time() - t
    except urllib.error.HTTPError as e: return {"_err": e.code, "_body": e.read().decode()[:200]}, time.time() - t
Q = "A farmer has 17 sheep. All but 9 run away. How many are left? Answer with just the number."
def chat(extra, label):
    body = {"model": "qwen3-8-27b", "max_tokens": 16000, "messages": [{"role": "user", "content": Q}], **extra}
    d, dt = post("/v1/chat/completions", body)
    if "_err" in d: print(f"{label:38s} HTTP {d['_err']}: {d['_body'][:110]}"); return
    m = d["choices"][0]["message"]; r = m.get("reasoning_content") or m.get("reasoning") or ""
    print(f"{label:38s} think_chars={len(r):6d} completion_tok={d['usage']['completion_tokens']:5d} answer={(m.get('content') or '').strip()[:20]!r} {dt:.1f}s")
chat({}, "default (no effort)")
for e in ["low", "medium", "xhigh", "high", "none", "minimal"]:
    chat({"reasoning_effort": e}, f"reasoning_effort={e}")
chat({"chat_template_kwargs": {"enable_thinking": False}}, "enable_thinking=false")
chat({"chat_template_kwargs": {"reasoning_effort": "xhigh"}}, "chat_template_kwargs effort=xhigh")
chat({"chat_template_kwargs": {"thinking_enabled": False}}, "thinking_enabled=false (gateway UI key)")
for th, label in [({"type": "disabled"}, "anthropic thinking disabled"), ({"type": "enabled", "budget_tokens": 4000}, "anthropic thinking enabled"), (None, "anthropic no thinking field")]:
    body = {"model": "qwen3-8-27b", "max_tokens": 16000, "messages": [{"role": "user", "content": Q}]}
    if th: body["thinking"] = th
    d, dt = post("/v1/messages", body)
    if "_err" in d: print(f"{label:38s} HTTP {d['_err']}: {d['_body'][:110]}"); continue
    th_chars = sum(len(c.get("thinking", "")) for c in d["content"] if c["type"] == "thinking")
    print(f"{label:38s} think_chars={th_chars:6d} types={[c['type'] for c in d['content']]} {dt:.1f}s")
# image: 64x64 PNG, left half red, right half blue
def png(w, h):
    raw = b"".join(b"\x00" + b"".join((b"\xff\x00\x00" if x < w // 2 else b"\x00\x00\xff") for x in range(w)) for _ in range(h))
    ch = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b"")
img = "data:image/png;base64," + base64.b64encode(png(64, 64)).decode()
d, dt = post("/v1/chat/completions", {"model": "qwen3-8-27b", "max_tokens": 4000, "chat_template_kwargs": {"enable_thinking": False},
    "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}}, {"type": "text", "text": "What two colors are in this image, left then right? One line."}]}]})
print("IMAGE", d.get("_err"), d.get("_body", "")[:150] if "_err" in d else d["choices"][0]["message"].get("content"), f"{dt:.1f}s")
