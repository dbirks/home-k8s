"""KV-dtype A/B probe (stdlib only): run the same fixed, temperature-0 workload against an endpoint and save it.

usage: python3 -I kv_ab.py BASE_URL OUT.json [MODEL] [NEEDLE_TOKENS e.g. 32000,60000]
       python3 -I kv_ab.py --compare A.json B.json
Records: math answers (thinking allowed), executed-code results, a tool call, needle recall (5 codes) at the
given prompt sizes, and the full temp-0 texts so two KV dtypes can be diffed for silent corruption.
"""
import json, random, re, subprocess, sys, tempfile, time, urllib.request

if sys.argv[1] == "--compare":
    a, b = (json.load(open(f)) for f in sys.argv[2:4])
    for k in a:
        va, vb = a[k], b.get(k)
        same = va.get("text") == (vb or {}).get("text")
        print(f"{k:14s} A={va.get('ok')!s:5s} B={(vb or {}).get('ok')!s:5s} identical_text={same}  {va.get('info','')[:40]!r} | {(vb or {}).get('info','')[:40]!r}")
    sys.exit(0)

BASE, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else "ripple-65b"
NEEDLES = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "32000,60000").split(",")]


def chat(messages, max_tokens=6000, **kw):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0, **kw}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return m, d.get("usage", {}), time.time() - t


R = {}
MATH = [("A train leaves at 09:40 and arrives at 13:15. How many minutes is the trip? Think it through, then give the number.", "215"),
        ("If x + y = 10 and x - y = 4, what is x*y? Think it through, then give the number.", "21"),
        ("What is the remainder when 7^100 is divided by 13? Think it through, then give the number.", "9"),
        ("How many primes are there below 50? Think it through, then give the number.", "15")]
for i, (q, want) in enumerate(MATH):
    m, u, dt = chat([{"role": "user", "content": q}])
    c = m.get("content") or ""
    R[f"math{i}"] = {"ok": re.findall(r"-?\d+", c)[-1:] == [want], "info": c.strip()[-40:], "text": (m.get("reasoning_content") or "") + "|" + c}

CODE = ("Write a Python function `lcs_len(a, b)` returning the length of the longest common subsequence of strings a and b. "
        "Reply with only one python code block.", "assert lcs_len('ABCBDAB','BDCABA')==4\nassert lcs_len('','x')==0\nassert lcs_len('abc','abc')==3\nprint('ok')")
m, u, dt = chat([{"role": "user", "content": CODE[0]}])
blk = re.findall(r"```(?:python)?\n(.*?)```", m.get("content") or "", re.S)
ok = False
if blk:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(blk[0] + "\n" + CODE[1])
    ok = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=30).stdout.strip() == "ok"
R["code"] = {"ok": ok, "info": "ran" if blk else "no block", "text": m.get("content")}

ADD = {"type": "function", "function": {"name": "add_numbers", "description": "Add two integers.", "strict": True,
       "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                      "required": ["a", "b"], "additionalProperties": False}}}
m, u, dt = chat([{"role": "user", "content": "Use add_numbers to calculate 1234 + 4321."}], tools=[ADD], tool_choice="auto")
tc = m.get("tool_calls") or []
R["tool"] = {"ok": bool(tc) and json.loads(tc[0]["function"]["arguments"]) == {"a": 1234, "b": 4321},
             "info": str(tc[:1])[:80], "text": json.dumps([t["function"] for t in tc])}

for n in NEEDLES:
    rnd = random.Random(n)
    facts = {f"agent-{rnd.randint(100, 999)}": f"{rnd.choice(['RED', 'BLUE', 'GOLD', 'JADE', 'ONYX'])}-{rnd.randint(1000, 9999)}" for _ in range(5)}
    lines = [f"log line {i}: routine status ok, nothing to report, value {i % 89}." for i in range(n // 18)]
    for j, (a, code) in enumerate(facts.items()):
        lines.insert((j + 1) * len(lines) // 6, f"IMPORTANT: the access code for {a} is {code}.")
    q = "\n".join(lines) + "\n\nList the access code for each of these agents, one per line as 'agent: code': " + ", ".join(facts)
    m, u, dt = chat([{"role": "user", "content": q}], max_tokens=3000)
    c = m.get("content") or ""
    got = sum(code in c for code in facts.values())
    R[f"needle{n // 1000}k"] = {"ok": got == 5, "info": f"{got}/5 prompt={u.get('prompt_tokens')} {dt:.0f}s", "text": c}

for k, v in R.items():
    print(f"{k:12s} {'PASS' if v['ok'] else 'FAIL'}  {v['info']}")
print(f"TOTAL {sum(v['ok'] for v in R.values())}/{len(R)}")
json.dump(R, open(OUT, "w"), indent=1)
