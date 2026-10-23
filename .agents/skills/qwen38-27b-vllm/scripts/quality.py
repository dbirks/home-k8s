import json, re, sys, time, urllib.request, concurrent.futures as cf, random, subprocess, tempfile, os
BASE = sys.argv[1]
def chat(msg, max_tokens=8192):
    body = {"model": "qwen3-8-27b", "max_tokens": max_tokens, "messages": [{"role": "user", "content": msg}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return (d["choices"][0]["message"].get("content") or "").strip(), d["choices"][0]["finish_reason"]

MATH = [
    ("What is 37 * 43? Reply with only the number.", "1591"),
    ("What is the sum of the first 100 positive integers? Reply with only the number.", "5050"),
    ("How many primes are there below 50? Reply with only the number.", "15"),
    ("What is 2^20? Reply with only the number.", "1048576"),
    ("A train leaves at 09:40 and arrives at 13:15. How many minutes is the trip? Reply with only the number.", "215"),
    ("What is the remainder when 7^100 is divided by 13? Reply with only the number.", "9"),
    ("If x + y = 10 and x - y = 4, what is x*y? Reply with only the number.", "21"),
    ("How many letters 'r' are in the word 'strawberry'? Reply with only the number.", "3"),
]
CODE = [
    ("Write a Python function `merge_intervals(iv)` that merges overlapping closed intervals given as a list of [a,b] lists and returns them sorted. Reply with only one python code block.",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]\nassert merge_intervals([[5,6],[1,2]])==[[1,2],[5,6]]"),
    ("Write a Python function `is_balanced(s)` that returns True if the brackets ()[]{} in s are balanced (ignore other characters). Reply with only one python code block.",
     "assert is_balanced('a(b)[c]{d}')\nassert not is_balanced('(]')\nassert not is_balanced('((')\nassert is_balanced('')\nassert is_balanced('{[()()]}')"),
    ("Write a Python function `lcs_len(a, b)` returning the length of the longest common subsequence of strings a and b. Reply with only one python code block.",
     "assert lcs_len('ABCBDAB','BDCABA')==4\nassert lcs_len('','x')==0\nassert lcs_len('abc','abc')==3"),
    ("Write a Python function `roman(n)` converting an int 1..3999 to a Roman numeral string. Reply with only one python code block.",
     "assert roman(1994)=='MCMXCIV'\nassert roman(3999)=='MMMCMXCIX'\nassert roman(4)=='IV'\nassert roman(58)=='LVIII'"),
]
def math_case(c):
    out, fin = chat(c[0]); nums = re.findall(r"-?\d+", out.replace(",", ""))
    return ("math", c[1] in nums[-1:] , out[:40], fin)
def code_case(c):
    out, fin = chat(c[0]); m = re.findall(r"```(?:python)?\n(.*?)```", out, re.S)
    if not m: return ("code", False, "no code block", fin)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(m[0] + "\n" + c[1] + "\nprint('PASS')\n")
    r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=30); os.unlink(f.name)
    return ("code", "PASS" in r.stdout, (r.stderr.strip().splitlines() or ["ok"])[-1][:60], fin)
def needles(n_words, k=5):
    rnd = random.Random(n_words)
    facts = {f"agent-{rnd.randint(100,999)}": f"{rnd.choice(['RED','BLUE','GOLD','JADE','ONYX'])}-{rnd.randint(1000,9999)}" for _ in range(k)}
    lines = [f"log line {i}: routine status ok, nothing to report, value {i % 89}." for i in range(n_words // 10)]
    for j, (a, code) in enumerate(facts.items()):
        lines.insert((j + 1) * len(lines) // (k + 1), f"IMPORTANT: the access code for {a} is {code}.")
    q = "\n".join(lines) + "\n\nList the access code for each of these agents, one per line as 'agent: code': " + ", ".join(facts)
    out, fin = chat(q)
    got = sum(code in out for code in facts.values())
    return ("needles@%dw" % (n_words // 1000), got == k, f"{got}/{k} codes", fin)
t = time.time()
with cf.ThreadPoolExecutor(12) as ex:
    res = list(ex.map(math_case, MATH)) + list(ex.map(code_case, CODE))
res += [needles(20000), needles(60000), needles(90000)]
for r in res: print("%-12s %-5s %-45s %s" % (r[0], "PASS" if r[1] else "FAIL", r[2].replace("\n", " "), r[3]))
print("TOTAL %d/%d pass in %.0fs" % (sum(r[1] for r in res), len(res), time.time() - t))
