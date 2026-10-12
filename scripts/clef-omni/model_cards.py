"""Render the Hugging Face model card for every Clef-Omni quant from one data file (issue #150).

    python3 scripts/clef-omni/model_cards.py            # writes cards/<name>.md next to quants.json

Every card opens with the same "which quant should I use" comparison table, so the family stays
consistent. A PUBLIC card lists only public repos (links to private repos would 404); private cards
list everything. Numbers live in quants.json and are filled in as runs finish.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "quants.json").read_text())
SCRIPT_URL = "https://github.com/dbirks/home-k8s/blob/main/scripts/clef-omni/serve_clef.py"


def fmt(v, spec="{}", missing="pending"):
    return missing if v is None else spec.format(v)


def table(rows):
    out = ["| Model | Format | Size on disk | VRAM (weights) | Agrees with BF16 (top answer) | Decision Index 0.2.1 vs BF16 | Best for |",
           "|---|---|---|---|---|---|---|"]
    for q in rows:
        link = f"[{q['repo']}](https://huggingface.co/{q['repo']})"
        agree = "reference" if q["id"] == "bf16" else fmt(q["parity"].get("top1"), "{:.1%}")
        b = q.get("bench", {})
        if q["id"] == "bf16" and b.get("mean") is not None:
            di = f"reference (avg {b['mean']:.1f}; Cloudflare's table {b['mean_cf']:.1f})"
        elif b.get("mean_delta") is not None:
            di = f"{b['mean_delta']:+.2f} pts avg ({b['within2']} within 2 pts)"
        else:
            di = "pending"
        out.append(f"| {link} | {q['format']} | {fmt(q['size_gb'], '{:.1f} GB')} | {fmt(q.get('vram_gib'), '{:.0f} GiB')} "
                   f"| {agree} | {di} | {q['best_for']} |")
    return "\n".join(out)


def bench_section(q, all_q):
    """Per-benchmark Decision Index table: Cloudflare's published BF16, our BF16, this quant, paired delta."""
    b = q.get("bench", {})
    if not b.get("summary_file"):
        return []
    def load(f):
        d = json.loads((HERE / f).read_text()); d = d.get("benchmarks", d)
        return {x["dataset"]: x for x in (d.values() if isinstance(d, dict) else d)}
    base = next(x for x in all_q if x["id"] == "bf16")["bench"]
    B, Q, P = load(base["summary_file"]), load(b["summary_file"]), load(b["paired_file"])
    cf = DATA["cloudflare_published"]
    def cf_for(name):
        w = name.split()[0].lower().rstrip(":")
        return next((v for k, v in cf.items() if k.split()[0].lower() == w), None)
    full = []
    if b.get("index") is not None and DATA.get("board_note"):
        full = ["### Full Decision Index", "", DATA["board_note"], "", "### Paired sample vs BF16", ""]
    out = ["## Benchmarks", ""] + full + [DATA["bench_note"], "",
           "| Benchmark (metric) | n | Cloudflare BF16 (published) | BF16 (ours) | This quant | Δ vs BF16 (same questions) |",
           "|---|---|---|---|---|---|"]
    for name in sorted(B, key=lambda n: B[n]["catalog_id"]):
        x, y, z = B[name], Q.get(name, {}), P.get(name, {})
        if x.get("score") is None or y.get("score") is None:
            continue
        lower = "brier" in x["metric"].lower()
        sc = 100   # Cloudflare reports Brier x100 too
        ref = z.get("reference_same_cases")
        delta = f"{(y['score'] - ref) * sc:+.1f}" if ref is not None else "n/a"
        c = cf_for(name)
        out.append(f"| {name} ({x['metric']}{', lower is better' if lower and 'lower' not in x['metric'] else ''}) | {x['scored_requests']} "
                   f"| {fmt(c, '{:.1f}', 'n/a')} | {x['score'] * sc:.1f} | {y['score'] * sc:.1f} | {delta} |")
    return out + [""]


def card(q, all_q):
    shown = [x for x in all_q if x["public"] or not q["public"]]
    fm = ["---", "license: apache-2.0", "base_model: Cloudflare/clef-omni", "base_model_relation: quantized",
          "tags:"] + [f"  - {t}" for t in q["tags"]] + ["---", ""]
    body = [f"# {q['repo'].split('/')[-1]}", "", q["summary"], ""]
    if q.get("status"):
        body += [f"> **Status:** {q['status']}", ""]
    body += ["## Which Clef-Omni quant should I use?", "", table(shown), ""]
    body += ["**Recommended for:**", ""] + [f"- {r}" for r in DATA["recommendations"] if r.split("**")[1] in
                                             {x["repo"].split("/")[-1] for x in shown} or "Cloudflare" in r] + [""]
    body += [DATA["table_notes"], ""]
    body += ["## Quick start", "", "Serve it with [`serve_clef.py`](serve_clef.py) (also in "
             f"[the repo it is maintained in]({SCRIPT_URL})), a single-file [PEP 723](https://peps.python.org/pep-0723/) "
             "script, so `uv` installs everything it needs:", "", "```bash",
             f"# uv runs a PEP 723 script straight from its URL (or download serve_clef.py and `uv run` it locally)",
             f"uv run https://huggingface.co/{q['repo']}/resolve/main/serve_clef.py --model {q['repo']} --demo",
             f"uv run https://huggingface.co/{q['repo']}/resolve/main/serve_clef.py --model {q['repo']} --port 8000",
             "```", "", "The first command answers one built-in decision and exits; the second starts the SystemOne "
             "HTTP server (`GET /healthz` turns 200 once the model is loaded):", "", "```bash",
             "curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{",
             '  "model": "clef-omni",',
             '  "state": {"observation": "The player is facing a wall."},',
             '  "questions": {"action": {"type": "choice", "criteria": {"left": "Turn left", "right": "Turn right", "forward": "Move forward"}}}',
             "}'", "```", "",
             "Answers come from Cloudflare's own `systemone()` in `joint_schema_model.py`: one forward pass, "
             "per-option probabilities from the original BF16 joint decision head. Inputs can include images, "
             "audio and video (see the [base model card](https://huggingface.co/Cloudflare/clef-omni)).", ""]
    body += ["## How this checkpoint runs", ""] + q["serving"] + [""]
    body += ["## What was quantized", ""] + q["quantized"] + [""]
    body += ["## Recipe", ""] + q["recipe"] + [""]
    body += ["## Evaluation", "", DATA["eval_method"], ""] + q.get("eval_extra", []) + [""]
    body += bench_section(q, all_q)
    if q.get("versions"):
        body += ["## Version history", "", "Each version is an annotated git tag on this repo (`revision=\"v0.1\"` etc. "
                 "pins it). Newest last.", "", "| Version | Date | Notes |", "|---|---|---|"]
        body += [f"| `{v['tag']}` | {v['date']} | {v['note']} |" for v in q["versions"]] + [""]
    body += ["## License", "", "Apache-2.0, as the base model. All credit for the model goes to Cloudflare; "
             "this repo only changes the weight precision.", ""]
    return "\n".join(fm + body)


def main():
    out = HERE / "cards"
    out.mkdir(exist_ok=True)
    for q in DATA["quants"]:
        if q["id"] == "bf16":
            continue
        (out / f"{q['repo'].split('/')[-1]}.md").write_text(card(q, DATA["quants"]))
        print("wrote", out / f"{q['repo'].split('/')[-1]}.md")


if __name__ == "__main__":
    main()
