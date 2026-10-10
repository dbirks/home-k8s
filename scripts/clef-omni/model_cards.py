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
    out = ["| Model | Format | Size on disk | VRAM (weights) | Agrees with BF16 (top answer) | Decision Index 0.2.1 | Best for |",
           "|---|---|---|---|---|---|---|"]
    for q in rows:
        link = f"[{q['repo']}](https://huggingface.co/{q['repo']})"
        agree = "reference" if q["id"] == "bf16" else fmt(q["parity"].get("top1"), "{:.1%}")
        out.append(f"| {link} | {q['format']} | {fmt(q['size_gb'], '{:.1f} GB')} | {fmt(q.get('vram_gib'), '{:.0f} GiB')} "
                   f"| {agree} | {fmt(q['bench'].get('index'), '{:.2f}')} | {q['best_for']} |")
    return "\n".join(out)


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
