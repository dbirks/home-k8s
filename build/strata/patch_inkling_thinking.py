"""Turn Inkling 2 Mini's thinking levels back on in Strata's serve layer (Strata 0.2.19 forces reasoning_effort "none").

The model's own chat_template.jinja reads reasoning_effort none|minimal|low|medium|high|xhigh|max (0.0..0.99) and
the model emits <|content_thinking|> spans; upstream discards them and pins effort to "none". This keeps the
request's level (OpenAI reasoning_effort / chat_template_kwargs, Anthropic thinking / output_config), defaults to
$STRATA_INKLING_EFFORT (default "medium", like our vLLM and Basalt configs), and streams thinking as reasoning_content.
Every replacement is asserted, so an upstream change fails the image build instead of silently doing nothing.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/strata/terminal")


def sub(path, old, new, count=1):
    p = root / path
    s = p.read_text(encoding="utf-8")
    n = s.count(old)
    assert n == count, f"{path}: expected {count} match(es) of {old[:60]!r}, found {n}"
    p.write_text(s.replace(old, new), encoding="utf-8")


D = "serve/inkling_dialect.py"
sub(D, """            # Inkling-2-Mini has no thinking mode (off / "none" only): silently discard any raw <|content_thinking|> span.
            return
""", """            out.append(("reasoning", chunk))      # home-k8s patch: thinking is streamed as reasoning
            return
""")
sub(D, '''DEFAULT_EFFORT = "none"
EFFORT = {
    "none": "none", "off": "none", "no_think": "none", "disabled": "none", "false": "none", "0": "none", "0.0": "none",
}
''', '''import os as _os
EFFORT = {
    "none": "none", "off": "none", "no_think": "none", "disabled": "none", "false": "none", "0": "none", "0.0": "none",
    "minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max",
    "on": "medium", "true": "medium", "enabled": "medium",
}
DEFAULT_EFFORT = EFFORT.get(_os.environ.get("STRATA_INKLING_EFFORT", "medium").strip().lower(), "medium")
''')
sub(D, '''    """Inkling-2-Mini has no thinking mode: always normalizes to `"none"` (`Thinking effort level: 0`)."""
    return DEFAULT_EFFORT
''', '''    """home-k8s patch: a known level (or a number 0..0.99) is kept; anything else is the default level."""
    if value is None or value == "":
        return DEFAULT_EFFORT
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return min(max(float(value), 0.0), 0.99)
    return EFFORT.get(str(value).strip().lower(), DEFAULT_EFFORT)
''')
sub(D, '''    out = dict(kwargs)
    out.pop("thinking_mode", None)
    out["enable_thinking"] = False
    out["reasoning_effort"] = DEFAULT_EFFORT
    return out
''', '''    out = dict(kwargs)
    out.pop("thinking_mode", None)
    off = out.pop("enable_thinking", None) is False
    out["reasoning_effort"] = "none" if off else normalize_effort(out.get("reasoning_effort"))
    return out
''')

S = "serve/server.py"
sub(S, '''        if self.inkling:
            return None
        return self.shared.get("reasoning_effort") or self.config_effort
''', '''        if self.inkling:                               # home-k8s patch: the template default (STRATA_INKLING_EFFORT)
            return self.shared.get("reasoning_effort") or self.config_effort or inkling_dialect.DEFAULT_EFFORT
        return self.shared.get("reasoning_effort") or self.config_effort
''')
print("patched", root)
