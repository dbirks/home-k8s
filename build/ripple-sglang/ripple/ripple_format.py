"""Ripple's chat format for SGLang: reasoning parser, tool-call parser and tokenizer check.

The model generates the layout chat_template.jinja renders:

    <think>\\n{reasoning}\\n</think>\\n{content}
    <think>\\n{reasoning}\\n</think>\\n\\n<tool_call>\\n{"name": "f", "arguments": {...}}\\n</tool_call>

`--reasoning-parser ripple` returns reasoning and content without the separator newlines the
template adds back, so an assistant message sent back unchanged re-renders to the tokens the
model generated (SGLang's deepseek-v3 parser keeps them, doubling each newline on every turn).
Output counts as reasoning only when it opens with <think>.

`--tool-call-parser ripple` is hermes with the template's call layout, so strict tools and
tool_choice=required constrain decoding to `<tool_call>\\n{"name": "f", "arguments": ...}`
(hermes forces `<tool_call>{"name":"f"`, which the model was not trained on). Text before the
first call keeps its whitespace, minus the newline the template puts before <tool_call>.

Output the template cannot render (`</think>\\n<tool_call>` with one newline, text after a call)
re-renders in the template's canonical layout.

`register()` adds both parsers and checks every tokenizer SGLang loads against the checkpoint's
tokenizer.json (Transformers 5 can rebuild the pre-tokenizer from `tokenizer_class`).
"""

import json
import logging

THINK_START, THINK_END = "<think>", "</think>"

# Exercise pre-tokenizer differences: digit runs, contractions, whitespace runs, non-ASCII.
TOKENIZER_PROBES = (
    "x = 1234567890 + 3.14159; y=-42",
    "I'M here, you'RE there, it's 4:05pm. DON'T",
    "a  b   c\n\n\n  d\t\te \r\n f",
    "    def f(self):\n        return {'k': [1, 2]}\n",
    "café naïve 日本語テキスト 한국어 🙂👍 Ελληνικά",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nok\n</think>\nHello<|im_end|>",
)


def _drop_newline(text, lead):
    if lead:
        return text[1:] if text.startswith("\n") else text
    return text[:-1] if text.endswith("\n") else text


def _partial_suffix(text, token):
    """Length of the longest suffix of text that is a strict prefix of token."""
    for i in range(min(len(text), len(token) - 1), 0, -1):
        if token.startswith(text[-i:]):
            return i
    return 0


def _reasoning_detector_class():
    from sglang.srt.parser.reasoning_parser import (
        BaseReasoningFormatDetector,
        StreamingParseResult,
    )

    class RippleReasoningDetector(BaseReasoningFormatDetector):
        def __init__(self, stream_reasoning=True, force_reasoning=False, **kwargs):
            # force_reasoning is ignored: the generation prompt does not open <think>.
            super().__init__(
                THINK_START,
                THINK_END,
                force_reasoning=False,
                stream_reasoning=stream_reasoning,
                reasoning_default="explicit_thinking",  # as deepseek-v3
                **kwargs,
            )
            self._state = "start"  # start -> lead -> think -> after -> content

        def _detect_and_parse_impl(self, text):
            if self.continue_final_message:
                return super()._detect_and_parse_impl(text)
            if not text.startswith(THINK_START):
                return StreamingParseResult(normal_text=text)
            body = _drop_newline(text[len(THINK_START):], lead=True)
            reasoning, end, content = body.partition(THINK_END)
            if not end:  # truncated inside reasoning
                return StreamingParseResult(reasoning_text=reasoning)
            return StreamingParseResult(
                normal_text=_drop_newline(content, lead=True),
                reasoning_text=_drop_newline(reasoning, lead=False),
            )

        def _parse_streaming_increment_impl(self, new_text):
            if self.continue_final_message:
                return super()._parse_streaming_increment_impl(new_text)
            self._buffer += new_text
            reasoning, normal = "", ""
            while self._buffer:
                buf = self._buffer
                if self._state == "start":
                    if buf.startswith(THINK_START):
                        self._buffer, self._state, self._in_reasoning = buf[len(THINK_START):], "lead", True
                    elif THINK_START.startswith(buf):
                        break
                    else:
                        self._state = "content"
                elif self._state == "lead":
                    self._buffer, self._state = _drop_newline(buf, lead=True), "think"
                elif self._state == "think":
                    i = buf.find(THINK_END)
                    if i >= 0:
                        reasoning += _drop_newline(buf[:i], lead=False)
                        self._buffer, self._state, self._in_reasoning = buf[i + len(THINK_END):], "after", False
                        continue
                    if not self.stream_reasoning:
                        break
                    # Hold back a possible "\n</think>" prefix: its newline is the separator.
                    hold = max(_partial_suffix(buf, "\n" + THINK_END), _partial_suffix(buf, THINK_END))
                    reasoning += buf[:len(buf) - hold]
                    self._buffer = buf[len(buf) - hold:]
                    break
                elif self._state == "after":
                    self._buffer, self._state = _drop_newline(buf, lead=True), "content"
                else:
                    normal += buf
                    self._buffer = ""
            return StreamingParseResult(normal_text=normal, reasoning_text=reasoning)

        def finish(self):
            if self.continue_final_message:
                return super().finish()
            buf, self._buffer = self._buffer, ""
            if self._state in ("lead", "think"):  # truncated inside reasoning
                if self._force_nonempty_content:
                    text, self._accumulated_reasoning = self._accumulated_reasoning + buf, ""
                    return StreamingParseResult(normal_text=text)
                return StreamingParseResult(reasoning_text=buf)
            return StreamingParseResult(normal_text=buf)

    return RippleReasoningDetector


def _tool_detector_class():
    from sglang.srt.function_call.core_types import StructureInfo, ToolCallItem
    from sglang.srt.function_call.hermes_detector import HermesDetector

    class RippleToolDetector(HermesDetector):
        """Text before the first call keeps its whitespace except the one newline the template
        puts before <tool_call> (hermes strips it all)."""

        def __init__(self):
            super().__init__()
            self._held_newline = ""
            self._in_calls = False
            # Hermes can leave whole calls buffered when one speculative delta spans
            # multiple calls. Reconcile the full output with emitted deltas at finish.
            self._seen = ""
            self._emitted = {}

        def _record(self, calls):
            for call in calls:
                emitted = self._emitted.setdefault(call.tool_index, ["", ""])
                emitted[0] += call.name or ""
                emitted[1] += call.parameters or ""
                # SGLang's serving layer also flushes arguments after finish(). Keep
                # its accounting current so it cannot send the same suffix twice.
                while len(self.streamed_args_for_tool) <= call.tool_index:
                    self.streamed_args_for_tool.append("")
                self.streamed_args_for_tool[call.tool_index] = emitted[1]

        def detect_and_parse(self, text, tools):
            res = super().detect_and_parse(text, tools)
            idx = text.find(self.bot_token)
            if idx != -1 and res.normal_text == text[:idx].strip():  # parsed (not the error path)
                res.normal_text = _drop_newline(text[:idx], lead=False)
            return res

        def parse_streaming_increment(self, new_text, tools):
            self._seen += new_text
            res = super().parse_streaming_increment(new_text, tools)
            self._record(res.calls)
            if self._in_calls:  # text after the first call is dropped, as in detect_and_parse
                res.normal_text = ""
                return res
            call_started = bool(self._buffer.startswith(self.bot_token) or self.current_tool_id >= 0 or res.calls)
            self._in_calls = call_started
            if res.normal_text:
                text, self._held_newline = self._held_newline + res.normal_text, ""
                if call_started:  # its trailing newline is the separator before <tool_call>
                    text = _drop_newline(text, lead=False)
                elif text.endswith("\n"):  # undecided until the next text arrives
                    text, self._held_newline = text[:-1], "\n"
                res.normal_text = text
            elif call_started:
                self._held_newline = ""
            return res

        def finish(self, tools):
            res = super().finish(tools)
            self._record(res.calls)
            if self.bot_token in self._seen:
                for i, call in enumerate(self.detect_and_parse(self._seen, tools).calls):
                    name, args = self._emitted.get(i, ["", ""])
                    want = call.parameters or ""
                    if want.startswith(args):
                        suffix = want[len(args):]
                        if suffix or not name:
                            item = ToolCallItem(
                                tool_index=i, name=None if name else call.name, parameters=suffix,
                            )
                            res.calls.append(item)
                            self._record([item])
            if self._in_calls:
                res.normal_text = ""
            elif self._held_newline:
                res.normal_text, self._held_newline = self._held_newline + (res.normal_text or ""), ""
            return res

        def structure_info(self):
            return lambda name: StructureInfo(
                begin='<tool_call>\n{"name": ' + json.dumps(name, ensure_ascii=False) + ', "arguments": ',
                end="}\n</tool_call>",
                trigger="<tool_call>",
            )

    return RippleToolDetector


def check_tokenizer(tokenizer, tokenizer_json):
    """Make `tokenizer` encode and decode like the raw tokenizer.json, or raise.

    Restores the normalizer, pre-tokenizer and decoder from tokenizer.json when they differ."""
    from tokenizers import Tokenizer

    raw = Tokenizer.from_file(tokenizer_json)

    def mismatches():
        bad = []
        for text in TOKENIZER_PROBES:
            ids = raw.encode(text, add_special_tokens=False).ids
            got = tokenizer.encode(text, add_special_tokens=False)
            if got != ids or tokenizer.decode(ids) != raw.decode(ids, skip_special_tokens=False):
                bad.append(text)
        return bad

    if not mismatches():
        return False
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is not None:
        backend.normalizer, backend.pre_tokenizer = raw.normalizer, raw.pre_tokenizer
        backend.decoder = raw.decoder
    bad = mismatches()
    if bad:
        raise RuntimeError(
            f"{type(tokenizer).__name__} does not match {tokenizer_json} on {bad!r}; "
            "export the tokenizer with tokenizer_class PreTrainedTokenizerFast"
        )
    return True


def _tokenizer_json(name, revision=None):
    from sglang.srt.utils.hf_transformers import tokenizer as hf_tok

    try:
        return hf_tok._resolve_local_or_cached_file(name, "tokenizer.json", revision)
    except (OSError, ValueError):  # no tokenizer.json (e.g. a slow-only tokenizer)
        return None


def register():
    """Add the `ripple` parsers to SGLang and check tokenizers on load. Idempotent; call at
    import time so SGLang's spawned processes register too."""
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.parser.reasoning_parser import ReasoningParser
    from sglang.srt.utils.hf_transformers import tokenizer as hf_tok

    ReasoningParser.DetectorMap.setdefault("ripple", _reasoning_detector_class())
    FunctionCallParser.ToolCallParserEnum.setdefault("ripple", _tool_detector_class())

    fixes = hf_tok._apply_post_load_fixes  # fail loudly if SGLang renames it
    if getattr(fixes, "_ripple", False):
        return

    def apply_post_load_fixes(tokenizer, name, revision=None):
        tokenizer = fixes(tokenizer, name, revision)
        path = _tokenizer_json(name, revision)
        if path is not None and check_tokenizer(tokenizer, path):
            logging.getLogger(__name__).warning(
                "Restored %s's normalizer, pre-tokenizer and decoder from tokenizer.json", name)
        return tokenizer

    apply_post_load_fixes._ripple = True
    hf_tok._apply_post_load_fixes = apply_post_load_fixes
