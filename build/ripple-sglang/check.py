import hashlib, py_compile, sys, traceback

sha = hashlib.sha256(open("/opt/ripple/chat_template.jinja", "rb").read()).hexdigest()
assert sha == "877c2dbf365243dda14738ca250b31398059b35ffd91d4ed0ec68be45ca6ebd3", sha
import sglang, torch  # noqa: E401
print("sglang", sglang.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)
assert sglang.__version__.startswith("0.5.19"), sglang.__version__
for f in ("model.py", "ripple_format.py", "serve.py"):
    py_compile.compile("/opt/ripple/" + f, doraise=True)
sys.path.insert(0, "/opt/ripple")
try:
    import serve  # noqa: F401  (registers the architecture and both parsers)
except (ImportError, OSError, RuntimeError) as e:
    traceback.print_exc()
    if any(w in str(e).lower() for w in ("libcuda", "cuda", "nvidia", "driver")):
        print("WARN: CUDA-only import failure on a GPU-less builder; byte-compile passed")
        sys.exit(0)
    raise
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.parser.reasoning_parser import ReasoningParser

assert "ripple" in ReasoningParser.DetectorMap
assert "ripple" in FunctionCallParser.ToolCallParserEnum
assert "RicursiveForCausalLM" in ModelRegistry.models
print("ripple registered")
