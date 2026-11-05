"""home-k8s launcher: Ripple-65B (NVFP4 MLPs / BF16 attention) on stock SGLang 0.5.19, one SM120 GPU.

Derived from Ricursive's inference/server/serve.py (B200, BF16, TP=2) minus its defaults: every flag
comes from the Deployment args. Registration runs at import time so SGLang's spawned scheduler
processes (which re-import this __main__ module) see the Ripple architecture and parsers too.
"""

import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sglang.srt.models.registry import ModelRegistry  # noqa: E402

from model import RicursiveForCausalLM  # noqa: E402
import ripple_format  # noqa: E402

ModelRegistry.models["RicursiveForCausalLM"] = RicursiveForCausalLM
ripple_format.register()


def main():
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    args = prepare_server_args(sys.argv[1:])
    if args.dtype != "bfloat16":
        raise ValueError("Ripple's BF16 attention needs --dtype bfloat16")
    try:
        launch_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
