# dbirks/home-k8s#158: FP8 KV cache for Flint on the stock (non-SM100) path.
#
# FlintAttention asserts a BF16 cache because its fused Triton qk_norm_rope_cache kernel writes K/V into the
# cache itself, in BF16. With an fp8 cache we pass no cache views (the kernel then only norms/ropes q and k,
# k in place in qkv) and let vLLM's own unified_kv_cache_update (FlashInfer do_kv_cache_update ->
# reshape_and_cache_flash) quantize K/V into the cache, exactly what a stock Attention layer does. FlashInfer on
# SM120 keeps Q in BF16 with an fp8 cache (fa2 prefill, XQA decode), both with sinks + sliding window.
# The SM100 Flint decode kernels read a BF16 cache, so fp8 is refused there.
import os
import sys

import vllm

p = os.path.join(os.path.dirname(vllm.__file__), "models/flint/model.py")
s = open(p).read()


def sub(old, new):
    global s
    assert s.count(old) == 1, f"patch anchor not found exactly once: {old[:80]!r}"
    s = s.replace(old, new)


sub(
    "    unified_attention_with_output,\n)",
    "    unified_attention_with_output,\n    unified_kv_cache_update,\n)",
)
sub(
    '        assert self.attn.kv_cache_dtype in ("auto", "bfloat16")\n',
    "        # dbirks/home-k8s#158: an fp8 cache is written by vLLM's reshape_and_cache, not the fused kernel.\n"
    '        self.fp8_kv = self.attn.kv_cache_dtype.startswith("fp8")\n'
    '        assert self.attn.kv_cache_dtype in ("auto", "bfloat16") or self.fp8_kv\n',
)
sub(
    "        self.flint_decode = _flint_kernels(quant_config)\n",
    "        self.flint_decode = _flint_kernels(quant_config)\n"
    '        assert not (self.fp8_kv and self.flint_decode), "Flint decode kernels need a BF16 KV cache"\n',
)
sub(
    "        qkv, _ = self.qkv_proj(hidden_states)\n"
    "        k_cache, v_cache, layer_slots = self._cache_views()\n",
    "        qkv, _ = self.qkv_proj(hidden_states)\n"
    "        k_cache, v_cache, layer_slots = (\n"
    "            (None, None, None) if self.fp8_kv else self._cache_views()\n"
    "        )\n",
)
sub(
    "        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)\n"
    "        unified_attention_with_output(\n"
    "            q.view(num_tokens, self.num_heads, self.head_dim),\n"
    "            k.view(num_tokens, self.num_kv_heads, self.head_dim),\n"
    "            v.view(num_tokens, self.num_kv_heads, self.head_dim),\n"
    "            attn_output.view(num_tokens, self.num_heads, self.head_dim),\n"
    "            self.attn.layer_name,\n"
    "        )\n",
    "        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)\n"
    "        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)\n"
    "        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)\n"
    "        dep = None\n"
    "        if self.fp8_kv:\n"
    "            dep = unified_kv_cache_update(k, v, self.attn.layer_name)\n"
    "        unified_attention_with_output(\n"
    "            q.view(num_tokens, self.num_heads, self.head_dim),\n"
    "            k,\n"
    "            v,\n"
    "            attn_output.view(num_tokens, self.num_heads, self.head_dim),\n"
    "            self.attn.layer_name,\n"
    "            kv_cache_dummy_dep=dep,\n"
    "        )\n",
)

out = sys.argv[1] if len(sys.argv) > 1 else p
open(out, "w").write(s)
print("fp8 KV patch applied to", out)
