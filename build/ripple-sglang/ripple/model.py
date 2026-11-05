"""Ricursive transformer implemented with SGLang layers.

Used here for NVFP4-MLP/BF16-attention checkpoints with DFlash2.
The checkpoint supplies configuration, tokenizer and weights.
"""

from typing import Iterable, Optional, Tuple

import torch
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers
from torch import nn
from transformers import PretrainedConfig


def rotary_cache_length(config) -> int:
    """Include DFlash lookahead beyond the admitted context boundary."""
    from sglang.srt.server_args import get_global_server_args

    args = get_global_server_args()
    native = int(args.context_length or config.max_position_embeddings)
    return native + (
        int(args.speculative_num_draft_tokens) if args.speculative_algorithm == "DFLASH" else 0
    )


class RicursiveMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_clamp: Optional[float] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.activation_clamp = activation_clamp
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        if self.activation_clamp is not None:
            gate, up = gate_up.chunk(2, dim=-1)
            gate = gate.clamp(max=self.activation_clamp)
            up = up.clamp(min=-self.activation_clamp, max=self.activation_clamp)
            gate_up = torch.cat((gate, up), dim=-1)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class RicursiveAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % tp_size or self.total_num_kv_heads % tp_size:
            raise ValueError(
                "Ricursive separate Q/K/V projections require heads divisible by attention TP"
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Q/K/V/G are kept as separate projections (not fused QKV): V is computed
        # from a different input than Q/K when value residuals are enabled.
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("k_proj", prefix),
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("v_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        if getattr(config, "use_gated_attention", True):
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("g_proj", prefix),
            )
        else:
            self.g_proj = None

        # Per-head Q/K RMSNorm (over head_dim).
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.use_valres = bool(
            getattr(config, "use_valres", getattr(config, "use_value_residuals", False))
        )
        if self.use_valres:
            self.value_res_lambda = nn.Parameter(torch.zeros(1))

        # Per-layer attention pattern: sliding_attention -> sliding window + RoPE;
        # full_attention -> full attention with NO positional encoding (NoPE).
        layer_type = config.layer_types[layer_id]
        self.is_sliding = layer_type == "sliding_attention"
        sliding_window = (config.sliding_window - 1) if self.is_sliding else -1

        self.rotary_emb = None
        if self.is_sliding:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=rotary_cache_length(config),
                base=config.rope_theta,
                is_neox_style=True,
            )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.q_norm.forward_native(q.reshape(-1, self.head_dim)).reshape(-1, self.q_size)
        k = self.k_norm.forward_native(k.reshape(-1, self.head_dim)).reshape(-1, self.kv_size)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        value_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        value_input = hidden_states
        if value_residual is not None and self.use_valres:
            blend = torch.sigmoid(self.value_res_lambda[0]).to(dtype=hidden_states.dtype)
            value_input = torch.lerp(
                value_residual.to(dtype=hidden_states.dtype), hidden_states, blend
            )

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(value_input)

        q, k = self._apply_qk_norm(q, k)

        # RoPE only on sliding (local) layers; global layers are NoPE.
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)

        # Elementwise sigmoid gate (computed in fp32) on the attention output.
        if self.g_proj is not None:
            gate, _ = self.g_proj(hidden_states)
            gate = torch.sigmoid(gate.float()).to(attn_output.dtype)
            attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output


class RicursiveDecoderLayer(nn.Module):
    """PeriNorm decoder layer: 4 RMSNorms, post-norm inside the residual on both sublayers."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = RicursiveAttention(
            config, layer_id, quant_config, prefix=add_prefix("self_attn", prefix)
        )
        self.mlp = RicursiveMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation_clamp=getattr(
                config, "ffn_activation_clamp", getattr(config, "ffn_clamp_threshold", None)
            ),
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        value_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions, hidden_states, forward_batch, value_residual=value_residual
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states)
        return hidden_states


class RicursiveModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: RicursiveDecoderLayer(
                config=config, layer_id=idx, quant_config=quant_config, prefix=prefix
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # embed_scale: applied at runtime when the checkpoint is NON-folded (config carries the
        # scale, e.g. sqrt(hidden_size)). Folded checkpoints set embed_scale=None -> no-op. This
        # mirrors the trainer's modeling so NCCL-synced weights are interpreted identically.
        self.embed_scale = getattr(config, "embed_scale", None)
        self.use_valres = bool(getattr(config, "use_valres", False))
        self.dflash_capture_layers: tuple[int, ...] = ()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        if self.embed_scale is not None:
            hidden_states = hidden_states * self.embed_scale
        # Layer-0 embeddings (post embed_scale) are the value-residual source, threaded to every layer.
        value_residual = hidden_states if self.use_valres else None

        captured = {}
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                positions, hidden_states, forward_batch, value_residual=value_residual
            )
            if index in self.dflash_capture_layers:
                # DFlash was trained on post-block states, before the final norm.
                # PeriNorm closes each residual inside the block.
                captured[index] = hidden_states

        hidden_states = self.norm(hidden_states)
        if self.dflash_capture_layers:
            aux_hidden_states = [captured[index] for index in self.dflash_capture_layers]
            return hidden_states, aux_hidden_states
        return hidden_states


class RicursiveForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.model = RicursiveModel(config, quant_config, prefix=add_prefix("model", prefix))
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        """Expose the target embedding table shared by the DFlash worker."""
        return self.model.embed_tokens

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        """Capture post-block, pre-final-norm states in the drafter's configured order.

        :param layer_ids: Zero-based target block indices from the draft configuration.
        :raises ValueError: If indices are empty, duplicated, or outside the target model.
        """
        if (
            not layer_ids
            or any(type(index) is not int for index in layer_ids)
            or len(set(layer_ids)) != len(layer_ids)
            or any(index < 0 or index >= len(self.model.layers) for index in layer_ids)
        ):
            raise ValueError("DFlash requires unique, valid target layer indices")
        self.model.dflash_capture_layers = tuple(layer_ids)

    def get_attention_sliding_window_size(self) -> Optional[int]:
        """Return the window used by Ricursive local-attention layers.

        SGLang's native ``--disable-hybrid-swa-memory`` flag controls compact KV allocation.
        Per-layer attention semantics are architectural and must not depend on an environment
        variable.
        """
        layer_types = getattr(self.config, "layer_types", ()) or ()
        if "sliding_attention" not in layer_types:
            return None
        sliding_window = getattr(self.config, "sliding_window", None)
        if sliding_window is None:
            return None
        return sliding_window - 1

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        if self.model.dflash_capture_layers:
            hidden_states, aux_hidden_states = hidden_states
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Map the checkpoint's separate gate/up tensors into SGLang's fused MLP."""
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded, unused = set(), []  # home-k8s: load-gap check ported from inference/server/model.py
        for name, weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for source, shard in (("gate_proj", 0), ("up_proj", 1)):
                if source in name:
                    name = name.replace(source, "gate_up_proj")
                    if name in params:
                        params[name].weight_loader(params[name], weight, shard)
                        loaded.add(name)
                    else:
                        unused.append(name)
                    break
            else:
                if name in params:
                    param = params[name]
                    getattr(param, "weight_loader", default_weight_loader)(param, weight)
                    loaded.add(name)
                else:
                    unused.append(name)
        tied = {"lm_head.weight"} if self.config.tie_word_embeddings else set()
        # ModelOpt adds RadixAttention k_scale/v_scale params (init -1, set to 1.0 after loading when absent);
        # this checkpoint has none and runs BF16 KV, so they are expected gaps.
        missing = sorted(n for n in set(params) - loaded - tied
                         if not n.endswith((".attn.k_scale", ".attn.v_scale")))
        if unused or missing:
            import logging
            import os

            msg = (f"Ripple load gaps: {len(unused)} unused checkpoint tensors {unused[:8]}, "
                   f"{len(missing)} unloaded params {missing[:8]}")
            if os.environ.get("RIPPLE_STRICT_LOAD", "1") == "1":
                raise ValueError(msg)
            logging.getLogger(__name__).error(msg)


class _NoRope(nn.Module):
    """Identity RoPE for DFlash2 draft layers listed in config.no_rope_layers."""

    # SGLang's fused draft context-KV path applies one RoPE table to every layer; it
    # only accepts neox-style RoPE, so this marker turns it off for drafts with NoPE layers.
    is_neox_style = False

    def forward(self, positions, q, k):
        return q, k


# DFlash2 drafts are trained on 16K-token rows (12,288 tokens of context + 4,096 predicted),
# so a global draft layer never attends further back than this.
DRAFT_TRAINING_CONTEXT = 16384


def _patch_draft_attention():
    """Fit SGLang 0.5.19's DFlash draft attention to DFlash2 drafts with global layers.

    It applies RoPE in every layer, ignoring no_rope_layers, and its draft window
    (--speculative-draft-window-size) truncates every layer, including full attention,
    so the window must cover the draft's training context.
    """
    from sglang.srt.models.dflash import DFlashAttention
    from sglang.srt.server_args import get_global_server_args

    init = DFlashAttention.__init__

    def __init__(self, config, layer_id, *args, **kwargs):
        init(self, config, layer_id, *args, **kwargs)
        if layer_id in (getattr(config, "no_rope_layers", None) or ()):
            self.rotary_emb = _NoRope()
            self.use_table_qk_norm_rope = False
        layer_types = getattr(config, "layer_types", None) or ()
        window = get_global_server_args().speculative_draft_window_size
        if (
            layer_id < len(layer_types)
            and layer_types[layer_id] == "full_attention"
            and window is not None
            and window < DRAFT_TRAINING_CONTEXT
        ):
            raise ValueError(
                f"Draft layer {layer_id} attends globally over its {DRAFT_TRAINING_CONTEXT}-token "
                f"training rows; use --speculative-draft-window-size >= {DRAFT_TRAINING_CONTEXT}"
            )

    DFlashAttention.__init__ = __init__


_patch_draft_attention()


def _raise_on_load_gaps(model: str, unused, unloaded) -> None:
    """Refuse checkpoint tensors nothing consumed and parameters nothing loaded."""
    if unused or unloaded:
        raise ValueError(f"{model}: unused tensors {sorted(unused)[:20]}, "
                         f"unloaded parameters {sorted(unloaded)[:20]}")


def _strict_draft_loading():
    """SGLang 0.5.19's DFlash loader skips tensors it cannot place: refuse such drafts."""
    from sglang.srt.models.dflash import DFlashDraftModel

    load, shared = DFlashDraftModel.load_weights, ("inv_freq", "embed_tokens.weight", "lm_head.weight")

    def load_weights(self, weights):
        if (getattr(self.config, "dflash_config", None) or {}).get("block_norm", "pre") != "pre":
            raise ValueError("Only pre-norm DFlash drafts are supported")
        params, calls, unused = dict(self.named_parameters()), [], []
        own = {n: p.weight_loader for n, p in params.items() if "weight_loader" in p.__dict__}
        for n, p in params.items():  # record which parameters receive a tensor
            f = own.get(n, default_weight_loader)
            p.weight_loader = lambda q, *a, _f=f, _n=n: (calls.append(_n), _f(q, *a))[1]

        def watch(stream):  # a tensor that reached no weight_loader was skipped
            for name, weight in stream:
                before = len(calls)
                yield name, weight
                if len(calls) == before and not name.endswith(shared):
                    unused.append(name)

        try:
            load(self, watch(weights))
        finally:
            for n, p in params.items():
                del p.weight_loader
                if n in own:
                    p.weight_loader = own[n]
        unloaded = {n for n in params if not n.endswith(("k_scale", "v_scale"))} - set(calls)
        _raise_on_load_gaps("DFlash draft", unused, unloaded)

    DFlashDraftModel.load_weights = load_weights


_strict_draft_loading()
