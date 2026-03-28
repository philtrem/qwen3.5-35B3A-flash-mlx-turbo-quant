"""
Qwen3.5-35B-A3B model with flash-loaded MoE experts.

Uses the correct qwen3_5.py architecture from mlx-lm 0.31.1+, with only the
MoE block replaced to stream expert weights from SSD.
"""

from __future__ import annotations

import os as _os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# Profiling accumulator (opt-in via PROFILE_MOE=1)
_PROFILE = _os.environ.get("PROFILE_MOE", "0") == "1"

class _MoeProfiler:
    def __init__(self):
        self.async_queue = []  # async_eval queueing + shared expert graph build
        self.expert_io = []    # materialize indices + Rust load_experts
        self.py_overhead = []  # tensor build + remap
        self.compute = []      # gather_qmm + combine (forced eval)
        self.total = []

    def report(self):
        if not self.total:
            return
        n = len(self.total)
        def ms(lst): return sum(lst) * 1000
        def avg_ms(lst): return ms(lst) / len(lst) if lst else 0
        total_s = sum(self.total)
        print(f"\n=== MoE Profiling ({n} calls, {total_s:.2f}s total) ===")
        print(f"  async_queue:{ms(self.async_queue):8.1f}ms ({100*sum(self.async_queue)/total_s:.1f}%) avg {avg_ms(self.async_queue):.2f}ms")
        print(f"  expert_io:  {ms(self.expert_io):8.1f}ms ({100*sum(self.expert_io)/total_s:.1f}%) avg {avg_ms(self.expert_io):.2f}ms")
        print(f"  py_overhead:{ms(self.py_overhead):8.1f}ms ({100*sum(self.py_overhead)/total_s:.1f}%) avg {avg_ms(self.py_overhead):.2f}ms")
        print(f"  compute:    {ms(self.compute):8.1f}ms ({100*sum(self.compute)/total_s:.1f}%) avg {avg_ms(self.compute):.2f}ms")
        print(f"  total:      {ms(self.total):8.1f}ms (100%) avg {avg_ms(self.total):.2f}ms")

    def reset(self):
        self.__init__()

moe_profiler = _MoeProfiler()


# Routing logger — enable/disable at runtime via enable_routing_recording()
from collections import Counter
_RECORD_ROUTING = False
_routing_counts: Counter = Counter()  # (layer_idx, expert_idx) → activation count

# Python-side expert tensor cache — stores mx.arrays, skips Rust copy chain on hit
_expert_tensor_cache: dict = {}  # (layer_idx, expert_idx) → dict of 9 mx.arrays

def enable_routing_recording():
    global _RECORD_ROUTING
    _RECORD_ROUTING = True

def disable_routing_recording():
    global _RECORD_ROUTING
    _RECORD_ROUTING = False

def get_routing_counts() -> dict:
    """Return {(layer_idx, expert_idx): count} for all observed routings."""
    return dict(_routing_counts)

def reset_routing_counts():
    _routing_counts.clear()


_layer_quant_info: dict = {}  # layer_idx → (bits, group_size)

def preload_python_cache(expert_manager, warm_experts: list):
    """Pre-populate _expert_tensor_cache with mx.arrays for warm set experts.

    Args:
        expert_manager: FlashExpertManager instance
        warm_experts: list of [layer_idx, expert_idx] pairs
    Returns:
        number of experts cached
    """
    from collections import defaultdict
    by_layer = defaultdict(list)
    for layer, expert in warm_experts:
        by_layer[layer].append(expert)

    total = 0
    for layer_idx in sorted(by_layer):
        expert_ids = by_layer[layer_idx]
        expert_data = expert_manager.load_experts(layer_idx, expert_ids)
        _layer_quant_info[layer_idx] = (expert_data["quant_bits"], expert_data["quant_group_size"])
        n = len(expert_ids)
        for i, eidx in enumerate(expert_ids):
            tensors = _parse_expert_tensors(expert_data, i, n)
            _expert_tensor_cache[(layer_idx, eidx)] = tensors
            total += 1

    return total


def _parse_expert_tensors(expert_data, idx_in_batch, batch_size):
    """Extract single expert's tensors from a stacked Rust batch as mx.arrays."""
    tensors = {}
    for name in ("gate_weight", "up_weight", "down_weight"):
        data = expert_data[name]
        shape = expert_data[f"{name}_shape"]
        per_expert = len(data) // shape[0]
        start = idx_in_batch * per_expert
        tensors[name] = mx.array(
            np.frombuffer(data[start:start + per_expert], dtype=np.uint32)
            .reshape(1, shape[1], shape[2])
        )
    for name in ("gate_scales", "gate_biases", "up_scales", "up_biases",
                  "down_scales", "down_biases"):
        data = expert_data[name]
        shape = expert_data[f"{name}_shape"]
        per_expert = len(data) // shape[0]
        start = idx_in_batch * per_expert
        tensors[name] = mx.array(
            np.frombuffer(data[start:start + per_expert], dtype=np.uint16)
            .reshape(1, shape[1], shape[2])
        ).view(mx.bfloat16)
    return tensors


# Import the correct model components from mlx-lm
from mlx_lm.models.qwen3_5 import (
    Attention,
    GatedDeltaNet,
    MLP,
    RMSNormGated,
    SparseMoeBlock,
    TextModelArgs,
)
from mlx_lm.models.base import (
    create_attention_mask,
    create_ssm_mask,
)
from mlx_lm.models.cache import ArraysCache, KVCache

from flash_qwen._native import FlashExpertManager
from flash_qwen.turbo_quant import TurboQuantCache


class FlashSparseMoeBlock(nn.Module):
    """MoE block that streams expert weights from SSD via the Rust loader."""

    def __init__(self, args: TextModelArgs, layer_idx: int, expert_manager: FlashExpertManager):
        super().__init__()
        dim = args.hidden_size

        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        # Router stays in RAM
        self.gate = nn.Linear(dim, self.num_experts, bias=False)

        # Shared expert stays in RAM
        self.shared_expert = MLP(dim, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

        # No switch_mlp -- expert weights live on SSD
        self.expert_manager = expert_manager
        self.layer_idx = layer_idx

    def __call__(self, x: mx.array) -> mx.array:
        _p = _PROFILE
        if _p: _t0 = time.perf_counter()

        # 1. Router (lazy graph)
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        # 2. Dispatch router to GPU, then build shared expert graph.
        #    async_eval queues GPU work without blocking Python.
        #    shared_expert is independent of routing — GPU can execute it
        #    during the Rust SSD I/O below (since GIL is released in Rust).
        mx.async_eval(inds)
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        mx.async_eval(shared_y)

        if _p: _t1 = time.perf_counter()

        # 3. Materialize router indices (.tolist blocks until GPU done)
        orig_shape = inds.shape
        flat_inds = inds.reshape(-1).tolist()
        unique_experts = sorted(set(flat_inds))

        if _RECORD_ROUTING:
            for eidx in unique_experts:
                _routing_counts[(self.layer_idx, eidx)] += 1

        # 4. Remap indices + prepare gather_qmm inputs
        remap = {orig: local for local, orig in enumerate(unique_experts)}
        flat_remapped = [remap[i] for i in flat_inds]
        local_inds = mx.array(flat_remapped, dtype=mx.uint32).reshape(orig_shape)

        x_exp = mx.expand_dims(x, (-2, -3))

        do_sort = local_inds.size >= 64
        idx = local_inds
        inv_order = None
        if do_sort:
            M = local_inds.shape[-1]
            flat_idx = local_inds.flatten()
            order = mx.argsort(flat_idx)
            inv_order = mx.argsort(order)
            x_exp = x_exp.flatten(0, -3)[order // M]
            idx = flat_idx[order]

        if _p: _t2 = time.perf_counter()

        # 5. Rust: load experts + build tensors + dispatch gather_qmm on GPU
        #    One Rust call replaces: load_experts + 9 tensor constructions + 3 gather_qmm calls
        x_down = self.expert_manager.moe_gather_qmm(
            x_exp, idx, self.layer_idx, unique_experts, do_sort,
        )

        if _p: _t3 = time.perf_counter()

        # 6. Unsort + weighted sum (stays in Python/MLX)
        if do_sort:
            x_down = x_down[inv_order]
            x_down = mx.unflatten(x_down, 0, local_inds.shape)

        x_down = x_down.squeeze(-2)
        y = (x_down * scores[..., None]).sum(axis=-2)

        # 7. Combine (shared_y already computed on GPU during step 4)
        result = y + shared_y

        # 8. Prefetch: predict next layer needs similar experts, load in background
        next_layer = self.layer_idx + 1
        if next_layer < self.expert_manager.num_layers():
            self.expert_manager.submit_prefetch(next_layer, unique_experts)

        if _p:
            mx.eval(result)
            _t4 = time.perf_counter()
            moe_profiler.async_queue.append(_t1 - _t0)
            moe_profiler.expert_io.append(_t2 - _t1)
            moe_profiler.py_overhead.append(_t3 - _t2)
            moe_profiler.compute.append(_t4 - _t3)
            moe_profiler.total.append(_t4 - _t0)

        return result


class FlashDecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int, expert_manager: FlashExpertManager):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if args.num_experts > 0:
            self.mlp = FlashSparseMoeBlock(args, layer_idx, expert_manager)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x, mask=None, cache=None):
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class FlashTextModel(nn.Module):
    def __init__(self, args: TextModelArgs, expert_manager: FlashExpertManager):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            FlashDecoderLayer(args=args, layer_idx=i, expert_manager=expert_manager)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def __call__(self, inputs, cache=None):
        hidden_states = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm(hidden_states)


class FlashLanguageModel(nn.Module):
    """Matches the TextModel structure from qwen3_5.py."""

    def __init__(self, args: TextModelArgs, expert_manager: FlashExpertManager):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FlashTextModel(args, expert_manager)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self, kv_bits=None):
        def _make_kv():
            if kv_bits is not None:
                return TurboQuantCache(
                    head_dim=self.args.head_dim, bits=kv_bits,
                )
            return KVCache()

        return [ArraysCache(size=2) if l.is_linear else _make_kv() for l in self.layers]


class Model(nn.Module):
    """Top-level model matching qwen3_5.Model's weight structure.

    Weights live under language_model.model.layers.N...
    """

    def __init__(self, args: TextModelArgs, expert_manager: FlashExpertManager):
        super().__init__()
        self.args = args
        self.language_model = FlashLanguageModel(args, expert_manager)

    def __call__(self, inputs, cache=None):
        return self.language_model(inputs, cache=cache)

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self, kv_bits=None):
        return self.language_model.make_cache(kv_bits=kv_bits)
