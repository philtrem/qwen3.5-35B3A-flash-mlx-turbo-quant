#!/usr/bin/env python3
"""
Measure expert prediction accuracy when approximating layer L's output
WITHOUT MoE contribution (dense MLP only) and feeding it to layer L+1.

Tests the hypothesis: if we compute h_approx = attention + dense_mlp (skip MoE),
does the next layer's router select the same experts as in the real forward pass?

Usage:
    python3 scripts/measure_moe_prediction.py --model-path mlx-community/gemma-4-26b-a4b-it-4bit
    python3 scripts/measure_moe_prediction.py --model-path /local/path/to/model

First run downloads ~13 GB from HuggingFace. MLX uses mmap so 16 GB RAM is fine.
"""

import argparse
import sys
import os
import copy
import importlib
import importlib.util
import time

import mlx.core as mx
import numpy as np

# ── Defaults ──────────────────────────────────────────────
DEFAULT_MODEL = "./gemma-4-26b-a4b-it-MLX-4bit"
DEFAULT_PROMPT = "Explain the theory of general relativity in simple terms."
DEFAULT_NUM_TOKENS = 20
TOP_K = 8         # model's actual top-k
PREDICT_K = 12    # predict this many for overlap measurement
# ──────────────────────────────────────────────────────────


def register_gemma4_modules():
    """Dynamically register gemma4 + gemma4_text with mlx_lm if not built-in."""
    for mod_name in ("gemma4_text", "gemma4"):
        fqn = f"mlx_lm.models.{mod_name}"
        if fqn in sys.modules:
            continue
        try:
            importlib.import_module(fqn)
            continue
        except (ImportError, ValueError):
            pass

        # Use local copy from split_gemma4
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        src = os.path.join(project_dir, "split_gemma4", f"{mod_name}.py")
        if not os.path.exists(src):
            print(f"Error: {src} not found. Need gemma4 model definitions.")
            sys.exit(1)

        spec = importlib.util.spec_from_file_location(fqn, src)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "mlx_lm.models"
        sys.modules[fqn] = mod
        spec.loader.exec_module(mod)
        print(f"  registered {fqn}")


def patch_sanitize():
    """
    Patch gemma4.Model.sanitize to handle philtrem/gemma-4-26b-a4b-it-MLX-4bit naming:
    - weight_scales → scales, weight_biases → biases
    - Fused experts.gate_up_proj → split into switch_glu.gate_proj + switch_glu.up_proj
    - experts.down_proj → switch_glu.down_proj
    """
    gemma4_mod = sys.modules["mlx_lm.models.gemma4"]
    original_sanitize = gemma4_mod.Model.sanitize

    def patched(self, weights):
        weights = original_sanitize(self, weights)

        result = {}
        for k, v in weights.items():
            nk = k
            # Rename quantization keys
            if nk.endswith(".weight_scales"):
                nk = nk[: -len(".weight_scales")] + ".scales"
            elif nk.endswith(".weight_biases"):
                nk = nk[: -len(".weight_biases")] + ".biases"

            # Fused gate_up_proj → split into gate_proj + up_proj under switch_glu
            if ".experts.gate_up_proj" in nk and ".switch_glu." not in nk:
                base = nk.split(".experts.gate_up_proj")[0] + ".experts.switch_glu."
                tail = nk.split(".experts.gate_up_proj")[1]  # "" / "_scales" / "_biases"
                if tail == "":
                    sfx = ".weight"
                elif tail == "_scales":
                    sfx = ".scales"
                elif tail == "_biases":
                    sfx = ".biases"
                else:
                    sfx = tail
                mid = v.shape[1] // 2
                result[base + "gate_proj" + sfx] = v[:, :mid]
                result[base + "up_proj" + sfx] = v[:, mid:]
                continue

            # experts.down_proj → switch_glu.down_proj
            if ".experts.down_proj" in nk and ".switch_glu." not in nk:
                base = nk.split(".experts.down_proj")[0] + ".experts.switch_glu."
                tail = nk.split(".experts.down_proj")[1]
                if tail == "":
                    sfx = ".weight"
                elif tail == "_scales":
                    sfx = ".scales"
                elif tail == "_biases":
                    sfx = ".biases"
                else:
                    sfx = tail
                result[base + "down_proj" + sfx] = v
                continue

            result[nk] = v
        return result

    gemma4_mod.Model.sanitize = patched


def cosine_similarity(a, b):
    """Cosine similarity between two MLX arrays (flattened)."""
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    na = mx.sqrt(mx.sum(a_flat * a_flat))
    nb = mx.sqrt(mx.sum(b_flat * b_flat))
    return (dot / (na * nb + 1e-8)).item()


def routing_overlap(actual_idx, predicted_idx, top_k=TOP_K):
    """Fraction of actual top-k experts found in predicted set."""
    actual = set(actual_idx.reshape(-1).tolist())
    predicted = set(predicted_idx.reshape(-1).tolist())
    return len(actual & predicted) / len(actual)


def decomposed_layer_forward(layer, x, mask, cache):
    """
    Run a MoE decoder layer step-by-step, returning intermediates.
    Returns: (h_real, h_post_attn, h_no_moe, actual_indices)
    """
    # ── Attention block ──
    residual = x
    h = layer.input_layernorm(x)
    h = layer.self_attn(h, mask, cache)
    h = layer.post_attention_layernorm(h)
    h_post_attn = residual + h

    residual_ff = h_post_attn

    # ── Dense MLP path ──
    h1 = layer.pre_feedforward_layernorm(h_post_attn)
    h1 = layer.mlp(h1)
    h1 = layer.post_feedforward_layernorm_1(h1)

    # ── Router ──
    actual_indices, actual_weights = layer.router(h_post_attn)

    # ── Expert path ──
    h2_input = layer.pre_feedforward_layernorm_2(h_post_attn)
    h2 = layer.experts(h2_input, actual_indices, actual_weights)
    h2 = layer.post_feedforward_layernorm_2(h2)

    # ── Combine: real output ──
    h_real = layer.post_feedforward_layernorm(h1 + h2)
    h_real = residual_ff + h_real
    if layer.layer_scalar is not None:
        h_real = h_real * layer.layer_scalar

    # ── Combine: approximate output (no MoE) ──
    h_no_moe = layer.post_feedforward_layernorm(h1)
    h_no_moe = residual_ff + h_no_moe
    if layer.layer_scalar is not None:
        h_no_moe = h_no_moe * layer.layer_scalar

    return h_real, h_post_attn, h_no_moe, actual_indices


def predict_routing(layer, h_input, mask, cache_entry, predict_k=PREDICT_K):
    """
    Run one layer's attention + router on an approximate input.
    Uses a COPY of the cache (does not modify the real cache).
    Returns (top_k_indices, top_predict_k_indices).
    """
    temp_cache = copy.deepcopy(cache_entry)
    residual = h_input
    h = layer.input_layernorm(h_input)
    h = layer.self_attn(h, mask, temp_cache)
    h = layer.post_attention_layernorm(h)
    h_post_attn = residual + h

    # Standard top-k (model's own top_k=8)
    top8_indices, _ = layer.router(h_post_attn)

    # Extended top-k for broader prediction window
    if predict_k > TOP_K:
        router = layer.router
        x = router.norm(h_post_attn)
        x = x * router._root_size
        x = x * router.scale
        scores = router.proj(x)
        topk_indices = mx.argpartition(-scores, kth=predict_k - 1, axis=-1)[
            ..., :predict_k
        ]
    else:
        topk_indices = top8_indices

    return top8_indices, topk_indices


def run_non_moe_layer(layer, x, mask, cache):
    """Forward pass for a non-MoE layer (shouldn't happen for 26B, but just in case)."""
    residual = x
    h = layer.input_layernorm(x)
    h = layer.self_attn(h, mask, cache)
    h = layer.post_attention_layernorm(h)
    h = residual + h

    residual_ff = h
    h = layer.pre_feedforward_layernorm(h)
    h = layer.mlp(h)
    h = layer.post_feedforward_layernorm(h)
    h = residual_ff + h
    if layer.layer_scalar is not None:
        h = h * layer.layer_scalar
    return h


def measure_token(inner, token_ids, cache):
    """
    Process one token, measuring per-layer prediction accuracy.

    For each MoE layer L (starting from L=1):
      1. PREDICT: run L's attention+router on h_no_moe_{L-1} (cache not yet updated)
      2. REAL: run L's full forward pass, capturing h_no_moe_L and actual indices
      3. Compare predicted vs actual routing

    Returns: list of per-layer dicts with 'cosine', 'overlap', 'actual_indices'
    """
    from mlx_lm.models.base import create_attention_mask

    h = inner.embed_tokens(token_ids)

    # ── Mask creation ──
    # Find representative cache entries for each attention type
    global_cache_idx = None
    sliding_cache_idx = None
    for i, layer in enumerate(inner.layers):
        if layer.layer_type == "full_attention" and global_cache_idx is None:
            global_cache_idx = i
        elif layer.layer_type == "sliding_attention" and sliding_cache_idx is None:
            sliding_cache_idx = i

    global_mask = create_attention_mask(
        h, cache[global_cache_idx] if global_cache_idx is not None else None
    )
    sliding_mask = create_attention_mask(
        h,
        cache[sliding_cache_idx] if sliding_cache_idx is not None else None,
        window_size=inner.window_size,
    )

    results = []
    prev_h_no_moe = None
    prev_h_post_attn = None  # for level B (cheapest)

    for i, layer in enumerate(inner.layers):
        is_global = layer.layer_type == "full_attention"
        mask = global_mask if is_global else sliding_mask

        layer_result = {"layer": i, "is_moe": layer.enable_moe}

        if layer.enable_moe:
            # ── LEVEL C: full prediction (dense MLP + attention + router) ──
            pred_top8 = None
            pred_topk = None
            if prev_h_no_moe is not None:
                pred_top8, pred_topk = predict_routing(
                    layer, prev_h_no_moe, mask, cache[i]
                )
                mx.eval(pred_top8, pred_topk)

            # ── LEVEL A: skip attention (router on h_no_moe directly) ──
            pred_a_topk = None
            if prev_h_no_moe is not None:
                router = layer.router
                x = router.norm(prev_h_no_moe)
                x = x * router._root_size
                x = x * router.scale
                scores = router.proj(x)
                pred_a_topk = mx.argpartition(-scores, kth=PREDICT_K - 1, axis=-1)[
                    ..., :PREDICT_K
                ]
                mx.eval(pred_a_topk)

            # ── LEVEL B: skip attention + MLP (router on h_post_attn directly) ──
            pred_b_topk = None
            if prev_h_post_attn is not None:
                router = layer.router
                x = router.norm(prev_h_post_attn)
                x = x * router._root_size
                x = x * router.scale
                scores = router.proj(x)
                pred_b_topk = mx.argpartition(-scores, kth=PREDICT_K - 1, axis=-1)[
                    ..., :PREDICT_K
                ]
                mx.eval(pred_b_topk)

            # ── REAL forward pass ──
            h_real, h_post_attn, h_no_moe, actual_indices = decomposed_layer_forward(
                layer, h, mask, cache[i]
            )
            mx.eval(h_real, h_no_moe, actual_indices)

            # ── Metrics ──
            layer_result["cosine"] = cosine_similarity(h_real, h_no_moe)

            if pred_top8 is not None:
                layer_result["overlap_top8"] = routing_overlap(
                    actual_indices, pred_top8, top_k=TOP_K
                )
                layer_result["overlap_topk"] = routing_overlap(
                    actual_indices, pred_topk, top_k=TOP_K
                )
            if pred_a_topk is not None:
                layer_result["overlap_a"] = routing_overlap(
                    actual_indices, pred_a_topk, top_k=TOP_K
                )
            if pred_b_topk is not None:
                layer_result["overlap_b"] = routing_overlap(
                    actual_indices, pred_b_topk, top_k=TOP_K
                )

            layer_result["actual_indices"] = actual_indices.tolist()

            prev_h_no_moe = h_no_moe
            prev_h_post_attn = h_post_attn
            h = h_real
        else:
            # Non-MoE layer
            h = run_non_moe_layer(layer, h, mask, cache[i])
            mx.eval(h)
            prev_h_no_moe = None
            prev_h_post_attn = None

        results.append(layer_result)

    # Final norm + logits (for next token generation)
    h = inner.norm(h)
    return results, h


def main():
    parser = argparse.ArgumentParser(description="Measure MoE prediction accuracy")
    parser.add_argument("--model-path", default=DEFAULT_MODEL,
                        help=f"HuggingFace model ID or local path (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-tokens", type=int, default=DEFAULT_NUM_TOKENS,
                        help="Number of tokens to generate and measure")
    args = parser.parse_args()

    print("Registering model modules...")
    register_gemma4_modules()
    patch_sanitize()

    print(f"Loading model: {args.model_path}")
    from mlx_lm import load

    # Fix tokenizer_config.json if extra_special_tokens is a list (transformers compat)
    import json
    tc_path = os.path.join(args.model_path, "tokenizer_config.json")
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            tc = json.load(f)
        if isinstance(tc.get("extra_special_tokens"), list):
            tc["extra_special_tokens"] = {}
            with open(tc_path, "w") as f:
                json.dump(tc, f, indent=2)
            print("  fixed tokenizer_config.json extra_special_tokens")

    model, tokenizer = load(args.model_path)

    # Access internal structure
    # gemma4.Model -> .language_model -> gemma4_text.Model -> .model -> Gemma4TextModel
    if hasattr(model, "language_model"):
        text_model = model.language_model  # gemma4.Model wraps gemma4_text.Model
        inner = text_model.model           # Gemma4TextModel
    else:
        text_model = model
        inner = model.model

    num_layers = len(inner.layers)
    moe_layers = [i for i, l in enumerate(inner.layers) if l.enable_moe]
    print(f"Model: {num_layers} layers, {len(moe_layers)} MoE layers")

    # Tokenize prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    print(f"Prompt: '{args.prompt}' ({len(prompt_tokens)} tokens)")

    # Setup cache
    cache = model.make_cache()

    # ── Prefill: process prompt normally ──
    print("Running prefill...")
    prompt_ids = mx.array(prompt_tokens)[None]  # [1, seq_len]
    logits = model(prompt_ids, cache=cache)
    mx.eval(logits)
    next_token = mx.argmax(logits[:, -1:, :], axis=-1)
    mx.eval(next_token)

    generated = []

    # ── Generate tokens with measurement ──
    print(f"\nGenerating {args.num_tokens} tokens with measurement...\n")

    # Per-layer accumulators
    cosine_sums = {i: 0.0 for i in moe_layers}
    overlap8_sums = {i: 0.0 for i in moe_layers}
    overlapk_sums = {i: 0.0 for i in moe_layers}
    overlap_a_sums = {i: 0.0 for i in moe_layers}  # level A: skip attn
    overlap_b_sums = {i: 0.0 for i in moe_layers}  # level B: skip attn+mlp
    overlap_counts = {i: 0 for i in moe_layers}
    token_count = 0

    for t in range(args.num_tokens):
        t0 = time.time()

        results, h_final = measure_token(inner, next_token, cache)

        # Get next token
        out = h_final
        if hasattr(text_model, 'tie_word_embeddings') and text_model.tie_word_embeddings:
            logits_out = inner.embed_tokens.as_linear(out)
        elif hasattr(text_model, 'lm_head'):
            logits_out = text_model.lm_head(out)
        else:
            logits_out = inner.embed_tokens.as_linear(out)

        # Softcap
        softcap = getattr(text_model, 'final_logit_softcapping', None)
        if softcap:
            logits_out = mx.tanh(logits_out / softcap) * softcap

        next_token = mx.argmax(logits_out[:, -1:, :], axis=-1)
        mx.eval(next_token)

        token_str = tokenizer.decode(next_token.item())
        generated.append(token_str)

        elapsed = time.time() - t0

        # Accumulate
        token_count += 1
        for r in results:
            i = r["layer"]
            if not r["is_moe"]:
                continue
            if "cosine" in r:
                cosine_sums[i] += r["cosine"]
            if "overlap_top8" in r:
                overlap8_sums[i] += r["overlap_top8"]
                overlapk_sums[i] += r["overlap_topk"]
                overlap_counts[i] += 1
            if "overlap_a" in r:
                overlap_a_sums[i] += r["overlap_a"]
            if "overlap_b" in r:
                overlap_b_sums[i] += r["overlap_b"]

        # Print progress
        moe_r = [r for r in results if "overlap_a" in r]
        avg_c = sum(r.get("overlap_topk", 0) for r in moe_r) / max(len(moe_r), 1)
        avg_a = sum(r["overlap_a"] for r in moe_r) / max(len(moe_r), 1)
        avg_b = sum(r.get("overlap_b", 0) for r in moe_r) / max(len(moe_r), 1)

        sys.stdout.write(
            f"\r  tok {t+1:3d}/{args.num_tokens}  "
            f"C(full)={avg_c:.0%} A(no-attn)={avg_a:.0%} B(cheap)={avg_b:.0%}  "
            f"({elapsed:.1f}s)"
        )
        sys.stdout.flush()

    print("\n")

    # ── Report ──
    print("=" * 72)
    print(f"MoE Prediction Accuracy — {token_count} tokens")
    print(f"Method: h_approx = attention + dense_mlp (skip MoE) → next layer router")
    print("=" * 72)
    print()
    hdr = (f"{'Layer':>6}  {'Type':>8}  {'Cosine':>8}"
           f"  {'C:full':>8}  {'A:no-attn':>10}  {'B:cheap':>9}")
    print(hdr)
    print("-" * len(hdr))

    total_c = total_a = total_b = 0.0
    total_n = 0
    total_cosine = 0.0

    for i in moe_layers:
        avg_cos = cosine_sums[i] / token_count if token_count > 0 else 0
        total_cosine += avg_cos

        if overlap_counts[i] > 0:
            oc = overlapk_sums[i] / overlap_counts[i]
            oa = overlap_a_sums[i] / overlap_counts[i]
            ob = overlap_b_sums[i] / overlap_counts[i]
            total_c += oc
            total_a += oa
            total_b += ob
            total_n += 1
            sc, sa, sb = f"{oc:.0%}", f"{oa:.0%}", f"{ob:.0%}"
        else:
            sc = sa = sb = "—"

        lt = inner.layers[i].layer_type[:4]
        print(f"  {i:4d}  {lt:>8}  {avg_cos:>8.4f}  {sc:>8}  {sa:>10}  {sb:>9}")

    print("-" * len(hdr))
    if total_n > 0:
        mc = total_c / total_n
        ma = total_a / total_n
        mb = total_b / total_n
        mcos = total_cosine / len(moe_layers)
        print(f"  {'AVG':>4}  {'':>8}  {mcos:>8.4f}  {mc:>7.0%}  {ma:>9.0%}  {mb:>8.0%}")
        print()
        print(f"  Level C (MLP + attn + router, ~1ms/layer):   {mc:.1%} top-{PREDICT_K} overlap")
        print(f"  Level A (MLP + router, ~0.5ms/layer):        {ma:.1%} top-{PREDICT_K} overlap")
        print(f"  Level B (router only, ~0.3ms/layer):         {mb:.1%} top-{PREDICT_K} overlap")
        print(f"  Baseline co-occurrence:                      50.5% top-{PREDICT_K} overlap")
        print()
        best = max((mc, "C"), (ma, "A"), (mb, "B"), key=lambda x: (x[0] >= 0.80, -{"C":3,"A":2,"B":1}[x[1]]))
        if ma >= 0.80:
            print(f"  → Level A (skip attention) looks sufficient at {ma:.0%}.")
            print(f"    Cost: dense MLP + router = ~0.5 ms/layer, ~15 ms/token.")
        elif mb >= 0.80:
            print(f"  → Level B (router only) is good enough at {mb:.0%}.")
            print(f"    Cost: just router = ~0.3 ms/layer, ~9 ms/token.")
        elif mc >= 0.80:
            print(f"  → Need full Level C for ≥80% accuracy.")

    print()
    print(f"Generated: {''.join(generated)}")


if __name__ == "__main__":
    main()
