"""
Collect pre-attention hidden states and router decisions from Gemma4 26B-A4B.
Run on a cloud GPU instance with >= 24GB VRAM (A10G with 4-bit, or A100 for bf16).

Usage:
    pip install transformers torch bitsandbytes accelerate datasets
    python collect_routing_data.py --output routing_data.npz --num-tokens 10000

Outputs:
    routing_data.npz containing per-layer:
        layer_{i}_hidden: [N, 2816] float16 - pre-attention hidden states
        layer_{i}_experts: [N, 8] int16 - selected expert indices
    Also prints zero-training accuracy (existing router on pre-attention state).
"""

import argparse
import numpy as np
import torch
from pathlib import Path


def find_router_modules(model):
    """Find router modules in the model, handling different HF naming conventions."""
    routers = {}
    for name, module in model.named_modules():
        # Match patterns: layers.N.router, layers.N.block_sparse_moe.gate, etc.
        if name.endswith(".router") and "layers." in name:
            layer_idx = int(name.split("layers.")[1].split(".")[0])
            routers[layer_idx] = (name, module)
        elif name.endswith(".gate") and "block_sparse_moe" in name:
            layer_idx = int(name.split("layers.")[1].split(".")[0])
            routers[layer_idx] = (name, module)
    return routers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-26b-a4b-it",
                        help="HuggingFace model ID")
    parser.add_argument("--output", default="routing_data.npz",
                        help="Output file path")
    parser.add_argument("--num-tokens", type=int, default=10000,
                        help="Number of tokens to collect")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Sequence length per forward pass")
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                        help="Load model in 4-bit (for 24GB GPUs)")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Load in bf16 (needs 52GB+ VRAM)")
    parser.add_argument("--dataset", default="wikitext",
                        help="Dataset to use (wikitext or a text file path)")
    args = parser.parse_args()

    use_4bit = args.load_in_4bit and not args.no_4bit

    # Load model
    print(f"Loading {args.model} ({'4-bit' if use_4bit else 'bf16'})...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    # Find model structure
    # Try common paths for the decoder layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise RuntimeError("Cannot find decoder layers. Check model structure with print(model)")

    num_layers = len(layers)
    print(f"Found {num_layers} decoder layers")

    # Find routers
    routers = find_router_modules(model)
    if not routers:
        # Fallback: check layer attributes directly
        for i, layer in enumerate(layers):
            if hasattr(layer, "router"):
                routers[i] = (f"layers.{i}.router", layer.router)
            elif hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "gate"):
                routers[i] = (f"layers.{i}.block_sparse_moe.gate", layer.block_sparse_moe.gate)

    moe_layers = sorted(routers.keys())
    print(f"Found {len(moe_layers)} MoE layers: {moe_layers[:5]}...{moe_layers[-3:]}")

    # Data storage
    collected = {i: {"hidden": [], "experts": []} for i in moe_layers}
    total_tokens = 0

    # Register hooks
    hooks = []

    # Hook on each MoE layer to capture input (pre-attention state)
    def make_layer_hook(layer_idx):
        def hook(module, args, output):
            # The input to the layer is the pre-attention hidden state
            x = args[0] if isinstance(args, tuple) else args
            if isinstance(x, torch.Tensor):
                collected[layer_idx]["_layer_input"] = x.detach().float().cpu()
        return hook

    # Hook on router to capture routing decisions
    def make_router_hook(layer_idx):
        def hook(module, args, output):
            # Router output: (top_k_indices, top_k_weights) or just indices
            if isinstance(output, tuple):
                indices = output[0]
            else:
                indices = output
            indices = indices.detach().cpu().to(torch.int16)

            # Get the saved layer input
            if "_layer_input" in collected[layer_idx]:
                hidden = collected[layer_idx].pop("_layer_input")
                # hidden shape: [B, S, D], indices shape: [B, S, K]
                # Flatten batch and sequence dims
                B, S, D = hidden.shape
                hidden_flat = hidden.reshape(-1, D).to(torch.float16)
                indices_flat = indices.reshape(-1, indices.shape[-1])
                collected[layer_idx]["hidden"].append(hidden_flat.numpy())
                collected[layer_idx]["experts"].append(indices_flat.numpy())
        return hook

    for i in moe_layers:
        hooks.append(layers[i].register_forward_hook(make_layer_hook(i)))
        _, router_module = routers[i]
        hooks.append(router_module.register_forward_hook(make_router_hook(i)))

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if Path(args.dataset).exists():
        with open(args.dataset) as f:
            text = f.read()
        all_ids = tokenizer.encode(text)
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        text = "\n".join([t for t in ds["text"] if len(t) > 100][:500])
        all_ids = tokenizer.encode(text)

    print(f"Dataset: {len(all_ids)} tokens available, collecting {args.num_tokens}")

    # Process in chunks
    with torch.no_grad():
        offset = 0
        while total_tokens < args.num_tokens and offset < len(all_ids):
            chunk_size = min(args.batch_size, len(all_ids) - offset, args.num_tokens - total_tokens)
            input_ids = torch.tensor([all_ids[offset:offset + chunk_size]], device=model.device)

            model(input_ids)

            offset += chunk_size
            total_tokens += chunk_size

            if total_tokens % 2000 < chunk_size:
                print(f"  {total_tokens}/{args.num_tokens} tokens collected")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate and save
    print(f"\nCollected {total_tokens} tokens across {len(moe_layers)} MoE layers")

    save_dict = {"num_layers": num_layers, "moe_layers": np.array(moe_layers)}
    for i in moe_layers:
        if collected[i]["hidden"]:
            hidden = np.concatenate(collected[i]["hidden"], axis=0)
            experts = np.concatenate(collected[i]["experts"], axis=0)
            save_dict[f"layer_{i}_hidden"] = hidden
            save_dict[f"layer_{i}_experts"] = experts
            print(f"  Layer {i}: {hidden.shape[0]} samples, hidden={hidden.shape[1]}, "
                  f"experts_per_token={experts.shape[1]}")

    np.savez_compressed(args.output, **save_dict)
    print(f"\nSaved to {args.output} ({Path(args.output).stat().st_size / 1e6:.1f} MB)")

    # Measure zero-training accuracy: predict same layer using router on layer input
    print("\n=== Zero-training accuracy (router on pre-attention state) ===")
    measure_zero_training_accuracy(model, layers, routers, collected, moe_layers)

    # Measure cross-layer accuracy
    print("\n=== Cross-layer accuracy (layer L input -> layer L+1 router) ===")
    measure_cross_layer_accuracy(model, layers, routers, collected, moe_layers)


def measure_zero_training_accuracy(model, layers, routers, collected, moe_layers):
    """Test: run each layer's router on the pre-attention state instead of post-attention."""
    for i in moe_layers:
        if not collected[i]["hidden"]:
            continue
        hidden = torch.tensor(np.concatenate(collected[i]["hidden"], axis=0),
                              dtype=torch.float32)
        actual = np.concatenate(collected[i]["experts"], axis=0)

        # Run router on pre-attention state
        _, router_module = routers[i]
        device = next(router_module.parameters()).device
        with torch.no_grad():
            # Process in batches to avoid OOM
            pred_indices = []
            for start in range(0, len(hidden), 256):
                batch = hidden[start:start+256].to(device).to(torch.bfloat16)
                batch = batch.unsqueeze(0)  # [1, N, D]
                top_k_indices, _ = router_module(batch)
                pred_indices.append(top_k_indices.squeeze(0).cpu().numpy())
            predicted = np.concatenate(pred_indices, axis=0)

        # Compute top-k overlap accuracy
        hits = 0
        total = 0
        for j in range(len(actual)):
            actual_set = set(actual[j].tolist())
            pred_set = set(predicted[j].tolist())
            hits += len(actual_set & pred_set)
            total += len(actual_set)

        acc = hits / total if total > 0 else 0
        print(f"  Layer {i:2d}: {acc:.1%} ({hits}/{total})")


def measure_cross_layer_accuracy(model, layers, routers, collected, moe_layers):
    """Test: use layer L's input with layer L+1's router to predict L+1's experts."""
    for idx in range(len(moe_layers) - 1):
        i = moe_layers[idx]
        j = moe_layers[idx + 1]

        if not collected[i]["hidden"] or not collected[j]["experts"]:
            continue

        hidden_i = torch.tensor(np.concatenate(collected[i]["hidden"], axis=0),
                                dtype=torch.float32)
        actual_j = np.concatenate(collected[j]["experts"], axis=0)

        # Use min samples (in case different counts)
        n = min(len(hidden_i), len(actual_j))
        hidden_i = hidden_i[:n]
        actual_j = actual_j[:n]

        # Run layer j's router on layer i's input
        _, router_j = routers[j]
        device = next(router_j.parameters()).device
        with torch.no_grad():
            pred_indices = []
            for start in range(0, n, 256):
                batch = hidden_i[start:start+256].to(device).to(torch.bfloat16)
                batch = batch.unsqueeze(0)
                top_k_indices, _ = router_j(batch)
                pred_indices.append(top_k_indices.squeeze(0).cpu().numpy())
            predicted = np.concatenate(pred_indices, axis=0)

        hits = 0
        total = 0
        for k in range(n):
            actual_set = set(actual_j[k].tolist())
            pred_set = set(predicted[k].tolist())
            hits += len(actual_set & pred_set)
            total += len(actual_set)

        acc = hits / total if total > 0 else 0
        print(f"  Layer {i:2d} -> {j:2d}: {acc:.1%} ({hits}/{total})")


if __name__ == "__main__":
    main()
