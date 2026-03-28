"""Expert routing profiler and warm set generator.

Usage:
    # Profile routing across diverse prompts
    python run.py profile-routing --model-path ./split_model --tokenizer-path ... [--max-tokens 128]

    # Build warm set from profile data
    python run.py build-warm-set --profile routing_profile.json [--budget-mb 6144]
"""

import json
import os
from pathlib import Path

# Profiling prompts covering user's primary workloads
PROFILING_PROMPTS = {
    "code_python": "Write a Python async HTTP server with graceful shutdown and structured error handling.",
    "code_rust": "Implement a lock-free concurrent hash map in Rust with linear probing.",
    "code_react": "Create a React component with TypeScript that implements infinite scroll with virtualization.",
    "code_csharp": "Write a C# REST API controller with Entity Framework Core, including pagination and filtering.",
    "code_systems": "Write a memory allocator in C that uses a buddy system with coalescing.",
    "debug_oom": "Explain this error and suggest fixes:\n```\nRuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 15.78 GiB total capacity; 13.52 GiB already allocated)\n```",
    "debug_segfault": "Debug this segfault:\n```\nThread 1 received signal SIGSEGV, Segmentation fault.\n0x0000555555555189 in insert (head=0x0, val=42) at list.c:15\n15\t    while (curr->next != NULL)\n```",
    "debug_csharp": "Why does this C# async method deadlock?\n```csharp\npublic string GetData() {\n    return GetDataAsync().Result;\n}\nprivate async Task<string> GetDataAsync() {\n    var data = await httpClient.GetStringAsync(url);\n    return data;\n}\n```",
    "debug_stacktrace": "Explain this Python traceback:\n```\nTraceback (most recent call last):\n  File \"app.py\", line 42, in handle_request\n    result = await process(data)\n  File \"app.py\", line 67, in process\n    return json.loads(raw)\njson.decoder.JSONDecodeError: Expecting ',' delimiter: line 1 column 42\n```",
    "tool_call_json": 'Generate a JSON function call to search the web:\n{"function": "web_search", "parameters": {"query": "...", "num_results": 5}}\nSearch for: best practices for MLX model optimization on Apple Silicon',
    "tool_call_structured": "Generate a structured tool call response:\n1. Read the file /src/main.rs\n2. Find all functions that allocate memory\n3. Return the results as JSON with function name, line number, and allocation size",
    "web_search_html": "Summarize the key points from this documentation excerpt:\n<div class='api-docs'><h2>mx.eval()</h2><p>Evaluate an array or list of arrays. MLX uses lazy evaluation by default, meaning operations are not computed until their results are needed. The eval function forces immediate evaluation.</p><h3>Parameters</h3><ul><li>args: arrays to evaluate</li></ul></div>",
    "web_search_docs": "Based on this search result, explain how to use mmap in Rust:\n\nFrom docs.rs/memmap2:\n```\nuse memmap2::MmapOptions;\nuse std::fs::File;\n\nlet file = File::open(\"data.bin\")?;\nlet mmap = unsafe { MmapOptions::new().map(&file)? };\nassert_eq!(b\"Hello\", &mmap[0..5]);\n```\nMemory-mapped files allow treating file contents as a byte slice...",
}


def run_profiling(engine, max_tokens: int = 128, verbose: bool = True):
    """Run all profiling prompts and return routing counts."""
    from flash_qwen.model import (
        enable_routing_recording, disable_routing_recording,
        get_routing_counts, reset_routing_counts,
    )
    reset_routing_counts()
    enable_routing_recording()

    total_tokens = 0
    for name, prompt in PROFILING_PROMPTS.items():
        if hasattr(engine.tokenizer, "apply_chat_template"):
            formatted = engine.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            formatted = prompt

        if verbose:
            print(f"  [{name}] generating {max_tokens} tokens...", end="", flush=True)
        output = engine.generate(formatted, max_tokens=max_tokens, temperature=0.7, verbose=False)
        tokens = len(output.split())  # approximate
        total_tokens += tokens
        if verbose:
            print(f" done (~{tokens} words)")

    disable_routing_recording()
    counts = get_routing_counts()
    if verbose:
        unique_experts = len(counts)
        total_activations = sum(counts.values())
        print(f"\nRouting profile: {unique_experts} unique experts, {total_activations} total activations")

    return counts, total_tokens


def save_profile(counts: dict, total_tokens: int, output_path: str):
    """Save routing profile to JSON."""
    # Convert tuple keys to lists for JSON
    data = {
        "total_tokens": total_tokens,
        "num_prompts": len(PROFILING_PROMPTS),
        "experts": {f"{layer},{expert}": count for (layer, expert), count in counts.items()},
    }
    Path(output_path).write_text(json.dumps(data, indent=2))
    print(f"Saved routing profile to {output_path}")


def load_profile(profile_path: str) -> dict:
    """Load routing profile, return {(layer, expert): count}."""
    data = json.loads(Path(profile_path).read_text())
    counts = {}
    for key, count in data["experts"].items():
        layer, expert = key.split(",")
        counts[(int(layer), int(expert))] = count
    return counts


def build_warm_set(counts: dict, budget_mb: int, expert_stride: int = 3_538_944) -> dict:
    """Build warm set from routing profile.

    Returns dict suitable for writing to warm_experts.json.
    """
    budget_bytes = budget_mb * 1024 * 1024
    max_experts = budget_bytes // expert_stride

    # Rank by frequency
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    total_activations = sum(counts.values())

    selected = []
    covered = 0
    for (layer, expert), count in ranked:
        if len(selected) >= max_experts:
            break
        selected.append([layer, expert])
        covered += count

    coverage = covered / total_activations if total_activations > 0 else 0
    actual_mb = len(selected) * expert_stride // (1024 * 1024)

    result = {
        "budget_mb": actual_mb,
        "num_experts": len(selected),
        "experts": selected,
        "coverage": round(coverage, 4),
    }

    print(f"Warm set: {len(selected)} experts ({actual_mb} MB), coverage: {coverage:.1%}")
    return result


def safe_cache_budget_mb() -> int:
    """Compute max expert cache that won't cause swap on this machine."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical = pages * page_size
    except (ValueError, OSError):
        physical = 16 * 1024**3  # fallback: 16 GB
    reserved = int((2.57 + 0.5 + 3.5) * 1024**3)  # resident + model + OS
    available = physical - reserved
    return max(1024, available // (1024 * 1024))
