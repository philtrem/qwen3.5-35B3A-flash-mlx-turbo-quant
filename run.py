#!/usr/bin/env python3
"""CLI for flash-loaded Qwen3.5-35B-A3B inference."""

import argparse
import json
import sys
from pathlib import Path


def cmd_split(args):
    from flash_qwen.split import main as split_main
    sys.argv = ["split", "--model-path", args.model_path, "--output-path", args.output_path]
    split_main()


def cmd_profile_routing(args):
    from flash_qwen.engine import FlashInferenceEngine
    from flash_qwen.profile_routing import run_profiling, save_profile, PROFILING_PROMPTS

    engine = FlashInferenceEngine(
        split_model_path=args.model_path,
        cache_size_mb=args.cache_size_mb,
        original_model_path=args.tokenizer_path,
        use_mmap=getattr(args, 'use_mmap', False),
    )

    print(f"Profiling routing across {len(PROFILING_PROMPTS)} prompts, {args.max_tokens} tokens each...")
    counts, total_tokens = run_profiling(engine, max_tokens=args.max_tokens)

    output = args.output or str(Path(args.model_path) / "routing_profile.json")
    save_profile(counts, total_tokens, output)


def cmd_build_warm_set(args):
    from flash_qwen.profile_routing import load_profile, build_warm_set, safe_cache_budget_mb

    counts = load_profile(args.profile)
    budget = args.budget_mb or safe_cache_budget_mb()
    print(f"Memory budget: {budget} MB")

    warm = build_warm_set(counts, budget)
    output = args.output or "warm_experts.json"
    Path(output).write_text(json.dumps(warm, indent=2))
    print(f"Saved warm set to {output}")


def cmd_generate(args):
    from flash_qwen.engine import FlashInferenceEngine

    engine = FlashInferenceEngine(
        split_model_path=args.model_path,
        cache_size_mb=args.cache_size_mb,
        original_model_path=args.tokenizer_path,
        kv_bits=args.kv_bits,
        warm_experts=args.warm_experts,
        use_mmap=args.use_mmap,
    )

    if args.interactive:
        print("\nInteractive mode (type 'quit' to exit)")
        print("-" * 50)
        while True:
            try:
                prompt = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue

            # Apply chat template if available
            if hasattr(engine.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                formatted = engine.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted = prompt

            output = engine.generate(
                formatted,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"\nAssistant: {output}")
    else:
        prompt = args.prompt
        if not prompt:
            print("Error: --prompt required in non-interactive mode", file=sys.stderr)
            sys.exit(1)

        if hasattr(engine.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = engine.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        output = engine.generate(
            formatted,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(output)


def main():
    parser = argparse.ArgumentParser(
        description="Flash-loaded Qwen3.5-35B-A3B inference"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Split command
    split_parser = subparsers.add_parser("split", help="Split model into resident + expert files")
    split_parser.add_argument("--model-path", required=True, help="Original MLX model path")
    split_parser.add_argument("--output-path", required=True, help="Output directory")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--model-path", required=True, help="Path to split model")
    gen_parser.add_argument("--tokenizer-path", help="Path to tokenizer (if different from model)")
    gen_parser.add_argument("--prompt", help="Input prompt")
    gen_parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    gen_parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    gen_parser.add_argument("--cache-size-mb", type=int, default=6144, help="Expert LRU cache size in MB")
    gen_parser.add_argument("--kv-bits", type=int, choices=[2, 3, 4], default=None,
                            help="TurboQuant KV cache compression bits (default: None = FP16)")
    gen_parser.add_argument("--warm-experts", type=str, default=None,
                            help="Path to warm_experts.json (auto-detects in model dir if not set)")
    gen_parser.add_argument("--use-mmap", action="store_true",
                            help="Use mmap backend instead of pread+F_NOCACHE")

    # Profile routing command
    prof_parser = subparsers.add_parser("profile-routing", help="Profile expert routing across diverse prompts")
    prof_parser.add_argument("--model-path", required=True, help="Path to split model")
    prof_parser.add_argument("--tokenizer-path", help="Path to tokenizer")
    prof_parser.add_argument("--max-tokens", type=int, default=128, help="Tokens per prompt")
    prof_parser.add_argument("--cache-size-mb", type=int, default=6144, help="Expert LRU cache size in MB")
    prof_parser.add_argument("--output", type=str, default=None, help="Output path for routing_profile.json")
    prof_parser.add_argument("--use-mmap", action="store_true",
                            help="Use mmap backend instead of pread+F_NOCACHE")

    # Build warm set command
    warm_parser = subparsers.add_parser("build-warm-set", help="Build warm expert set from routing profile")
    warm_parser.add_argument("--profile", required=True, help="Path to routing_profile.json")
    warm_parser.add_argument("--budget-mb", type=int, default=None, help="Memory budget in MB (default: auto)")
    warm_parser.add_argument("--output", type=str, default=None, help="Output path for warm_experts.json")

    args = parser.parse_args()

    if args.command == "split":
        cmd_split(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "profile-routing":
        cmd_profile_routing(args)
    elif args.command == "build-warm-set":
        cmd_build_warm_set(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
