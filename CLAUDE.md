# CLAUDE.md

## Git

Do NOT add Claude as a co-author in commit messages.

## Context

Detailed implementation history, decisions, and performance data are in the auto-memory system. Check MEMORY.md before exploring the codebase — it likely has what you need.

## What this is

Flash-loading inference engine for Qwen3.5-35B-A3B (36.3 GB, 9-bit MLX) on Mac M4 base 16GB. Streams MoE expert weights from SSD, pins resident weights in RAM, with optional TurboQuant KV cache compression and warm set preloading.

## Build

```bash
source .venv/bin/activate  # Python 3.14 via Homebrew
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

## Run

```bash
# Split model (one-time)
python run.py split --model-path /Users/philtrem/.lmstudio/models/inferencerlabs/Qwen3.5-35B-A3B-MLX-9bit --output-path ./split_model

# Profile routing + build warm set (one-time, ~30 min)
python run.py profile-routing --model-path ./split_model --tokenizer-path /Users/philtrem/.lmstudio/models/inferencerlabs/Qwen3.5-35B-A3B-MLX-9bit
python run.py build-warm-set --profile ./split_model/routing_profile.json

# Generate (warm set auto-detected from split_model/warm_experts.json)
python run.py generate --model-path ./split_model \
  --tokenizer-path /Users/philtrem/.lmstudio/models/inferencerlabs/Qwen3.5-35B-A3B-MLX-9bit \
  --prompt "Hello" --max-tokens 256 --cache-size-mb 9656

# With TurboQuant KV cache (2, 3, or 4 bit)
python run.py generate ... --kv-bits 4

# With mmap backend (alternative to pread+F_NOCACHE)
python run.py generate ... --use-mmap
```

## Architecture

- **Rust (PyO3)**: `src/` — SSD expert loading (`pread`+`F_NOCACHE` or `mmap`), LRU cache with split mutex (GIL released during I/O), bulk warm set preloader with rayon, mlock, prefetch thread, model splitter
- **Python (MLX)**: `flash_qwen/` — model definition (async_eval + shared expert overlap), inference engine, TurboQuant KV cache, routing profiler + warm set generator
- Model type is `qwen3_5_moe` mapping to `mlx_lm.models.qwen3_5` (NOT `qwen3_next`)
- 40 layers: 30 linear-attention (GatedDeltaNet/ArraysCache) + 10 full-attention (Attention/KVCache), every 4th layer is full-attention
- Weights are already sanitized (mlx-sanitized: 0.30.7) — do NOT re-sanitize
- Expert dimensions: hidden=2048, intermediate=512, 256 experts/layer, top_k=8, 8-bit quant, group_size=32

## Performance

Best stable: **1.7 tok/s** with warm set (2861 experts, 9.6 GB mlocked, 80% hit rate). Bottleneck is Rust→PyBytes→numpy→mx.array copy chain (~27 MB × 40 layers/token).

## Key gotchas

- `mx.linalg.qr` requires `stream=mx.cpu` — not supported on GPU
- MLX has no `searchsorted` — use boundary comparison: `(x[..., None] > boundaries).sum(-1)`
- TurboQuantCache must NOT have a `bits` attribute — triggers quantized SDPA path in `mlx_lm/models/base.py:117`
- Weight prefix: split model has `model.layers.N...`, must re-add `language_model.` at load time
- **Do NOT dual-cache**: can't have both Rust byte cache + Python mx.array cache on 16 GB — causes swap storms
- **Rust-via-PyO3 calling MLX is slower than Python calling MLX** — don't move gather_qmm to Rust side
- LZ4 compression of experts gets ~1.5× ratio (tested) — `lz4_flex` in Cargo.toml but format not yet implemented
