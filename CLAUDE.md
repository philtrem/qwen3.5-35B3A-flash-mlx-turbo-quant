# CLAUDE.md

## Git

Do NOT add Claude as a co-author in commit messages.

## Context

Detailed implementation history, decisions, and performance data are in the auto-memory system. Check MEMORY.md before exploring the codebase — it likely has what you need.

## What this is

Flash-loading inference engine for Qwen3.5-35B-A3B (36.3 GB, 9-bit MLX) on Mac M4 base 16GB. All-Rust single binary via `mlx-rs`, on-demand expert loading via pread().

## Build

```bash
cargo build --release
```

## Run

```bash
# Split model (one-time, creates safetensors format)
./target/release/flash-qwen split \
  --model-path /Users/philtrem/.lmstudio/models/inferencerlabs/Qwen3.5-35B-A3B-MLX-9bit \
  --output-path ./split_model_st

# Generate (warm set auto-detected from split_model_st/warm_experts.json)
./target/release/flash-qwen generate \
  --model-path ./split_model_st \
  --tokenizer-path /Users/philtrem/.lmstudio/models/inferencerlabs/Qwen3.5-35B-A3B-MLX-9bit \
  --prompt "Hello" --max-tokens 256
```

## Architecture

### `src/`
- **main.rs** — CLI (clap): split + generate subcommands
- **model/** — Model/TextModel/DecoderLayer, GatedDeltaNet, Attention, SparseMoeBlock, RMSNorm, MLP
- **memory.rs** — ExpertMemoryManager: pread() expert extraction from safetensors files, warm set madvise via mmap
- **engine.rs** — generate() loop + nucleus sampling
- **perf.rs** — PerfStats: per-phase timing accumulator for decode analysis (routing eval, sort eval, layer eval, extract, CPU work)
- **ffi.rs** — gather_qmm FFI wrapper via mlx-sys
- **splitter.rs** — model splitter (original → resident + per-layer expert safetensors)
- Expert weights loaded on-demand: pread() → Vec → Array::from_raw_data → gather_qmm (~27 MB/layer)
- Per-layer eval barriers at CPU/GPU boundaries (argsort is CPU, gather_qmm is GPU)
- 120 eval barriers per decode token: 40 routing + 40 sort + 40 layer (GDN proj eval skipped during decode)

### Model
- Model type is `qwen3_5_moe` mapping to `mlx_lm.models.qwen3_5` (NOT `qwen3_next`)
- 40 layers: 30 linear-attention (GatedDeltaNet/ArraysCache) + 10 full-attention (Attention/KVCache), every 4th layer is full-attention
- Weights are already sanitized (mlx-sanitized: 0.30.7) — do NOT re-sanitize
- Expert dimensions: hidden=2048, intermediate=512, 256 experts/layer, top_k=8, 8-bit quant, group_size=32

## Performance

- **Short prompt**: 1.9 tok/s decode (30 tokens), 1.6 tok/s prefill
- **Long prompt (programming)**: 1.3 tok/s decode steady-state (512 tokens)
- No swap storms. Peak expert memory ~27 MB per layer.
- Expert extraction uses pread() instead of mmap demand-paging (3.6× faster — eliminates page fault overhead)

### Current decode breakdown (short prompt, 519ms/tok):
- Extract experts (pread I/O): 396ms (76%) ← dominant bottleneck
- MoE routing eval: 57ms (11%)
- Layer eval (gather_qmm): 43ms (8%)
- MoE sort eval: 24ms (5%)

### Next optimization: ECB format + parallel I/O
See `ecb-optimization-plan.md` for the plan to reduce extract from ~400ms to ~40-80ms.

## Key gotchas

### MLX / mlx-rs
- `mx.linalg.qr` requires `stream=mx.cpu` — not supported on GPU
- MLX has no `searchsorted` — use boundary comparison: `(x[..., None] > boundaries).sum(-1)`
- `mlx_array_new_data` (and `Array::from_raw_data`) **copies** data — no zero-copy from mmap
- `Array::load_safetensors()` creates lazy arrays; when evaluated, reads file data into anonymous Metal buffers (NOT mmap-backed). Loading all 40 expert files (34.6 GB) causes swap storms on 16 GB.
- `gather_qmm` is NOT in mlx-rs — use `mlx_sys::mlx_gather_qmm` via FFI wrapper
- `argsort` runs on CPU; eval boundaries needed before GPU gather_qmm
- Activation dtype drift: bf16×f32 scalar promotes to f32 — cast scalars to input dtype

### Memory / UMA
- **Do NOT load all expert files via load_safetensors** — causes 25+ GB swap on 16 GB
- On-demand expert extraction via pread() is the correct approach (~27 MB per layer, not 864 MB)
- pread() is 3.6× faster than mmap demand-paging (page fault overhead: ~20μs/page × 55K pages/tok)
- madvise(MADV_WILLNEED) for warm set prefetch via mmap (kept alongside pread File handles), NOT mlock
- Per-layer eval ensures expert arrays are freed after each layer (peak ~27 MB, not cumulative)
- Expert LRU caching (both MLX Array and raw byte) does NOT help — working set (~3200 experts) >> cache size (960) on 16 GB; cache just displaces page cache

