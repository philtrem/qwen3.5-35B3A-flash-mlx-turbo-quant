# CLAUDE.md

## Git

Do NOT add Claude as a co-author in commit messages.

## Context

Detailed implementation history, decisions, and performance data are in the auto-memory system. Check MEMORY.md before exploring the codebase — it likely has what you need.

## What this is

Flash-loading inference engine for Qwen3.5-35B-A3B (36.3 GB, 9-bit MLX) on Mac M4 base 16GB. All-Rust single binary via `mlx-rs`, on-demand expert loading via pread() or zero-copy mmap Metal buffers.

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
- **memory.rs** — ExpertMemoryManager: pread + zero-copy extraction, warm set madvise, async I/O prefetch thread, warm/cold partitioning
- **engine.rs** — generate() loop + nucleus sampling
- **perf.rs** — PerfStats: per-phase timing accumulator for decode analysis
- **ffi.rs** — gather_qmm FFI + `array_from_mmap` zero-copy wrapper
- **ffi_zerocopy.cpp** — C++ shim: MLX array from mmap via Metal `newBufferWithBytesNoCopy`
- **splitter.rs** — model splitter (original → resident + per-layer expert safetensors/ECB)
- **build.rs** — compiles ffi_zerocopy.cpp with MLX C++ headers
- Two compute paths (toggle `USE_ZEROCOPY` in moe.rs):
  - **pread path**: pread → scatter → from_raw_data → gather_qmm (~1.2 tok/s)
  - **zero-copy path**: mmap → Metal buffer → per-expert quantized_matmul (~2.7 tok/s, verified coherent)
- Per-layer eval barriers at CPU/GPU boundaries (argsort is CPU, matmul is GPU)

### Model
- Model type is `qwen3_5_moe` mapping to `mlx_lm.models.qwen3_5` (NOT `qwen3_next`)
- 40 layers: 30 linear-attention (GatedDeltaNet/ArraysCache) + 10 full-attention (Attention/KVCache), every 4th layer is full-attention
- Weights are already sanitized (mlx-sanitized: 0.30.7) — do NOT re-sanitize
- Expert dimensions: hidden=2048, intermediate=512, 256 experts/layer, top_k=8, 8-bit quant, group_size=32

## Performance

### Zero-copy + async I/O thread (current best):
- **100 tokens**: 3.5 tok/s average decode, intervals 3.0-4.1 tok/s (warm cache)
- Decode breakdown (271ms/tok): layer eval 191ms (70%), extract 35ms (13%), routing eval 46ms (17%)
- Bottleneck: page fault latency for cold experts during quantized_matmul
- Async I/O thread pre-loads ~58% of cold experts via sequential pread (2.4 GB/s)
- Remaining cold experts fault at ~917 MB/s effective (macOS readahead helps)

### Without I/O thread (baseline zero-copy):
- **100 tokens**: 2.1 tok/s decode
- Decode breakdown (458ms/tok): layer eval 381ms (83%), extract 26ms (6%), routing eval 50ms (11%)

### pread path (USE_ZEROCOPY=false):
- **50 tokens**: ~1.2 tok/s decode (verified coherent)
- Decode breakdown (~773ms/tok): extract 661ms (86%), routing eval 52ms, layer eval 37ms, sort eval 23ms

### SSD benchmarks (M4 Mac Mini base):
- Sequential read (cold): 2.4 GB/s
- Random 16 KB read (cold): 153 MB/s (107μs per read)
- Gap: 16× — converting page faults to sequential pread is the key optimization

### General:
- No swap storms. Peak expert memory ~27 MB per layer.
- Warm set hit rate: ~63% (static, from 14-prompt profiling run)
- Expert reuse: 46% token-to-token overlap, 62% at k=5 window
- **Warm set page residency: only ~6.2 GB of 10.1 GB actually in page cache** (madvise is unreliable)

## Key gotchas

### MLX / mlx-rs
- `mx.linalg.qr` requires `stream=mx.cpu` — not supported on GPU
- MLX has no `searchsorted` — use boundary comparison: `(x[..., None] > boundaries).sum(-1)`
- `mlx_array_new_data` (and `Array::from_raw_data`) **copies** data — use `ffi_zerocopy.cpp` shim for zero-copy via `newBufferWithBytesNoCopy`
- `Array::load_safetensors()` creates lazy arrays; when evaluated, reads file data into anonymous Metal buffers (NOT mmap-backed). Loading all 40 expert files (34.6 GB) causes swap storms on 16 GB.
- `gather_qmm` is NOT in mlx-rs — use `mlx_sys::mlx_gather_qmm` via FFI wrapper
- `argsort` runs on CPU; eval boundaries needed before GPU gather_qmm
- Activation dtype drift: bf16×f32 scalar promotes to f32 — cast scalars to input dtype

### Memory / UMA
- **Do NOT load all expert files via load_safetensors** — causes 25+ GB swap on 16 GB
- On-demand expert extraction via pread() is the correct approach (~27 MB per layer, not 864 MB)
- pread() is 3.6× faster than mmap demand-paging (page fault overhead: ~107μs/page for cold 16 KB random reads)
- madvise(MADV_WILLNEED) for warm set prefetch via mmap — **unreliable**: only ~60% of advised pages are actually resident. mlock fixes this but breaks inference at 10 GB (too much wired memory for Metal)
- **Async I/O prefetch thread** converts cold-expert page faults (153 MB/s) to sequential pread (2.4 GB/s) — 67% speedup. Cold-first ordering maximizes lead time. Fire-and-forget (non-blocking) is correct; halting GPU to wait is WORSE (page faults overlap with GPU compute naturally)
- madvise from I/O thread does NOT work — it returns before pages are loaded (non-blocking hint). pread blocks until pages are resident, which IS the guarantee we need
- Speculative multi-layer lookahead from I/O thread HURTS — SSD contention between speculative reads and actual page faults
- Per-layer eval ensures expert arrays are freed after each layer (peak ~27 MB, not cumulative)
- Expert LRU caching (both MLX Array and raw byte) does NOT help — working set (~3200 experts) >> cache size (960) on 16 GB; cache just displaces page cache
- LZ4/Zstd compression does NOT help — decompression (4 GB/s) can't outrun SSD (2.5 GB/s) at realistic ratios
- Windowing/caching on UMA is net-zero — explicit cache displaces page cache, same total memory
- Zero-copy + stack() is WORSE than pread — stack triggers mass page faults (1.1 tok/s), destroying I/O-compute overlap
- Reducing dispatch count does NOT help — layer eval (~280-350ms) is page-fault-dominated, not dispatch-dominated
- Zero-copy per-expert scoring must use per-position weights during prefill (seq_len>1); scalar weighting corrupts output

