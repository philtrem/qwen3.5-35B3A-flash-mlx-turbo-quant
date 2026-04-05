# CLAUDE.md

## Git

Do NOT add Claude as a co-author in commit messages.

## Context

Detailed implementation history, decisions, and performance data are in the auto-memory system. Check MEMORY.md before exploring the codebase — it likely has what you need.

## What this is

Flash-loading inference engine for sparse MoE models on Mac M4 base 16GB. All-Rust single binary via `mlx-rs`, on-demand expert loading via GCD prefetch + zero-copy mmap Metal buffers.

**Supported models:**
- **Gemma 4 26B-A4B** (default) — `gemma4` architecture, 30 layers, 128 experts, top_k=8. Uses **Unsloth UD-MLX-3bit** dynamic quantization (3-bit experts, 8-bit attention, 6-bit embedding).
- **Qwen 3.5 35B-A3B** — `qwen3_5_moe` architecture, 40 layers, 256 experts, top_k=8. Uniform 4-bit.

Model type is auto-detected from config.json (`layer_types` field present → Gemma4, otherwise Qwen). Quantization supports both uniform and per-component mixed-precision (Unsloth UD format).

## Build

```bash
cargo build --release
```

## Run

```bash
# Split model (one-time, creates ECB format)
./target/release/flash-moe split \
  --model-path /path/to/model \
  --output-path ./split_output

# Generate — Gemma 4 UD-3bit (default model path: ./split_gemma4)
./target/release/flash-moe generate \
  --model-path ./split_gemma4_ud3 \
  --tokenizer-path ./gemma4-ud-3bit \
  --prompt "Hello" --max-tokens 256

# Qwen model (must specify paths)
./target/release/flash-moe generate \
  --model-path ./split_model_st \
  --tokenizer-path /Users/philtrem/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt "Hello" --max-tokens 256

# Optional flags:
#   --no-speculate    Disable speculative prefetch for predicted experts
#   --warm-set        Load warm set at startup (preads frequent experts into page cache)
#   --kv-quant-bits 3 TurboQuant KV cache (3-bit, default)
#   --no-kv-quant     Disable KV cache quantization (plain bf16)
```

## Architecture

### `src/`
- **main.rs** — CLI (clap): split + generate subcommands. Reads quant config from config.json. Default model: `./split_gemma4`.
- **config.rs** — `TextModelArgs` with `ModelType` enum (Qwen/Gemma4). Auto-detects from config.json. `QuantizationConfig` supports per-component bit-width overrides (Unsloth UD format) with `bits_for()`/`group_size_for()` lookups.
- **model/** — Model/TextModel/DecoderLayer with MoeVariant (Qwen/Gemma4), plus:
  - **attention.rs** — Qwen full attention (output gating, partial RoPE)
  - **gemma4_attention.rs** — Gemma4 attention (no gating, K==V for full-attn, v_norm, scale=1.0, per-layer RoPE/head_dim). `forward_speculative()` for Level C: runs attention with virtual KV append (no cache mutation).
  - **gated_delta.rs** — Qwen GatedDeltaNet linear attention
  - **moe.rs** — SparseMoeBlock (Qwen: shared expert, SiLU) + Gemma4MoeBlock (router with norm+scale+per_expert_scale, GELU). TransitionProfiler with RouterWeightsRef for Level B CPU prediction.
  - **mlp.rs** — QuantizedLinear, MLP (SiLU), GeLUMLP (Gemma4)
  - **norm.rs** — RMSNorm, RMSNormGated, RMSNormNoScale (Gemma4 v_norm)
- **memory.rs** — ExpertMemoryManager: GCD prefetch (speculative/reactive with QoS + cancel), zero-copy mmap extraction, warm set pread, F_RDADVISE
- **engine.rs** — generate() loop + nucleus sampling
- **perf.rs** — PerfStats: per-phase timing accumulator (routing eval, layer eval, GPU wait, extract, routing CPU)
- **ffi.rs** — gather_qmm FFI + `array_from_mmap` zero-copy wrapper
- **ffi_zerocopy.cpp** — C++ shim: MLX array from mmap via Metal `newBufferWithBytesNoCopy`
- **splitter.rs** — model splitter (original → resident + per-layer expert ECB). Auto-detects three layouts: Qwen (switch_mlp), Gemma4 (experts.gate_up_proj, fused), Gemma4Ud (experts.switch_glu, unfused). Canonicalizes resident weight names across prefix conventions.
- **build.rs** — compiles ffi_zerocopy.cpp with MLX C++ headers

### I/O strategy (current default, USE_ZEROCOPY=true)
- **GCD reactive prefetch** (default): After routing eval determines actual experts, `prefetch_gcd_reactive()` dispatches F_RDADVISE + madvise(WILLNEED) + prefault touch per expert on GCD userInitiated queue. Blocks via dispatch_group until all pages resident. Then mmap zerocopy arrays are created — GPU eval is fault-free. Cancels any in-flight speculative workers first to avoid SSD contention. Gated by `NOREACTIVE=1` env var for A/B testing.
- **GCD speculative prefetch** (on by default, `--no-speculate` to disable): After `async_eval(h)` submits GPU, `prefetch_gcd_speculative()` fires low-priority (utility QoS) **prefault-only** (no F_RDADVISE/madvise — those can't be cancelled once issued) for L+1's predicted experts. Cancellable page-by-page via atomic flag — reactive cancels these when exact experts are known. Pages touched before cancellation remain in page cache.
- **Warm set pread at startup** (opt-in, `--warm-set`): `mlock_warm_set()` uses parallel pread to guarantee warm expert pages are resident.
- Per-layer eval via `async_eval` + `eval` separates GPU submission from wait time in perf stats.

### Models

#### Gemma 4 26B-A4B (default)
- Architecture: `gemma4` / `gemma4_text`
- 30 layers, all with MoE + dense MLP in parallel
- Attention: 24 sliding (head_dim=256, kv_heads=8, rope_theta=10K) + 6 full (global_head_dim=512, kv_heads=2, rope_theta=1M, K==V)
- Embedding: 6-bit quantized (UD) or bf16 (old), scaled by √hidden_size. Tied word embeddings via quantized_matmul (or matmul if bf16).
- MoE: 128 experts, top_k=8, GELU activation. Router: RMSNormNoScale → scale → proj → per_expert_scale.
- Dense MLP: GELU, runs in parallel with MoE every layer. Output = dense + expert.
- Extra norms: pre/post_feedforward_layernorm, pre_feedforward_layernorm_2, post_feedforward_layernorm_{1,2}
- Layer scalar: per-layer learned scalar applied at end of decoder layer
- Logit softcapping: tanh(logits/30) × 30
- **UD-3bit (current, group_size=64)**: Mixed precision — 3-bit experts (2.6 MB each), 8-bit attention, 3-5 bit dense MLP (varies per layer), 3-bit router, 6-bit embedding. ECB file ~318 MB/layer, 8 active = ~20.8 MB/layer. 5-bit weights re-quantized to 4-bit at load time (MLX has no 5-bit Metal kernel; CPU dequantize → GPU quantize, affects 3 weights on layer 29 only).
- **Uniform 4-bit (old, group_size=64)**: per_expert_stride ~3.35 MB, ECB file ~428 MB/layer, 8 active = ~26.8 MB/layer.
- Weight naming: UD uses `scales`/`biases`, mlx-community uses `weight_scales`/`weight_biases`. Handled by `load_qlinear_flex()`.
- Source: `unsloth/gemma-4-26b-a4b-it-UD-MLX-3bit` (UD-3bit), `mlx-community/gemma-4-26b-a4b-it-4bit` (old)
- Split output: `./split_gemma4_ud3` (11 GB), `./split_gemma4` (13 GB, old)

#### Qwen 3.5 35B-A3B
- Architecture: `qwen3_5_moe` mapping to `mlx_lm.models.qwen3_5` (NOT `qwen3_next`)
- 40 layers: 30 linear-attention (GatedDeltaNet/ArraysCache) + 10 full-attention (Attention/KVCache), every 4th layer is full-attention
- MoE: 256 experts, top_k=8, SiLU activation. Shared expert + shared expert gate per layer.
- Attention: output gating (sigmoid gate), partial RoPE
- Quantization read from config.json (bits + group_size). Supports 4-bit and 8-bit.
- **4-bit (group_size=64)**: per_expert_stride ~1.77 MB, ECB file ~453 MB/layer, 8 active = ~14.2 MB/layer
- Source: `mlx-community/Qwen3.5-35B-A3B-4bit`
- Split output: `./split_model_st` (19 GB)
- Warm set backup: `~/warm_experts_qwen35_35b_a3b.json`

### Per-token I/O comparison (top_k=8)
| Model | Quant | Expert size | 8 active/layer | MoE layers | **I/O per token** |
|-------|-------|------------|----------------|------------|------------------|
| **Gemma4 26B (UD-3bit)** | 3-bit | 2.60 MB | 20.8 MB | 30 | **624 MB** |
| Gemma4 26B (old 4-bit) | 4-bit | 3.35 MB | 26.8 MB | 30 | 803 MB |
| Qwen 3.5 35B | 4-bit | 1.77 MB | 14.2 MB | 40 | **566 MB** |

UD-3bit reduced Gemma4 expert I/O by 22% (803→624 MB/tok). Smaller experts also fit better in page cache, amplifying the benefit beyond raw I/O reduction.

- 12 CPU threads available (M4)

## Performance

### Gemma 4 26B-A4B UD-3bit (default model, M4 16GB):
- **256 tokens**: **6.5 tok/s** (implied 7.1), peaking at 8.2
- **80 tokens**: **6.2 tok/s** (implied 6.7), peaking at 7.4
- Decode breakdown (142ms/tok @256): routing CPU 36ms (25%), layer eval 57ms (40%), routing eval 30ms (21%), extract 19ms (13%)
- GPU wait: 0.9ms (speculative prediction fills the idle eval window)
- I/O-bound: 624 MB/token (3-bit experts are 2.60 MB each)
- Level B prediction: **71.4% overall** (top-12). Per-layer range: 51% (layer 9) to 89% (layer 28).
- Resident weights: 1.90 GB (smaller than old 4-bit's 3.56 GB, freeing RAM for page cache)

### Gemma 4 26B-A4B uniform 4-bit (old, M4 16GB):
- **80 tokens**: **3.1 tok/s** (implied 4.1), peaking at 3.4
- Decode breakdown (243ms/tok): routing CPU 147ms (61%), layer eval 57ms (23%), routing eval 24ms (10%), extract 16ms (6%)
- I/O-bound: 803 MB/token (experts are 3.35 MB each, 2× bigger than Qwen)
- Level C prediction: **73% overall** (53%–90% per-layer). Dense MLP + speculative attention + next-layer router on h_post_attn.
- **UD-3bit is 2× faster**: 22% less I/O + 47% less resident RAM = 70% less routing CPU time.

### Qwen 3.5 35B-A3B 4-bit (M4 16GB):
- **50 tokens**: **7.7 tok/s** (implied 8.1), peaking at 8.7
- Decode breakdown (123ms/tok): routing CPU/GCD prefetch 48ms (39%), routing eval 34ms (28%), layer eval 32ms (26%), extract 10ms (8%)
- GPU wait: 27ms (pure compute, zero faults)
- GCD reactive prefault ensures fault-free eval. Speculative warms pages before reactive runs.
- With `--warm-set`: 7.0 tok/s (warm set preloads 63% of experts, but GCD speculative provides similar benefit)
- With `--kv-quant-bits 3`: ~5.3 tok/s (Hadamard rotation adds ~12ms/tok on 10 attention layers)

### Qwen 4-bit benchmark matrix (50 tokens, M4 16GB):
| Config | tok/s | Routing CPU ms/tok | Layer eval ms/tok | GPU wait ms/tok |
|--------|-------|--------------------|-------------------|-----------------|
| **GCD reactive + speculative (default)** | **7.7→8.7** | **47.5** | **31.7** | **26.6** |
| GCD reactive + speculative + warm set | 7.0→7.9 | 58.7 | 32.5 | 28.9 |
| pread reactive + warm set (old default) | 7.2→7.9 | 56.8 | 30.9 | 16.1 |
| pread reactive, no warm set | 6.9 | 62.5 | 31.6 | 26.3 |
| GCD reactive + speculative (no cancel) | 6.3 | 72.3 | 34.0 | 28.9 |
| Old pipeline (speculative hybrid pread) | 4.8 | 31.8 | 111.4 | 0.0 |
| Warm set only (no reactive) | 5.3→6.6 | 0.1 | 137.3 | 132.5 |

### GCD speculative + cancel: why it works on 4-bit
- **Prefault-only for speculative** (2026-04-04): speculative workers do ONLY prefault touch (one byte per 16KB page). F_RDADVISE and madvise are skipped for speculative because they issue kernel-level I/O that **cannot be cancelled** — causing SSD contention with reactive. Prefault touch is cancellable page-by-page via atomic generation counter.
- **Reactive uses full pipeline**: F_RDADVISE + madvise(WILLNEED) + prefault. Only speculative is restricted.
- Pages touched by speculative before cancellation remain in page cache, reducing reactive's work.
- Without cancel: 6.3 tok/s (SSD contention). With cancel: 7.7 tok/s (clean handoff) for Qwen.
- GCD QoS (utility vs userInitiated) provides OS-level thread priority differentiation.
- **Gemma4 UD-3bit speculative**: 6.5 tok/s @256 tokens with TurboQuant. Speculation adds zero overhead after prefault-only fix.
- **Gemma4 old 4-bit speculative with Level C prediction**: 3.6–3.8 tok/s (historical).

### Why pread-based speculative failed on 4-bit (historical)
- Experts are 1.69 MB (small) — page cache retains them between tokens
- GPU eval is ~0.65ms/layer — too short for blocking speculative I/O to overlap
- Blocking pread for speculative can't be cancelled — always contends with reactive
- Background rayon threads are 2.3× slower than main-thread on macOS

### 8-bit historical (Qwen3.5-35B-A3B-MLX-9bit, now deleted):
- Default mmap zerocopy: 3.8 tok/s (warm cache), 1.8-2.0 tok/s (cold/degraded)
- Reactive pread + no warm set: 2.8 tok/s (8-bit experts too large for page cache on 16 GB)
- async_eval pipeline: 1.8-2.1 tok/s (pread throughput bottleneck)
- Pure GPU compute: ~57ms/tok (fault-free eval)

### SSD benchmarks (M4 Mac Mini base):
- Actual SSD read rate: ~3 GB/s
- Random 16 KB read (cold): 153 MB/s (107μs per read)
- Page faults: 16KB per fault, synchronous kernel trap — 20× slower per byte than explicit pread

### Expert prediction:
- **Qwen** (measured 2026-03-30): **Pre-MoE + next LN, top-12**: 84-86% accuracy. Per-layer range: 57% (layer 1) to 96% (layer 34). Uses co-occurrence table.
- **Gemma4 UD-3bit** (measured 2026-04-05): **Level B prediction: 71.4% overall** (top-12). Per-layer range: 51% (layer 9) to 89% (layer 28).
- **Gemma4 old 4-bit** (measured 2026-04-04): **Level C prediction: 73% overall** (top-12). Per-layer range: 53% (layer 7) to 90% (layer 26).
- Three prediction tiers:
  - **Level C** (Gemma4): dense MLP + speculative attention (virtual KV append, no cache mutation) + next-layer router. ~0.6ms/layer GPU, runs lazily between async_eval/eval. Works with both plain and TurboQuant KV cache.
  - **Level A.5** (fallback): dense MLP + next-layer router (skip attention). ~0 extra cost (fills GPU idle time).
  - **Level B** (Qwen, Gemma4 fallback): CPU dequantized matmul of next-layer router on h_post_attn. ~0.1ms/layer, zero GPU impact. Pre-converted f32 weights in `RouterWeightsRef`.
- Old co-occurrence baseline for Gemma4 was 50.5% (still used as fallback for Qwen)

## Key gotchas

### MLX / mlx-rs
- **MLX Metal quantized_matmul/gather_qmm supports only {2, 3, 4, 6, 8}-bit**. Unsloth UD uses 5-bit for some dense MLP layers — `load_qlinear_flex()` auto-detects unsupported bit widths and re-quantizes at load time (CPU dequantize → GPU quantize to nearest supported). Currently only affects 3 weights on layer 29.
- `mlx_array_new_data` (and `Array::from_raw_data`) **copies** data — use `ffi_zerocopy.cpp` shim for zero-copy via `newBufferWithBytesNoCopy`
- `Array::load_safetensors()` creates lazy arrays; loading all expert files causes swap storms on 16 GB
- `gather_qmm` is NOT in mlx-rs — use `mlx_sys::mlx_gather_qmm` via FFI wrapper
- `argsort` runs on CPU; eval boundaries needed before GPU gather_qmm
- Activation dtype drift: bf16×f32 scalar promotes to f32 — cast scalars to input dtype
- `array_from_mmap` (Metal buffer creation via newBufferWithBytesNoCopy) is fast: 0.3ms for 108 arrays. NOT a bottleneck.

### Memory / UMA
- **Do NOT load all expert files via load_safetensors** — causes swap storms on 16 GB
- On-demand expert extraction via pread() is the correct approach
- **pread() is 3.6× faster than mmap demand-paging** (page fault overhead: ~107μs/page for cold 16 KB random reads)
- **GCD reactive prefault before eval is the current strategy**: F_RDADVISE + madvise(WILLNEED) + prefault touch (one byte per 16 KB page) on GCD userInitiated queue. Blocks via dispatch_group. Equivalent to pread for page warming, with cancellation support.
- **pread is also effective** (historical default): contiguous MB reads warm page cache; subsequent mmap zerocopy eval runs fault-free. Used at startup for warm set.
- **madvise(MADV_WILLNEED) alone is unreliable** — returns before pages are loaded. Combined with prefault touch in GCD workers, it becomes reliable.
- **mlock HURTS**: page table wire/unwire contends with GPU at kernel vm_map level.
- **Pread-based speculative is a net negative on 4-bit** — can't be cancelled, always contends. See benchmark matrix.
- **GCD speculative with cancellation works**: fire-and-forget prefault-only on utility queue, cancel via atomic when reactive starts. F_RDADVISE/madvise skipped for speculative (uncancellable kernel I/O causes contention). No SSD contention. Neutral-to-positive throughput impact.
- Per-layer eval ensures expert arrays are freed after each layer (peak ~13.5 MB for 4-bit, not cumulative)
- Expert LRU caching does NOT help — working set >> cache size on 16 GB
- **Warm set pread at startup** (opt-in via `--warm-set`): guarantees 63% of expert pages are resident. Less impactful now that GCD speculative provides similar warming.

### I/O architecture findings (2026-03-29/30/31):
- **Page faults as flow control**: GPU self-throttles to match SSD throughput. Natural pipelining — but explicit prefetch is more efficient per byte.
- **Pread-based speculative during eval causes SSD contention**: can't be cancelled. Consistently worse in all tested configurations.
- **GCD speculative with cancellation avoids contention**: atomic cancel flag lets reactive interrupt speculative. Pages already touched remain resident. Net positive (+0.8 tok/s).
- **Background par_iter is ~2.3× slower than main-thread par_iter**. GCD dispatch avoids this penalty.
- **Old pipeline (hybrid buffer pread between async_eval/eval) was always slower**: 4.8 tok/s vs 7.2 reactive-only. Removed.
- **The 2.7× pread gap** (8-bit): measured 400ms vs 150ms theoretical. Root cause: page cache can't hold warm set on 16 GB under memory pressure from 8-bit expert files.

### Env vars for testing:
- `NOREACTIVE=1` — skip reactive blocking pread (for A/B testing)

### Speculative prefetch methods tested:
- **GCD prefault-only + prediction (current, Gemma4)**: fire-and-forget prefault on utility queue, cancel via atomic. Level B/C prediction at 71-73%. Speculative adds zero overhead. **Net neutral to positive** (prediction warms cache, prefault-only avoids contention).
- **GCD cancellable prefault + co-occurrence (current, Qwen)**: fire-and-forget on utility queue, cancel via atomic when reactive starts. **Net positive** (+0.8 tok/s). The only method that avoids SSD contention.
- **F_RDADVISE + madvise in speculative (old, removed)**: kernel-level I/O hints can't be cancelled once issued. Causes SSD contention with reactive. **Net negative** on old Gemma4 4-bit (2.7 vs 3.7 tok/s without speculation).
- **F_RDADVISE only**: lightest (fcntl hint, ~0.5ms/tok). SSD contention with reactive. Net negative.
- **Main-thread pread**: perfect GPU overlap (wait=1ms) but 2.8ms/layer blocks pipeline. Net negative.
- **Background rayon pool (4-thread)**: 2.3× background-thread penalty on macOS. Net negative.
- **Background rayon (pre-alloc buffer)**: eliminating alloc didn't help — penalty is thread-level.
- **Background rayon (global pool, 12-thread)**: CPU oversubscription (24 threads on 12 cores).
- **POSIX AIO (aio_read)**: macOS emulates with kernel threads — same overhead as rayon.
- **madvise(MADV_WILLNEED)**: slow page-table walks (3.4ms/layer). Worse than pread.
- **mincore + pread (cold-only)**: mincore check overhead (480 syscalls) exceeds savings.
