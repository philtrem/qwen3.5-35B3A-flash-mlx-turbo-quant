# flash-moe

All-Rust inference engine for sparse Mixture-of-Experts models on Apple Silicon. Runs large MoE models on an M4 Mac Mini with 16 GB of RAM by loading experts on-demand from SSD.

**Supported models:**
- **Gemma 4 26B-A4B** — 26B params, 4B active. 30 layers, 128 experts, top-8.
- **Qwen 3.5 35B-A3B** — 35B params, 3B active. 40 layers, 256 experts, top-8.

The key idea: sparse MoE models only activate a small fraction of parameters per token (8 experts out of 128–256 per layer), so the full model doesn't need to fit in memory. flash-moe keeps resident weights in Metal buffers and loads experts on-demand from SSD via memory-mapped I/O, using GCD-dispatched prefetch to keep pages ahead of the GPU.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        CLI (clap)                       │
│                   split / generate                      │
├─────────────────────────────────────────────────────────┤
│  engine.rs         Token loop + nucleus sampling        │
├─────────────────────────────────────────────────────────┤
│  model/            Multi-model MoE (mlx-rs)             │
│    mod.rs          TextModel, DecoderLayer, MoeVariant  │
│    attention.rs    Qwen full attention (output gating)  │
│    gemma4_attn.rs  Gemma4 attention (K==V, sliding/full)│
│    gated_delta.rs  Qwen GatedDeltaNet linear attention  │
│    moe.rs          SparseMoeBlock + Gemma4MoeBlock      │
│    mlp.rs          MLP (SiLU) + GeLUMLP (Gemma4)       │
│    norm.rs         RMSNorm, RMSNormNoScale              │
├─────────────────────────────────────────────────────────┤
│  memory.rs         ExpertMemoryManager                  │
│                    ├ GCD speculative prefetch (utility) │
│                    ├ GCD reactive prefetch (userInit)   │
│                    ├ Zero-copy mmap → Metal buffers     │
│                    └ Warm set pread (optional)          │
├─────────────────────────────────────────────────────────┤
│  config.rs         ModelType auto-detection (Qwen/Gemma)│
│  cache.rs          KV cache + TurboQuant (optional)     │
│  ffi.rs            gather_qmm FFI + array_from_mmap     │
│  ffi_zerocopy.cpp  MLX Metal newBufferWithBytesNoCopy   │
│  splitter.rs       Model → resident + per-layer ECB     │
│  perf.rs           Per-phase timing (routing, eval, I/O)│
└─────────────────────────────────────────────────────────┘
         ▼ SSD (mmap)                    ▲ Metal GPU
   ┌──────────────┐              ┌──────────────┐
   │  Expert ECB  │  ──pages──▶  │  GPU eval    │
   │  files/layer │              │ (fault-free) │
   └──────────────┘              └──────────────┘
```

Model type is auto-detected from config.json. The splitter handles both Qwen (switch_mlp) and Gemma4 (fused gate_up_proj → unfused during ECB writing) expert layouts.

### I/O pipeline

The bottleneck isn't compute — it's getting expert bytes from SSD to GPU before it stalls. Without explicit prefetch, the GPU triggers page faults that pull data in 16 KB chunks — synchronous kernel traps that reduce effective SSD throughput to a fraction of what sequential reads achieve. flash-moe avoids this with a two-stage GCD prefetch pipeline:

1. **Speculative** (during GPU eval): After submitting the current layer to the GPU, fire off low-priority (utility QoS) GCD workers to prefault pages for the *next* layer's predicted experts. Uses routing pre-MoE signals for prediction.
2. **Reactive** (after routing): Once the router picks the actual 8 experts, cancel any in-flight speculative work (atomic flag — no SSD contention), then dispatch high-priority (userInitiated QoS) workers to prefault the exact pages needed. Blocks until all pages are resident.
3. **Eval** (zero faults): GPU reads from Metal buffers backed by already-resident mmap pages. Pure compute, no page faults.

Cancellation is what makes this work — without it, speculative I/O contends with reactive and throughput drops significantly.

## Requirements

- **macOS** on Apple Silicon (tested on M4 Mac Mini, 16 GB)
- **Rust** toolchain (stable)
- **Model weights** (one of):
  - [philtrem/gemma-4-26b-a4b-it-MLX-4bit](https://huggingface.co/philtrem/gemma-4-26b-a4b-it-MLX-4bit) (~13 GB after splitting)
  - [mlx-community/Qwen3.5-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) (~19 GB after splitting)
- C++ compiler (for the Metal zero-copy shim, built automatically via `build.rs`)

## Build

```bash
cargo build --release
```

## Usage

### 1. Split the model

Converts HuggingFace safetensors into resident weights + per-layer expert ECB files for on-demand loading:

```bash
./target/release/flash-moe split \
  --model-path /path/to/model \
  --output-path ./split_output
```

This is a one-time step. The split output is ~13 GB for Gemma4 4-bit, ~19 GB for Qwen 4-bit.

### 2. Generate

```bash
# Gemma 4 (default — model-path defaults to ./split_gemma4)
./target/release/flash-moe generate \
  --prompt "Explain the Riemann hypothesis in simple terms" \
  --max-tokens 256

# Qwen 3.5 (specify paths)
./target/release/flash-moe generate \
  --model-path ./split_model_st \
  --tokenizer-path /path/to/Qwen3.5-35B-A3B-4bit \
  --prompt "Hello" --max-tokens 256
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--no-speculate` | off | Disable speculative prefetch for predicted experts |
| `--warm-set` | off | Pread frequent experts into page cache at startup |
| `--kv-quant-bits N` | 3 | TurboQuant KV cache: 2, 3, or 4-bit quantization |
| `--no-kv-quant` | off | Disable KV cache quantization (plain bf16) |

## How it works

The model is split into two parts:

- **Resident weights** (~2.8–3.6 GB): embeddings, attention, norms, router weights, dense MLP (Gemma4), output head. Loaded once into Metal buffers at startup.
- **Expert files** (~428–453 MB/layer): one ECB (expert-centric binary) file per layer. Memory-mapped but never fully loaded — only the 8 active experts are paged in per layer per token.

On a 16 GB machine, the resident weights plus OS overhead leave roughly 10-12 GB for page cache. The SSD delivers ~3 GB/s, and the GCD prefetch pipeline ensures pages are resident before the GPU needs them, eliminating page fault stalls.

### Per-token I/O

| Model | Expert size | Active/layer | Layers | I/O per token |
|-------|------------|-------------|--------|--------------|
| Gemma 4 26B | 3.35 MB | 26.8 MB | 30 | 803 MB |
| Qwen 3.5 35B | 1.77 MB | 14.2 MB | 40 | 566 MB |

Gemma 4 reads 42% more data per token despite fewer layers, because its experts are ~2× bigger (hidden 2816×704 vs 2048×512). This is why Qwen achieves higher throughput despite being a larger model.

### Why not just load everything into RAM?

These models are 13–19 GB at 4-bit. On a 16 GB machine, that means swap, and swap means page faults during GPU eval — which is exactly what this project avoids. The MoE sparsity (3–6% active) makes on-demand loading viable: you only need the data you're actually using.

### Why not use the Neural Engine?

The M4's ANE isn't useful here. I/O is the bottleneck — even instant compute wouldn't dramatically change throughput. Beyond that, ANE doesn't support 4-bit quantized matmul (it handles float16/int8 via CoreML), so you'd have to dequantize to float16 first, doubling memory traffic. ANE dispatch latency is also tuned for large-batch CoreML inference, not single-token autoregressive decode where GPU compute is already a minority of wall time.

## License

MIT
