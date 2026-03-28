# All-Rust UMA-Native MoE Inference Engine

## Architecture Overview

Replace the current Rust(PyO3) + Python(MLX) stack with a single Rust binary that calls MLX directly via `mlx-rs`/`mlx-c`. No Python runtime at inference time.

```
Current:  Python (MLX) → builds lazy graph → MLX C++ → Metal GPU
          Rust (PyO3) → loads bytes → PyBytes → numpy → mx.array → copies

New:      Rust (mlx-rs) → builds lazy graph → MLX C++ → Metal GPU
          Rust (memmap2) → mlock/madvise → OS page cache → same Metal buffers
```

### Why

- **Eliminates Python overhead entirely** — no GIL, no object creation, no ref counting, no PyBytes/numpy intermediaries
- **UMA-native** — MLX's safetensors mmap creates MTLBuffers from file-mapped pages. GPU reads directly. Zero copies.
- **No tolist() sync** — routing indices stay as MLX arrays, passed directly to gather_qmm with raw 0-255 indices
- **Single binary** — no Python dependency, fast cold start, easy deployment
- **Simpler code** — one language, one build system, direct API calls

### Key Dependencies

| Crate | Purpose | Status |
|-------|---------|--------|
| `mlx-rs` (0.25) | Rust bindings to MLX via mlx-c | Active on crates.io |
| `mlx-sys` | Low-level C FFI (used by mlx-rs) | Available |
| `tokenizers` (0.21) | HuggingFace tokenizer.json loading | Production-ready |
| `minijinja` (2.x) | Jinja2 chat template rendering | Production-ready |
| `safetensors` (0.5) | Weight file I/O | Already in project |
| `memmap2` (0.9) | mmap for mlock/madvise | Already in project |
| `rayon` (1.10) | Parallel splitting | Already in project |
| `clap` | CLI argument parsing | Standard |

---

## Current Performance Baseline

- **1.7 tok/s** (14ms/layer × 40 layers = 560ms/token)
- Bottleneck: tolist() GPU sync (9-12ms/layer, 62%) + Rust copy chain (1.5-2ms/layer)

## Projected Performance

- Conservative: **2.3-2.5 tok/s** (eliminate sync + copies, GPU-bound)
- Optimistic: **2.5-3.0 tok/s** (MLX graph optimization across full lazy pipeline)
- With 4-bit later: **3.5+ tok/s**

---

## PHASE 1: Foundation

### 1A. Project Structure

Convert from PyO3 cdylib to standalone Rust binary:

```
src/
├── main.rs              — CLI: split / generate / profile-routing / build-warm-set
├── model/
│   ├── mod.rs           — Model + FlashLanguageModel + FlashTextModel
│   ├── attention.rs     — Attention (standard multi-head with RoPE + GQA)
│   ├── gated_delta.rs   — GatedDeltaNet (linear attention)
│   ├── moe.rs           — UMASparseMoeBlock (gather_qmm with 256 experts)
│   ├── mlp.rs           — MLP + shared expert
│   └── norm.rs          — RMSNorm
├── engine.rs            — Generation loop, sampling, token streaming
├── tokenizer.rs         — HF tokenizer + minijinja chat template
├── splitter.rs          — Model split → safetensors (reuse existing, adapt output)
├── memory.rs            — mlock/madvise for warm set pages
├── cache.rs             — KVCache, ArraysCache, TurboQuantCache
└── config.rs            — TextModelArgs from config.json
```

**Cargo.toml** changes:
```toml
[package]
name = "flash-qwen"
edition = "2021"

[[bin]]
name = "flash-qwen"
path = "src/main.rs"

[dependencies]
mlx-rs = "0.25"
tokenizers = "0.21"
minijinja = "2"
safetensors = "0.5"
memmap2 = "0.9"
rayon = "1.10"
libc = "0.2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
lz4_flex = "0.11"
```

Remove `pyo3` dependency entirely. Remove `[lib]` crate-type cdylib.

### 1B. Verify mlx-rs Coverage

Before writing model code, verify critical ops exist in mlx-rs:

| Operation | MLX Python | mlx-rs / mlx-c | Fallback |
|-----------|-----------|-----------------|----------|
| `gather_qmm` | `mx.gather_qmm()` | Verify in mlx-sys | `gather` + `quantized_matmul` separately |
| `rms_norm` | `mx.fast.rms_norm()` | Verify in mlx-sys | Manual: `x * rsqrt(mean(x², axis=-1) + eps)` |
| `silu` | `mx.nn.silu()` | Verify | `x * sigmoid(x)` |
| `argpartition` | `mx.argpartition()` | Verify | `argsort` + slice top-k |
| `softmax` | `mx.softmax()` | Likely available | `exp(x - max(x)) / sum(...)` |
| `rope` | `mx.fast.rope()` | Verify | Manual rotary embedding |
| `sdpa` | `mx.fast.scaled_dot_product_attention()` | Verify | Manual Q·K^T/√d · V |
| `metal_kernel` | `mx.fast.metal_kernel()` | Verify | See GatedDeltaNet fallback below |
| `load` (safetensors) | `mx.load()` | Should be available | Direct safetensors + array creation |

**Action**: Write a small test binary that imports mlx-rs, creates arrays, runs gather_qmm (or quantized_matmul + gather). If gather_qmm isn't wrapped, add the binding via mlx-sys FFI:

```rust
// If gather_qmm is missing from mlx-rs, call mlx-c directly:
extern "C" {
    fn mlx_gather_qmm(
        result: *mut mlx_array,
        x: mlx_array,
        w: mlx_array,
        scales: mlx_array,
        biases: mlx_array,
        // ... kwargs
    ) -> i32;
}
```

### 1C. Tokenizer + Chat Template

```rust
// tokenizer.rs
use tokenizers::Tokenizer;
use minijinja::Environment;

pub struct QwenTokenizer {
    tokenizer: Tokenizer,
    chat_env: Environment<'static>,
    eos_token_ids: Vec<u32>,
}

impl QwenTokenizer {
    pub fn from_dir(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))?;
        let template = fs::read_to_string(path.join("chat_template.jinja"))?;
        let mut env = Environment::new();
        env.add_template_owned("chat", template)?;
        // Read eos_token_id from config.json
        Ok(Self { tokenizer, chat_env: env, eos_token_ids })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> { ... }
    pub fn decode(&self, ids: &[u32]) -> String { ... }
    pub fn apply_chat_template(&self, messages: &[Message]) -> String { ... }
}
```

### 1D. Safetensors Splitter (adapt existing)

Rewrite `splitter.rs` to output per-layer safetensors instead of custom .bin:

```
split_model/experts/layer_00_experts.safetensors:
  gate_weight: (256, 512, 512)   uint32   ~256 MB
  gate_scales: (256, 512, 64)    bf16     ~16 MB
  gate_biases: (256, 512, 64)    bf16     ~16 MB
  up_weight:   (256, 512, 512)   uint32   ~256 MB
  up_scales:   (256, 512, 64)    bf16     ~16 MB
  up_biases:   (256, 512, 64)    bf16     ~16 MB
  down_weight: (256, 2048, 128)  uint32   ~256 MB
  down_scales: (256, 2048, 16)   bf16     ~16 MB
  down_biases: (256, 2048, 16)   bf16     ~16 MB
```

The original model already has expert tensors as stacked (256, d1, d2). The new splitter just groups them by layer and writes safetensors — simpler than the current deinterleaving + custom binary format.

---

## PHASE 2: Model Implementation

### 2A. Config Loading

```rust
// config.rs
#[derive(Deserialize)]
pub struct TextModelArgs {
    pub hidden_size: usize,         // 2048
    pub num_hidden_layers: usize,   // 40
    pub num_experts: usize,         // 256
    pub num_experts_per_tok: usize, // 8
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub full_attention_interval: usize, // 4
    pub norm_topk_prob: bool,
    pub tie_word_embeddings: bool,
    // ... quantization config
}
```

### 2B. RMSNorm

```rust
// norm.rs
pub struct RMSNorm {
    weight: Array,  // mlx-rs Array
    eps: f32,
}

impl RMSNorm {
    pub fn forward(&self, x: &Array) -> Array {
        // If mx.fast.rms_norm available via mlx-rs:
        mlx_rs::fast::rms_norm(x, &self.weight, self.eps)
        // Fallback:
        // let ms = x.square().mean(-1, true);
        // x * (ms + self.eps).rsqrt() * &self.weight
    }
}
```

### 2C. MLP + Shared Expert

```rust
// mlp.rs — straightforward quantized linear layers
pub struct MLP {
    gate_proj: QuantizedLinear,  // hidden → intermediate
    up_proj: QuantizedLinear,
    down_proj: QuantizedLinear,
}

impl MLP {
    pub fn forward(&self, x: &Array) -> Array {
        let gate = self.gate_proj.forward(x).silu();
        let up = self.up_proj.forward(x);
        self.down_proj.forward(&(gate * up))
    }
}
```

### 2D. Attention (Standard + RoPE + GQA)

```rust
// attention.rs — every 4th layer (layers 3, 7, 11, ..., 39)
pub struct Attention {
    q_proj: QuantizedLinear,
    k_proj: QuantizedLinear,
    v_proj: QuantizedLinear,
    o_proj: QuantizedLinear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
}

impl Attention {
    pub fn forward(&self, x: &Array, mask: Option<&Array>, cache: &mut KVCache) -> Array {
        let q = self.q_proj.forward(x).reshape(/* heads */);
        let k = self.k_proj.forward(x).reshape(/* kv_heads */);
        let v = self.v_proj.forward(x).reshape(/* kv_heads */);

        let (q, k) = apply_rope(q, k, cache.offset(), self.rope_theta);
        let (k, v) = cache.update(k, v);

        // Scaled dot-product attention (use mx.fast.sdpa if available)
        let attn = scaled_dot_product_attention(&q, &k, &v, mask);
        self.o_proj.forward(&attn.reshape(/* merge heads */))
    }
}
```

### 2E. GatedDeltaNet (Linear Attention) — THE CRITICAL PIECE

**Risk**: GatedDeltaNet uses `mx.fast.metal_kernel()` for a custom Metal shader (`gated_delta_step`). This compiles a Metal kernel at runtime from a string source.

**Three paths forward (in order of preference)**:

**Path A — Use metal_kernel from mlx-c/mlx-rs** (best):
If mlx-rs exposes `mx.fast.metal_kernel()` (or we add the binding), we can compile the same Metal shader source from Rust. The shader source is a string constant — same code, just invoked from Rust instead of Python.

**Path B — Pure-ops fallback** (safe, slower):
The Python code has a fallback path using standard MLX ops (no custom kernel). Translate this to Rust. Performance may be ~10-20% slower for the 30 linear-attention layers, but it works immediately.

```rust
// gated_delta.rs — fallback using standard ops
pub fn gated_delta_step(
    q: &Array, k: &Array, v: &Array,
    beta: &Array, state: &mut Array,
) -> Array {
    // delta rule: state = state * (1 - beta·k·k^T) + beta·k·v^T
    // output = q^T · state
    let kv = k.transpose(-2, -1).matmul(v);
    let kk = k.transpose(-2, -1).matmul(k);
    *state = state * (1.0 - beta * kk) + beta * kv;
    q.matmul(state)
}
```

**Path C — Raw Metal shader** (complex, maximum control):
Write the Metal shader directly via `metal-rs` crate, compile it at build time, submit via Metal command buffers. Bypasses MLX for this one kernel. Maximum performance but significant effort.

**Recommendation**: Start with Path B (pure ops). Profile. If GatedDeltaNet is a bottleneck, try Path A. Fall back to Path C only if needed.

```rust
// gated_delta.rs
pub struct GatedDeltaNet {
    in_proj_qkv: QuantizedLinear,
    in_proj_z: QuantizedLinear,
    in_proj_b: QuantizedLinear,
    in_proj_a: QuantizedLinear,
    out_proj: QuantizedLinear,
    conv: DepthwiseConv1d,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    a_param: Array,  // A_log parameter
    // dimensions...
}

impl GatedDeltaNet {
    pub fn forward(&self, x: &Array, mask: Option<&Array>, cache: &mut ArraysCache) -> Array {
        // 1. Project to Q, K, V, Z, Beta
        // 2. Apply conv1d + silu
        // 3. Apply q/k norms
        // 4. Compute gating A from A_log
        // 5. For each step: gated_delta_step (recurrent update)
        // 6. Apply output gate z (silu) and project
    }
}
```

### 2F. UMA-Native MoE Block

The simplest and most impactful component — fully lazy, no sync:

```rust
// moe.rs
pub struct UMASparseMoeBlock {
    gate: QuantizedLinear,          // router: hidden → 256
    shared_expert: MLP,
    shared_expert_gate: QuantizedLinear,  // hidden → 1
    top_k: usize,
    norm_topk_prob: bool,
    bits: u32,
    group_size: u32,
    // Pre-loaded mmap'd expert arrays: each (256, d1, d2)
    gate_weight: Array,   // (256, 512, 512) uint32
    gate_scales: Array,   // (256, 512, 64)  bf16
    gate_biases: Array,   // (256, 512, 64)  bf16
    up_weight: Array,
    up_scales: Array,
    up_biases: Array,
    down_weight: Array,
    down_scales: Array,
    down_biases: Array,
}

impl UMASparseMoeBlock {
    pub fn forward(&self, x: &Array) -> Array {
        // Router (LAZY)
        let gates = self.gate.forward(x).softmax(-1);
        let inds = gates.argpartition(-self.top_k as i32, -1)
            .slice_last(self.top_k);  // top-k indices (0-255)
        let scores = gates.take_along_axis(&inds, -1);
        let scores = if self.norm_topk_prob {
            &scores / scores.sum(-1, true)
        } else { scores };

        // Shared expert (LAZY, independent)
        let shared_y = self.shared_expert_gate.forward(x).sigmoid()
            * self.shared_expert.forward(x);

        // Sort for gather_qmm (LAZY)
        let x_exp = x.expand_dims(&[-2, -3]);
        let flat_idx = inds.reshape(-1);
        let order = flat_idx.argsort(-1);
        let inv_order = order.argsort(-1);
        let k = self.top_k;
        let x_sorted = x_exp.flatten(0, -3).take(&(order / k as u32), 0);
        let idx_sorted = flat_idx.take(&order, 0);

        // gather_qmm triad (LAZY — GPU reads from mmap'd Metal buffers)
        let x_gate = gather_qmm(
            &x_sorted, &self.gate_weight, &self.gate_scales, &self.gate_biases,
            &idx_sorted, self.bits, self.group_size, true);
        let x_up = gather_qmm(
            &x_sorted, &self.up_weight, &self.up_scales, &self.up_biases,
            &idx_sorted, self.bits, self.group_size, true);
        let x_act = x_gate.silu() * x_up;
        let x_down = gather_qmm(
            &x_act, &self.down_weight, &self.down_scales, &self.down_biases,
            &idx_sorted, self.bits, self.group_size, true);

        // Unsort + combine (LAZY)
        let x_down = x_down.take(&inv_order, 0)
            .unflatten(0, inds.shape())
            .squeeze(-2);
        let y = (x_down * scores.expand_dims(-1)).sum(-2, false);
        y + shared_y
    }
}
```

**Zero sync points. Zero copies. All 256 experts live in mmap'd Metal buffers.**

### 2G. Weight Loading

```rust
// In engine init:
fn load_model(split_path: &Path, args: &TextModelArgs) -> Model {
    // 1. Load resident weights (attention, router, shared expert, embeddings)
    let resident = mlx_rs::load_safetensors(split_path.join("resident/resident.safetensors"));

    // 2. Load expert weights (per-layer safetensors, mmap'd)
    let mut expert_tensors = Vec::new();
    for layer in 0..args.num_hidden_layers {
        let path = split_path.join(format!("experts/layer_{:02}_experts.safetensors", layer));
        let tensors = mlx_rs::load_safetensors(&path);  // mmap internally
        expert_tensors.push(tensors);
    }

    // 3. Build model, assign weights
    // Resident weights go to attention/router/shared_expert modules
    // Expert tensors go to UMASparseMoeBlock fields
    build_model(args, resident, expert_tensors)
}
```

---

## PHASE 3: Inference Engine

### 3A. Generation Loop

```rust
// engine.rs
pub fn generate(
    model: &Model,
    tokenizer: &QwenTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> String {
    let input_ids = tokenizer.encode(prompt);
    let mut cache = model.make_cache();

    // Prefill
    let input = Array::from_slice(&input_ids, &[1, input_ids.len()]);
    let logits = model.forward(&input, &mut cache);
    mlx_rs::eval(&logits);

    // Decode loop
    let mut next_token = sample(&logits.slice((.., -1, ..)), temperature, top_p);
    mlx_rs::eval(&next_token);
    let mut generated = vec![next_token.item::<u32>()];

    for _ in 1..max_tokens {
        let input = next_token.reshape(&[1, 1]);
        let logits = model.forward(&input, &mut cache);
        next_token = sample(&logits.slice((.., -1, ..)), temperature, top_p);
        mlx_rs::eval(&next_token);

        let tok_id = next_token.item::<u32>();
        if tokenizer.is_eos(tok_id) { break; }
        generated.push(tok_id);

        // Stream token to stdout
        print!("{}", tokenizer.decode(&[tok_id]));
    }

    tokenizer.decode(&generated)
}
```

### 3B. Sampling

```rust
fn sample(logits: &Array, temperature: f32, top_p: f32) -> Array {
    if temperature == 0.0 {
        return logits.argmax(-1);
    }
    let logits = logits / temperature;
    if top_p < 1.0 {
        // Nucleus sampling
        let sorted_idx = logits.argsort(-1).reverse(-1);
        let sorted_logits = logits.take_along_axis(&sorted_idx, -1);
        let probs = sorted_logits.softmax(-1);
        let cumulative = probs.cumsum(-1);
        let mask = (cumulative - &probs).gt(top_p);
        let filtered = sorted_logits.where_(&mask, f32::NEG_INFINITY);
        let inv_idx = sorted_idx.argsort(-1);
        let logits = filtered.take_along_axis(&inv_idx, -1);
    }
    let probs = logits.softmax(-1);
    mlx_rs::random::categorical(&probs.log())
}
```

### 3C. CLI

```rust
// main.rs
use clap::{Parser, Subcommand};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Split {
        #[arg(long)] model_path: PathBuf,
        #[arg(long)] output_path: PathBuf,
    },
    Generate {
        #[arg(long)] model_path: PathBuf,
        #[arg(long)] tokenizer_path: PathBuf,
        #[arg(long, default_value = "Hello")] prompt: String,
        #[arg(long, default_value_t = 256)] max_tokens: usize,
        #[arg(long, default_value_t = 0.7)] temperature: f32,
        #[arg(long, default_value_t = 0.9)] top_p: f32,
        #[arg(long)] kv_bits: Option<u32>,
        #[arg(long)] warm_experts: Option<PathBuf>,
    },
    ProfileRouting { ... },
    BuildWarmSet { ... },
}
```

---

## PHASE 4: UMA Memory Management

### 4A. Warm Set mlock

```rust
// memory.rs
pub struct ExpertMemoryManager {
    maps: Vec<memmap2::Mmap>,           // parallel mmap of expert safetensors
    tensor_offsets: Vec<TensorOffsets>,  // byte offsets per tensor per layer
}

impl ExpertMemoryManager {
    pub fn mlock_warm_set(&self, experts: &[(u32, u32)]) -> usize {
        let mut locked = 0;
        for &(layer, expert_idx) in experts {
            for offset_info in &self.tensor_offsets[layer as usize].all_tensors() {
                let start = offset_info.data_offset + expert_idx as usize * offset_info.per_expert;
                let len = offset_info.per_expert;
                let ptr = self.maps[layer as usize].as_ptr().add(start);
                unsafe {
                    libc::madvise(ptr as *mut _, len, libc::MADV_WILLNEED);
                    if libc::mlock(ptr as *const _, len) == 0 { locked += len; }
                }
            }
        }
        locked
    }

    pub fn prefetch_experts(&self, layer_idx: u32, expert_indices: &[u32]) {
        // madvise(MADV_WILLNEED) for predicted next-layer pages
    }
}
```

Both MLX's mmap and our parallel mmap point to the same files → same physical pages. mlock via our mmap also locks them for MLX.

### 4B. Profiling Infrastructure

Replace Python profiler with Rust timing:

```rust
// Wrap each forward pass section with timing
struct MoeProfiler {
    layer_times: Vec<Duration>,
}
// Use std::time::Instant around forward pass sections
// Report at end of generation
```

---

## Risk Assessment

### HIGH RISK: GatedDeltaNet Custom Metal Kernel

The Python implementation uses `mx.fast.metal_kernel()` to compile a custom shader at runtime. Three mitigation paths:

| Path | Effort | Performance | Risk |
|------|--------|-------------|------|
| B: Pure-ops fallback | Low | ~10-20% slower for 30 layers | None |
| A: metal_kernel via mlx-c | Medium | Same as Python | Depends on mlx-c coverage |
| C: Raw Metal shader | High | Best possible | Complex, maintenance burden |

**Start with Path B.** Profile. If the 30 GatedDeltaNet layers become the bottleneck (they might not — the MoE blocks dominate), upgrade to Path A or C.

### MEDIUM RISK: gather_qmm in mlx-rs

If `mlx-rs` doesn't wrap `gather_qmm`, add the binding via `mlx-sys` FFI (one extern function declaration + wrapper). This is straightforward — the function exists in mlx-c, just might not be wrapped yet.

### MEDIUM RISK: Quantized Weight Loading

MLX's quantized format (8-bit, group_size=32, packed uint32) must be loaded correctly. The resident weights are in safetensors with specific dtype encoding. Need to verify mlx-rs handles `nn.quantize`-style weight loading.

Fallback: Load raw arrays and manually construct the quantized representation.

### LOW RISK: RMSNorm / silu / RoPE

These are simple operations with trivial manual implementations if not exposed in mlx-rs.

---

## Memory Budget

| Component | Size |
|-----------|------|
| Resident model weights | 2.57 GB (mmap'd safetensors → Metal buffers) |
| Expert data (mmap virtual) | ~34 GB virtual, ~9.6 GB physical (mlocked warm set) |
| OS + MLX framework | ~3 GB |
| KV cache | ~0.2-0.5 GB |
| Rust binary + heap | ~50 MB |
| **Physical total** | **~15.4 GB** |

No Python runtime → saves ~200-400 MB vs current.

---

## Implementation Order

1. **1B** — Verify mlx-rs ops (gather_qmm, rms_norm, etc.) with test binary
2. **1A** — Project restructure (Cargo.toml, src/ layout)
3. **1C** — Tokenizer + chat template
4. **1D** — Safetensors splitter (adapt existing)
5. **2A-2C** — Config, RMSNorm, MLP (simple layers first)
6. **2F** — UMASparseMoeBlock (the core win)
7. **2D** — Attention with RoPE + GQA
8. **2E** — GatedDeltaNet (Path B fallback first)
9. **2G** — Weight loading + model assembly
10. **3A-3C** — Generation loop, sampling, CLI
11. **4A** — Warm set mlock
12. **Benchmark** — compare against 1.7 tok/s baseline
13. **4B** — Profiling, warm set refresh
14. **Deferred** — 4-bit requant, GatedDeltaNet Metal kernel (Path A/C)

## Verification

1. **Correctness**: Generate on reference prompts, diff output vs Python engine (same model, same prompt, temperature=0)
2. **Performance**: `std::time::Instant` around generation loop
3. **Memory**: `vm_stat` for wired pages after mlock
4. **Page faults**: `vm_stat` for page-ins during generation

## What's Deleted

- `flash_qwen/` — entire Python package
- `run.py` — replaced by `cargo run`
- PyO3 dependency — gone
- numpy dependency — gone
- Python venv — no longer needed at inference time (keep for reference/comparison)

## What's Kept

- `split_model/` — directory structure (new safetensors format)
- `warm_experts.json` — format unchanged
- `routing_profile.json` — format unchanged
- `src/splitter.rs` — adapted for safetensors output
- `Cargo.toml` — updated dependencies

---

## APPENDIX A: Model Configuration (from config.json)

```
text_config:
  hidden_size: 2048
  num_hidden_layers: 40
  num_attention_heads: 16
  num_key_value_heads: 2
  head_dim: 256
  vocab_size: 248320
  rms_norm_eps: 1e-6
  tie_word_embeddings: false
  full_attention_interval: 4     # every 4th layer is full attention
  hidden_act: silu

  # MoE
  num_experts: 256
  num_experts_per_tok: 8
  moe_intermediate_size: 512     # expert intermediate dim
  shared_expert_intermediate_size: 512
  norm_topk_prob: true

  # Linear attention (GatedDeltaNet) — 30 of 40 layers
  linear_num_value_heads: 32
  linear_num_key_heads: 16
  linear_key_head_dim: 128
  linear_value_head_dim: 128
  linear_conv_kernel_dim: 4

  # RoPE (for full attention layers only)
  rope_theta: 10000000
  partial_rotary_factor: 0.25    # only first 25% of head_dim rotated
  max_position_embeddings: 262144

  # Layer types (explicit list)
  layer_types: [linear, linear, linear, full, linear, linear, linear, full, ...]
  # Pattern: 3 linear + 1 full, repeating. Full attention at layers 3,7,11,...,39

quantization:
  bits: 8
  group_size: 32
  mode: affine
  # Per-layer overrides for gate and shared_expert_gate (all 8-bit, group_size=32)

eos_token_id: [248046, 248044]
```

## APPENDIX B: GatedDeltaNet Algorithm (30 of 40 layers)

### Dimensions
- Q, K: [B, S, num_k_heads=16, key_head_dim=128]
- V: [B, S, num_v_heads=32, value_head_dim=128]
- State: [B, num_v_heads=32, value_head_dim=128, key_head_dim=128] = [B, 32, 128, 128]
- Conv state: [B, kernel_size-1=3, key_dim*2 + value_dim] = [B, 3, 8192]
- Beta: [B, S, num_v_heads=32]
- G (decay): [B, S, num_v_heads=32]

### Forward Pass
```
1. Project: qkv = in_proj_qkv(x)        # [B, S, key_dim*2 + value_dim = 4096+4096+4096 = 8192]
            z = in_proj_z(x)              # [B, S, value_dim=4096] → reshape [B, S, 32, 128]
            b = in_proj_b(x)              # [B, S, 32]
            a = in_proj_a(x)              # [B, S, 32]

2. Conv1d: concat conv_state + qkv, apply depthwise conv1d (kernel=4, groups=8192), silu
   Update cache[0] = last 3 tokens of conv input

3. Split conv output → q [B,S,16,128], k [B,S,16,128], v [B,S,32,128]

4. Normalize: q = (1/√128)² × rms_norm(q)   k = (1/√128) × rms_norm(k)

5. Compute gating:
   beta = sigmoid(b)                        # [B, S, 32]
   g = exp(-exp(A_log) * softplus(a + dt_bias))  # [B, S, 32] — decay factor

6. If num_v_heads > num_k_heads (32 > 16): repeat q, k along head dim (2x)

7. Recurrent step (for each timestep t):
   state = state * g[t]                      # decay
   kv_mem = sum_k(state * k[t])              # [B, 32, 128] — memory readout
   delta = (v[t] - kv_mem) * beta[t]         # [B, 32, 128] — update signal
   state = state + outer(k[t], delta)        # [B, 32, 128, 128] — write to memory
   out[t] = sum_k(state * q[t])              # [B, 32, 128] — read from memory
   Update cache[1] = state

8. Output gate: out = RMSNormGated(out, z)   # norm with silu gating
9. Project: out_proj(out.reshape(B, S, 4096)) → [B, S, 2048]
```

### RMSNormGated
```
RMSNormGated(x, z):
  x_normed = rms_norm(x, weight, eps)
  return x_normed * silu(z)
```

### Metal Kernel vs Ops Fallback
- Metal kernel: processes full sequence in parallel, uses SIMD reductions
- Ops fallback: sequential loop over timesteps with compiled step function
- Start with ops fallback (Path B). Port Metal kernel later if needed.

## APPENDIX C: Attention Layer (10 of 40 layers — layers 3,7,11,...,39)

### Key Details
- Query has GATING: q_proj outputs 2× head_dim, split into queries + gate
- Per-head RMS norm on Q and K (before RoPE)
- Partial RoPE: only first 64 dims (25% of head_dim=256) are rotated
- GQA: 16 query heads, 2 KV heads (8:1 ratio)
- Output gating: `o_proj(output * sigmoid(gate))`

### Forward Pass
```
1. q_proj(x) → [B, L, 16 * 256 * 2 = 8192]
   Split into queries [B, L, 16, 256] and gate [B, L, 16*256]

2. k_proj(x) → [B, L, 2 * 256 = 512] → reshape [B, L, 2, 256]
   v_proj(x) → [B, L, 2 * 256 = 512] → reshape [B, L, 2, 256]

3. Per-head RMS norm: q_norm(queries), k_norm(keys)

4. Transpose to [B, heads, L, head_dim]

5. Apply RoPE (partial: first 64 of 256 dims), with cache offset

6. KV cache update: keys, values = cache.update_and_fetch(keys, values)

7. SDPA: mx.fast.scaled_dot_product_attention(q, k, v, scale=1/√256, mask)

8. Transpose back, reshape to [B, L, 16*256=4096]

9. Output: o_proj(output * sigmoid(gate))  → [B, L, 2048]
```

## APPENDIX D: Weight Naming Conventions

### Original Model (HuggingFace safetensors)
```
Expert tensors (detected by "switch_mlp" in key):
  language_model.model.layers.{N}.mlp.switch_mlp.gate_proj.weight   (256, 512, 512) uint32
  language_model.model.layers.{N}.mlp.switch_mlp.gate_proj.scales   (256, 512, 64)  bf16
  language_model.model.layers.{N}.mlp.switch_mlp.gate_proj.biases   (256, 512, 64)  bf16
  language_model.model.layers.{N}.mlp.switch_mlp.up_proj.weight     (256, 512, 512) uint32
  language_model.model.layers.{N}.mlp.switch_mlp.up_proj.scales     (256, 512, 64)  bf16
  language_model.model.layers.{N}.mlp.switch_mlp.up_proj.biases     (256, 512, 64)  bf16
  language_model.model.layers.{N}.mlp.switch_mlp.down_proj.weight   (256, 2048, 128) uint32
  language_model.model.layers.{N}.mlp.switch_mlp.down_proj.scales   (256, 2048, 16)  bf16
  language_model.model.layers.{N}.mlp.switch_mlp.down_proj.biases   (256, 2048, 16)  bf16

Resident tensors (everything else):
  language_model.model.embed_tokens.weight
  language_model.model.layers.{N}.input_layernorm.weight
  language_model.model.layers.{N}.post_attention_layernorm.weight
  language_model.model.layers.{N}.linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}.{weight,scales,biases}
  language_model.model.layers.{N}.linear_attn.conv1d.weight
  language_model.model.layers.{N}.linear_attn.norm.weight
  language_model.model.layers.{N}.linear_attn.dt_bias
  language_model.model.layers.{N}.linear_attn.A_log
  language_model.model.layers.{N}.self_attn.{q_proj,k_proj,v_proj,o_proj}.{weight,scales,biases}
  language_model.model.layers.{N}.self_attn.q_norm.weight
  language_model.model.layers.{N}.self_attn.k_norm.weight
  language_model.model.layers.{N}.mlp.gate.{weight,scales,biases}       # MoE router
  language_model.model.layers.{N}.mlp.shared_expert.{gate_proj,up_proj,down_proj}.{weight,scales,biases}
  language_model.model.layers.{N}.mlp.shared_expert_gate.{weight,scales,biases}
  language_model.model.norm.weight
  lm_head.{weight,scales,biases}
```

### Split Model (current format)
```
split_model/
├── config.json                     # full model config
├── resident/
│   ├── resident.safetensors        # ~2.57 GB, prefix "language_model." stripped
│   └── model.safetensors.index.json
├── experts/
│   ├── layer_00_experts.bin        # ~864 MB each, custom FEXP format
│   ├── layer_01_experts.bin
│   └── ... (40 files)
├── warm_experts.json               # {budget_mb, num_experts, coverage, experts: [[layer,expert],...]}
└── routing_profile.json            # {"total_tokens", "num_prompts", "experts": {"layer,expert": count}}
```

### New Split Model (safetensors format — target)
```
split_model/
├── config.json
├── resident/
│   ├── resident.safetensors        # unchanged
│   └── model.safetensors.index.json
├── experts/
│   ├── layer_00_experts.safetensors  # ~864 MB, 9 tensors (gate/up/down × weight/scales/biases)
│   ├── layer_01_experts.safetensors
│   └── ... (40 files)
├── warm_experts.json
└── routing_profile.json
```

## APPENDIX E: Cache Types

### ArraysCache (for GatedDeltaNet — 30 layers)
- `cache[0]` = conv_state: [B, 3, 8192] (last 3 tokens of conv input)
- `cache[1]` = recurrent_state: [B, 32, 128, 128] (RNN memory)
- Size per layer: 3×8192×2 + 32×128×128×2 = ~1.1 MB (bf16)

### KVCache (for Attention — 10 layers)
- keys: [B, 2, T, 256] — grows with sequence length
- values: [B, 2, T, 256]
- Pre-allocates in chunks of 256 tokens
- Size per layer per token: 2×2×256×2×2 = 4 KB (bf16, 2 KV heads)

### TurboQuantCache (optional, replaces KVCache)
- Quantizes K/V to 2/3/4-bit with rotation + Lloyd-Max codebook
- Must NOT have a `bits` attribute (triggers wrong SDPA path)
- ~50% memory savings vs FP16 KVCache

## APPENDIX F: Original Model Location

```
/Users/philtrem/.lmstudio/models/inferencerlabs/Qwen3.5-35B-A3B-MLX-9bit
```

This is the source for the split command. Contains sharded safetensors + tokenizer.json + chat_template.jinja.
