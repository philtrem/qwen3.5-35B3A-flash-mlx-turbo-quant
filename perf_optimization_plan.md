# Performance Optimization Plan: All-Rust Inference Engine

## Project State

All-Rust UMA-native inference engine for Qwen3.5-35B-A3B (36.3 GB, 9-bit MLX) on Mac M4 base 16GB. Replaces the Python+Rust(PyO3) stack with a single Rust binary using `mlx-rs` (Rust bindings to MLX via mlx-c).

- **Binary:** `cargo build --release` → `./target/release/flash-qwen generate ...`
- **Model path:** `./split_model_st/` (safetensors format, created by `flash-qwen split`)
- **Tokenizer path:** `/Users/philtrem/.lmstudio/models/inferencerlabs/Qwen3.5-35B-A3B-MLX-9bit`
- **Current output:** Generates correct tokens (`<think>Thinking Process`) at 0.006 tok/s
- **Baseline:** Python+Rust engine achieved 1.7 tok/s with warm set

Full project context (file inventory, mlx-rs API quirks, architecture) is in auto-memory: check `all-rust-engine-implementation.md` via MEMORY.md.

## Problems (4 root causes for 0.006 tok/s)

1. **No warm set mlock** — every expert page-faults from SSD (~34 MB/token of SSD reads)
2. **Activation dtype drift** — bf16→f32 promotion in GatedDeltaNet never casts back, doubling compute/memory
3. **Unnecessary eval barriers** — 8+ forced GPU syncs per layer prevent MLX lazy graph optimization
4. **Remaining debug prints** — per-layer/per-token eprintln calls in hot path

## Changes

### 1. Fix activation dtype drift (bf16→f32)

**File:** `src/model/gated_delta.rs`

**Root cause:** Lines 82-86 create `norm_w` as `ones::<f32>` and multiply q/k by f32 scalar `inv_scale`. In MLX, `bf16_array * f32_scalar` promotes to f32. This f32 propagates through the entire recurrent loop, RMSNormGated, and out to the residual connection — every subsequent layer receives f32.

**Fix:** Cast the RMS norm weight and scale factors to the input dtype:
```rust
// Lines 82-86: replace
let norm_w = mlx_rs::ops::ones::<f32>(&[self.head_k_dim as i32])?;
let q = mlx_rs::fast::rms_norm(&q, &norm_w, 1e-6)?;
let q = &q * (inv_scale * inv_scale);
let k = mlx_rs::fast::rms_norm(&k, &norm_w, 1e-6)?;
let k = &k * inv_scale;

// with:
let norm_w = mlx_rs::ops::ones::<f32>(&[self.head_k_dim as i32])?.as_dtype(x.dtype())?;
let q = mlx_rs::fast::rms_norm(&q, &norm_w, 1e-6)?;
let inv2 = Array::from_f32(inv_scale * inv_scale).as_dtype(x.dtype())?;
let q = &q * &inv2;
let k = mlx_rs::fast::rms_norm(&k, &norm_w, 1e-6)?;
let inv1 = Array::from_f32(inv_scale).as_dtype(x.dtype())?;
let k = &k * &inv1;
```

Also cast the initial state zeros to input dtype (line 176 — already does this but uses q.dtype() which is now f32 if q was promoted):
```rust
// Line 176: ensure state matches v's dtype, not q's (since q might be promoted)
None => mlx_rs::ops::zeros::<f32>(&[b, hv as i32, dv, dk])?.as_dtype(v.dtype())?,
```

The `x` parameter to `forward()` carries the correct dtype from the layer input, so use `x.dtype()` as the reference dtype throughout.

### 2. Remove unnecessary eval barriers

**Files:** `src/model/moe.rs`, `src/model/gated_delta.rs`, `src/model/mod.rs`

Remove all `mlx_rs::transforms::eval()` calls that were added for debugging. The only eval calls that should remain are:
- `engine.rs:29` — after prefill (needed: must materialize logits before sampling)
- `engine.rs:43` — after first sample (needed: must get token value for decode loop)
- `engine.rs:67` — after each sample (needed: must get token value)

**Remove these eval calls:**
- `src/model/gated_delta.rs:92` — eval before recurrent loop (unnecessary, lazy graph handles it)
- `src/model/gated_delta.rs:203` — eval after tile/repeat (unnecessary)
- `src/model/moe.rs:55` — eval after argpartition (unnecessary)
- `src/model/moe.rs:66` — eval after sort (unnecessary)
- `src/model/moe.rs:72` — eval after gate gather_qmm (unnecessary)
- `src/model/moe.rs:85` — eval after unsort (unnecessary)
- `src/model/moe.rs:88` — eval after unflatten (unnecessary)
- `src/model/mod.rs:91` — eval after embedding dequant (unnecessary)

### 3. Remove debug prints from hot path

**Files:** `src/model/mod.rs`, `src/model/norm.rs`

**mod.rs** — remove all eprintln in `DecoderLayer::forward()` and `TextModel::forward()`:
- Line 44: `eprintln!("  Layer forward: ...")`
- Line 79-81: `eprintln!("input_ids dtype=...")`, `eprintln!("flat_ids dtype=...")`
- Line 92: `eprintln!("hidden after embed: ...")`

Keep the load-time prints (lines 164-178, 277) since those run once.

**norm.rs** — remove the dead empty if-block:
```rust
// Lines 12-15: remove
let x_last = x.dim(x.ndim() as i32 - 1);
let w_size = self.weight.size();
if x_last as usize != w_size {
}
```

### 4. Fix warm set mlock (per-expert granularity)

**File:** `src/memory.rs`

**Problem:** Current implementation mlocks entire layer files (~864 MB each). With warm experts spread across most layers, this tries to mlock ~34 GB on 16 GB RAM → OOM.

**Fix:** Parse safetensors headers to get per-tensor byte offsets, then mlock only the specific expert slices within each tensor. Each expert is ~3.4 MB (9 tensors × varying strides). The warm set is 2861 experts = 9.4 GB which fits in 16 GB with room for resident weights (2.6 GB) and OS (~3 GB).

**New implementation:**
```rust
pub struct ExpertMemoryManager {
    maps: Vec<Mmap>,
    tensor_offsets: Vec<LayerTensorOffsets>,  // parsed from safetensors headers
}

struct LayerTensorOffsets {
    data_start: usize,                        // 8 + header_size
    tensors: Vec<TensorInfo>,                 // 9 tensors per layer
}

struct TensorInfo {
    data_offset: usize,                       // from safetensors header data_offsets[0]
    per_expert_stride: usize,                 // shape[1] * shape[2] * dtype_size
    num_experts: usize,                       // shape[0] = 256
}

pub fn mlock_warm_set(&self, experts: &[(u32, u32)]) -> usize {
    for &(layer, expert_idx) in experts {
        let offsets = &self.tensor_offsets[layer as usize];
        for tensor in &offsets.tensors {
            let start = offsets.data_start + tensor.data_offset
                        + expert_idx as usize * tensor.per_expert_stride;
            let len = tensor.per_expert_stride;
            // Page-align for mlock
            let page_size = 16384; // Apple Silicon page size
            let aligned_start = start & !(page_size - 1);
            let aligned_len = (start + len - aligned_start + page_size - 1) & !(page_size - 1);
            let ptr = self.maps[layer as usize].as_ptr().add(aligned_start);
            libc::madvise(ptr as *mut _, aligned_len, libc::MADV_WILLNEED);
            libc::mlock(ptr as *const _, aligned_len);
            locked += aligned_len;
        }
    }
}
```

The safetensors header for each layer file has this structure (from actual file):
```
data_start = 875 bytes (8-byte length prefix + 867-byte JSON header)
gate_proj.weight: data_offsets=[0, 268435456],           shape=[256, 512, 512],  dtype=U32  → stride=1048576
gate_proj.scales: data_offsets=[268435456, 285212672],    shape=[256, 512, 64],   dtype=BF16 → stride=65536
gate_proj.biases: data_offsets=[285212672, 301989888],    shape=[256, 512, 64],   dtype=BF16 → stride=65536
up_proj.weight:   data_offsets=[301989888, 570425344],    shape=[256, 512, 512],  dtype=U32  → stride=1048576
up_proj.scales:   data_offsets=[570425344, 587202560],    shape=[256, 512, 64],   dtype=BF16 → stride=65536
up_proj.biases:   data_offsets=[587202560, 603979776],    shape=[256, 512, 64],   dtype=BF16 → stride=65536
down_proj.weight: data_offsets=[603979776, 872415232],    shape=[256, 2048, 128], dtype=U32  → stride=1048576
down_proj.scales: data_offsets=[872415232, 889192448],    shape=[256, 2048, 16],  dtype=BF16 → stride=65536
down_proj.biases: data_offsets=[889192448, 905969664],    shape=[256, 2048, 16],  dtype=BF16 → stride=65536
```

**Re-enable warm set auto-detection in main.rs** after fixing:
```rust
// src/main.rs: restore auto-detection
let warm_path = warm_experts
    .or_else(|| {
        let auto = model_path.join("warm_experts.json");
        if auto.exists() { Some(auto) } else { None }
    });
```

### 5. (Bonus) Remove `safetensors` crate dependency

**File:** `Cargo.toml`

Since we now use `Array::load_safetensors()` (MLX's native loader) instead of the `safetensors` Rust crate, we can remove the `safetensors` dependency. The splitter still writes safetensors manually (raw header + data), which doesn't need the crate.

Check `src/splitter.rs` — it only uses `serde_json` for header construction, not the safetensors crate.

## Files to modify

| File | Changes |
|------|---------|
| `src/model/gated_delta.rs` | Fix dtype drift (cast norm_w and scalars to x.dtype()), remove 2 eval barriers |
| `src/model/moe.rs` | Remove 5 eval barriers |
| `src/model/mod.rs` | Remove 1 eval barrier, remove 4 debug prints |
| `src/model/norm.rs` | Remove dead if-block |
| `src/memory.rs` | Rewrite mlock_warm_set with per-expert granularity (parse safetensors headers) |
| `src/main.rs` | Re-enable warm set auto-detection |
| `Cargo.toml` | Remove `safetensors` dependency (optional) |

## Verification

1. **Build:** `cargo build --release`
2. **Confirm dtype stays bf16:** Add temporary print in first 2 layers: `eprintln!("layer {} output dtype={:?}", i, h.dtype())` — should show Bfloat16 for all layers. Remove after confirming.
3. **Run decode test:**
   ```
   ./target/release/flash-qwen generate \
     --model-path ./split_model_st \
     --tokenizer-path /Users/philtrem/.lmstudio/models/inferencerlabs/Qwen3.5-35B-A3B-MLX-9bit \
     --prompt "Hello" --max-tokens 16
   ```
4. **Check memory:** `vm_stat` during generation — wired pages should show ~9-12 GB (warm set + resident), not growing unbounded
5. **Measure tok/s** from generation output — target is >1 tok/s, ideally matching or exceeding 1.7 tok/s baseline
6. **Verify output coherence** — model should produce meaningful text, not garbage (confirms dtype fix didn't break computation)

## Expected performance impact

| Change | Expected speedup |
|--------|-----------------|
| Remove eval barriers (8 per token) | ~2-4x (eliminates GPU sync stalls) |
| Fix bf16 dtype | ~1.5-2x (halves activation memory + compute) |
| Warm set mlock | ~3-5x (80% of expert accesses hit RAM instead of SSD) |
| Remove debug prints | ~1.1x (minor) |
| **Combined** | **Target: 1.5-3.0 tok/s** |
