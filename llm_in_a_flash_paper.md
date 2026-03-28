# LLM in a Flash: Efficient Large Language Model Inference with Limited Memory

**Authors:** Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, Karen Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, Mehrdad Farajtabar (Apple)
**Published:** December 2023 (v1), July 2024 (v3) — ACL 2024
**Source:** https://arxiv.org/abs/2312.11514

## Abstract

Addresses running LLMs larger than available DRAM by storing model parameters in flash memory and loading them into DRAM on demand. Develops an inference cost model that takes into account flash memory characteristics. Two main techniques: **windowing** (reduces data transfer through neuron reuse) and **row-column bundling** (enables larger sequential reads from flash). Achieves models up to 2x the size of available DRAM, with 4-5x (CPU) and 20-25x (GPU) inference speedup compared to naive loading.

---

## 1. Flash Memory Cost Model

**Total latency = I/O latency + Memory management overhead + Compute latency**

I/O latency depends on:
- Data volume transferred from flash to DRAM
- Read throughput (GB/s), which varies with chunk size
- **Latency to first byte** dominates small read times

Key hardware insight: Flash throughput increases dramatically with sequential read size. On M1 Max 1TB SSD: >6 GiB/s for 1GiB linear read, but much lower for random small reads.

**Counterintuitive finding:** "Worthwhile to read more than needed (larger chunks) and discard, rather than strictly necessary parts in smaller chunks."

---

## 2. Windowing Strategy

### Core Mechanism
Maintain a DRAM cache of only the weight rows predicted required by recent subset of input tokens.

### Delta Loading Formula
- Let `s_agg(k)` = cumulative neuron usage across k input tokens
- For each new token, load `s_agg(k+1) - s_agg(k)` neurons
- The slope of aggregated neuron usage is decreasing — larger k values reduce incremental loads

### Window Size
- k=5 used across experiments
- Maximize k within available DRAM constraints

### Algorithm
1. Track `last_k_active` neurons for past k tokens
2. When processing new token, identify newly-required neurons
3. Load only the difference, delete oldest token's exclusive neurons

---

## 3. Row-Column Bundling

### Storage Layout
Column i from up-projection and row i from down-projection are stored together in flash, since both activate when intermediate neuron i fires.

### Byte-Level Details
- Without bundling: read `d_model x num_bytes` per access
- With bundling: read `2 x d_model x num_bytes` (doubled chunk size)

### Throughput Impact
- Random reads: 1.25 GB/s (sparse weights)
- With bundling: 2.25 GB/s (80% improvement)
- Dense sequential: 6.1 GB/s

---

## 4. Sparsity Predictor

### Architecture
Low-rank neural predictor per layer.

### OPT 6.7B Configuration
- Layers 1-28: rank r=128
- Layers 29-32: rank r=1024

### Training
- Dataset: 10,000 C4 samples, 2 epochs
- Hardware: A100 GPU, 4 hours per predictor
- Loss: Balanced loss over negative and positive samples

### Performance
- OPT 6.7B: 5% false negatives, 7% false positives average
- Predictor activates ~3x actual ReLU sparsity (OPT, Persimmon), ~2x for Phi-2
- Computational overhead: <2.4% of model weights and FLOPs

---

## 5. Memory Partitioning (50% DRAM Budget)

For OPT 6.7B with half-memory constraint (6.7 GB available):

| Component | Size |
|-----------|------|
| Embeddings | 3% |
| Attention weights | 32.3% |
| Predictors | 1.25% |
| Loaded FFN (windowed) | 15.5% |
| **Total** | **52.1%** |

With k=4 windowing: only 2.4% of FFN in DRAM on average, scaled to 24% for bundling overhead.

---

## 6. Data Management Algorithm

### Pre-Allocation Strategy
```
For layer i: allocate matrix of size Req_i x 2*d_model
where Req_i = max neurons needed for window k on C4 validation subset
```

### Memory Structure Per Layer
- Pointer vector: original neuron indices
- Matrix: concatenated [up-projection row | down-projection column]
- Bias: up-projection bias values
- `num_used`: count of active rows
- `last_k_active`: neurons from past k tokens

### Neuron Deletion (O(c) complexity)
1. Identify expired neurons using `last_k_active`
2. Replace with most recent elements (swap with last row)
3. Decrement `num_used` counter
4. Rewrite O(c x d_model) bytes

### Neuron Insertion
1. Load from flash to contiguous buffer
2. Append rows from `num_used` to `num_used + num_new`
3. No reallocation needed

---

## 7. Reading Patterns & I/O

- **Parallelization:** 32 threads for I/O to amortize latency-to-first-byte
- **Optimal chunk size:** 32 KiB minimum on modern hardware
- Formula for OPT 6.7B: `2 x d_model x 4 bytes = 32 KiB` per neuron load

---

## 8. Experimental Results

### OPT 6.7B on M1 Max CPU

| Phase | Naive (ms) | Optimized (ms) |
|-------|-----------|----------------|
| I/O | 2,196 | 105 |
| Memory mgmt | - | 57 |
| Compute | 986 | 506 |
| **Total** | **3,182** | **669** |

### GPU Results

| Backend | Naive (ms) | Optimized (ms) |
|---------|-----------|----------------|
| Metal GPU | 2,389 | 305 |
| NVIDIA RTX 4090 | 2,218 | 84 |

### Models Tested
OPT 6.7B, Falcon 7B (sparsified), Persimmon 8B, Phi-2, Llama 2 7B (FATReLU sparsified)

### 1000-Token Generation
No thermal throttling observed. Flash latency doesn't increase over long generation. Higher for first few tokens since allocated DRAM is empty.

---

## 9. Speculative Decoding Integration

- Receive lambda=4 draft tokens, verify with full model
- Keep window of size k ending at alpha(lambda+1)-th token (alpha = acceptance ratio)
- Result: 1.4x speedup (vs 1.58x for standard speculative decoding)

---

## 10. Negative Results

### Co-Activation Bundling (failed)
- Attempted neuron bundling based on "closest friend" (highest coactivation)
- Power-law coactivation distribution: highly active neurons are closest friends of many
- Result: multiple redundant loadings of popular neurons
- **Conclusion: Strategy ineffective**

---

## 11. Limitations (acknowledged by authors)

- Single-sequence inference only
- Assumes 50% DRAM availability
- Higher total energy consumption despite lower power (extended execution time)
- Limited exploration of power/thermal effects on edge devices
- No mixture-of-experts models evaluated

---

## 12. Key Architectural Insight

The method exploits three orthogonal bottlenecks:
1. **Data volume reduction** — sparsity + windowing
2. **Throughput improvement** — row-column bundling (larger sequential reads)
3. **Memory efficiency** — pointer remapping (no reallocation)

Each addresses different hardware characteristics of flash vs DRAM.
