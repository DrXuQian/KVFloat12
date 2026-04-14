# KVFloat13 / KVFloat12: Near-Lossless BF16 Compression for LLM Inference

## What is this?

BF16 uses 8-bit exponent, but real LLM tensors (weights and KV cache) only use ~30-32 unique exponent values out of 256. We exploit this by compressing the exponent from 8 bits to 5 bits (KVFloat13) or 4 bits (KVFloat12), keeping the 7-bit mantissa untouched.

```
BF16:       sign(1) + exp(8) + mantissa(7) = 16 bits
KVFloat13:  sign(1) + exp(5) + mantissa(7) = 13 bits  → 18.75% compression
KVFloat12:  sign(1) + exp(4) + mantissa(7) = 12 bits  → 25.0%  compression
```

Encode/decode uses a simple LUT: `exp5 = compress_lut[exp8]` / `exp8 = decompress_lut[exp5]`.

## Key Results

### Weight Compression — MMLU (Qwen3-4B, lm-eval 5-shot)

| Method | MMLU Acc | Δ Acc | Compression |
|--------|----------|-------|-------------|
| BF16 baseline | 70.12% | — | 0% |
| KVFloat13 (5-bit) | 70.09% | -0.03pp | 18.75% |
| KVFloat12 LUT [117-132] | 69.85% | -0.27pp | 25.0% |

### KV Cache Compression — Decode PPL (Qwen3-4B, Wikitext-2)

| Method | Bits | Compression | PPL (4K) | Δ |
|--------|------|-------------|----------|---|
| BF16 | 16 | 0% | 12.284 | — |
| KVFloat13 | 13 | 18.75% | 12.285 | +0.0001 |
| FP8 E4M3 | 8 | 50% | 12.097 | -0.188 |
| FP8 E5M2 | 8 | 50% | 12.356 | +0.072 |

### Long-Context Scaling (eval on last 1024 tokens)

| Context | BF16 | KVFloat13 Δ | FP8 E4M3 Δ |
|---------|------|-------------|------------|
| 4K | 16.05 | +0.042 | -0.227 |
| 8K | 12.70 | +0.022 | -0.069 |
| 16K | 9.01 | +0.008 | -0.034 |
| 32K | 13.44 | -0.022 | -0.191 |

KVFloat13 error stays near-zero across all context lengths. FP8 shows small PPL "improvement" (regularization effect from mantissa quantization noise).

## How It Works

### Storage Layout (per block of 128 values = 208 bytes)

```
signs[16]        — 128 × 1-bit sign, packed 8/byte
exp_hi[64]       — 128 × 4-bit (top 4 bits of exp5), nibble-packed
exp_lo_mant[128] — 128 × [exp_low(1) | mantissa(7)], 1 byte each
```

### Encode
```python
exp5 = compress_lut[exp8]        # 256-entry LUT, exp8 → 5-bit index
exp_hi4 = exp5 >> 1
exp_lo1 = exp5 & 1
em_byte = (exp_lo1 << 7) | mant7  # pack into stream3
```

### Decode
```python
exp5 = (exp_hi4 << 1) | (em_byte >> 7)
exp8 = decompress_lut[exp5]       # 32-entry LUT → original 8-bit exponent
mant7 = em_byte & 0x7F            # mantissa untouched
bf16 = (sign << 15) | (exp8 << 7) | mant7
```

## Design Decisions & Findings

### LUT Window Selection is Critical (not frequency-based!)

Selecting the LUT by **exponent frequency** (most common exponents) is WRONG — it misses large-value exponents (exp ≥ 128) that are rare but critical. This causes PPL to explode from 6.6 to 9594.

The correct approach: **slide a contiguous window to cover the top (large-value) exponents**. The optimal window varies by model:
- SmolLM-135M: [115-130]
- Qwen3-4B: [117-132]

### KVFloat12 (4-bit) Challenges for KV Cache

KV cache has wider exponent range than weights (up to exp=135 for Qwen3-4B), and errors accumulate during autoregressive decoding. A global 4-bit window gives PPL=13.0 (+0.73) vs KVFloat13's +0.0001.

### Per-layer LUT Windows Don't Help (Yet)

Per-layer windows for KVFloat12 actually **hurt** (PPL=20.6) due to overfitting to calibration data. More calibration data or a smarter selection strategy is needed.

### Per-block Base+Offset (Alternative to LUT)

Stores `base_exp` (1 byte) per block of 128 values + 4-bit offset per value. Fully adaptive, no calibration needed. 99.3% of weight blocks fit in 4-bit offset. Similar compression to KVFloat13 (18.4%) but slightly worse quality.

## File Guide

### Core Implementation
| File | Description |
|------|-------------|
| `task2_lut.py` | LUT construction from exponent frequencies |
| `task3_encode_decode.py` | Python encode/decode for KVFloat13 |
| `split_lut_kvfloat12.py` | KVFloat12 (4-bit) encode/decode |
| `per_block_base_offset.py` | Per-block base+offset adaptive scheme |
| `task5_cuda_kernel.cu` | CUDA decode kernel + fused dot product |
| `task6_neon_kernel.c` | ARM NEON + x86 AVX2 decode kernels |

### Analysis & Experiments
| File | Description |
|------|-------------|
| `task1_exponent_analysis.py` | Exponent distribution analysis (SmolLM-135M) |
| `per_layer_analysis.py` | Per-layer exponent range analysis |
| `per_token_kv_analysis.py` | KV cache per-token exponent variation |
| `kv_per_layer_qwen3.py` | Per-layer KV analysis on Qwen3-4B |
| `analyze_fewer_bits.py` | Can we use fewer than 5 bits? |
| `fix_lut_selection.py` | LUT window sliding experiment |
| `smart_lut_test.py` | Non-contiguous LUT strategies |
| `adaptive_analysis.py` | Per-block span feasibility |

### Evaluation
| File | Description |
|------|-------------|
| `test_qwen3_fast.py` | Vectorized encode/decode + PPL on Qwen3-4B |
| `test_mmlu_lmeval.py` | MMLU via lm-eval (weights compression) |
| `test_mmlu_remaining.py` | MMLU: base+offset + KVFloat13 |
| `test_kv_ppl_decode.py` | KV cache decode PPL (token-by-token) |
| `test_kv_fp8.py` | FP8 E4M3/E5M2 vs KVFloat13 comparison |
| `test_kv_long_context.py` | Long-context (4K-32K) PPL scaling |
| `test_kv_ppl_perlayer.py` | Per-layer KVFloat12 KV cache test |

### Results (JSON)
| File | Description |
|------|-------------|
| `mmlu_lmeval_results.json` | MMLU accuracy (lm-eval) |
| `kv_comparison_results.json` | KV cache: KVFloat13 vs FP8 |
| `long_context_results.json` | PPL at 4K/8K/16K/32K context |
| `wikitext_kv_results.json` | Wikitext-2 decode PPL |

## Models Tested

- **SmolLM-135M** (HuggingFaceTB/SmolLM-135M): Initial development, 100% bit-exact with 5-bit LUT
- **Qwen3-4B** (/root/autodl-tmp/Qwen3-4B): Full evaluation, MMLU + PPL + long-context
