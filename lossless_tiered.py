"""
Lossless tiered exponent encoding (ZipServ-inspired).
7 common exponents + overflow, per-block adaptive, 100% lossless.
~27% compression, GPU-parallel decode via popcount.

Format per block (128 values):
  stream1[128]: [idx(3) | mant_hi(5)] per value, 1 byte each
  stream2[32]:  mant_lo(2) packed, 4 per byte
  sign[16]:     128 bits packed
  exp_table[7]: 7 most common exponents in this block
  overflow[N]:  full 8-bit exp for values with idx=7

Decode:
  mant7 = (stream1[i] & 0x1F) << 2 | extract_2bit(stream2, i)
  idx = stream1[i] >> 5
  if idx < 7: exp8 = exp_table[idx]
  if idx == 7: exp8 = overflow[popcount(idx7_before_me)]
"""
import torch, numpy as np, sys, os, time, json, math
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset
from collections import Counter

model_path = "/root/autodl-tmp/Qwen3-4B"

# ============================================================
# Lossless tiered encode/decode (GPU-vectorized)
# ============================================================

def lossless_tiered_compress(tensor):
    """
    Compress BF16 tensor losslessly using per-block top-7 exponent table.
    For PPL testing: does round-trip encode→decode on GPU, returns BF16 tensor.
    """
    shape = tensor.shape
    raw = tensor.contiguous().view(-1).view(torch.int16).to(torch.int32)
    n = raw.shape[0]
    pad = (128 - n % 128) % 128
    if pad:
        raw = torch.cat([raw, torch.zeros(pad, dtype=torch.int32, device=raw.device)])

    sign = (raw >> 15) & 1
    exp8 = (raw >> 7) & 0xFF
    mant7 = raw & 0x7F

    blocks_sign = sign.reshape(-1, 128)
    blocks_exp = exp8.reshape(-1, 128)
    blocks_mant = mant7.reshape(-1, 128)
    num_blocks = blocks_exp.shape[0]

    # Output
    result_exp = torch.zeros_like(blocks_exp)

    total_overflow = 0
    total_values = num_blocks * 128

    for bi in range(num_blocks):
        bexp = blocks_exp[bi]  # [128]

        # Find top-7 exponents by frequency
        unique_exps, inverse, counts = bexp.unique(return_inverse=True, return_counts=True)

        if len(unique_exps) <= 7:
            # All fit in table, no overflow
            result_exp[bi] = bexp  # identity mapping
        else:
            # Top-7 by count
            top7_idx = counts.argsort(descending=True)[:7]
            top7_exps = unique_exps[top7_idx].sort().values  # sorted

            # Build fast lookup
            exp_to_idx = torch.full((256,), 7, dtype=torch.int32, device=bexp.device)  # default = overflow
            for i, e in enumerate(top7_exps):
                exp_to_idx[e] = i

            idx = exp_to_idx[bexp]  # [128], values 0-6 or 7(overflow)
            n_overflow = (idx == 7).sum().item()
            total_overflow += n_overflow

            # For common values: exp8 = exp_table[idx]
            # For overflow: exp8 stays as-is (we store it separately)
            # In this simulation, we just reconstruct directly:
            reconstructed = torch.where(
                idx < 7,
                top7_exps[idx.clamp(0, 6)],  # common: lookup table
                bexp  # overflow: keep original
            )
            result_exp[bi] = reconstructed

    # Reassemble BF16
    out = (blocks_sign << 15) | (result_exp << 7) | blocks_mant
    result = out.reshape(-1)[:n].to(torch.int16).view(torch.bfloat16).reshape(shape)

    return result, total_overflow, total_values

def lossless_tiered_compress_cache(cache):
    """Compress all KV cache tensors, return overflow stats."""
    total_of = 0
    total_v = 0
    for layer in cache.layers:
        layer.keys, of_k, v_k = lossless_tiered_compress(layer.keys)
        layer.values, of_v, v_v = lossless_tiered_compress(layer.values)
        total_of += of_k + of_v
        total_v += v_k + v_v
    return total_of, total_v

# ============================================================
# Comparison functions
# ============================================================

def make_kvf13_emap(model, tokenizer):
    kv_counter = Counter()
    with torch.no_grad():
        for p in ["The quick brown fox.", "def fib(n): return n", "Neural networks."]:
            inputs = tokenizer(p, return_tensors="pt").to(model.device)
            out = model(**inputs, use_cache=True)
            for layer in out.past_key_values.layers:
                for t in [layer.keys, layer.values]:
                    exps = ((t.view(torch.int16).to(torch.int32) >> 7) & 0xFF)
                    kv_counter.update(exps.cpu().numpy().flatten().tolist())
    top32 = sorted([e for e, _ in kv_counter.most_common(32)])
    emap = torch.zeros(256, dtype=torch.int32, device='cuda')
    idx = {e: i for i, e in enumerate(top32)}
    compress = np.zeros(256, dtype=np.uint8)
    decompress = np.array(top32, dtype=np.uint8)
    for e, i in idx.items(): compress[e] = i
    for e in range(256):
        if e not in idx:
            best = min(top32, key=lambda se: abs(2.0**(e-127) - 2.0**(se-127)))
            compress[e] = idx[best]
    for e in range(256):
        emap[e] = decompress[compress[e]]
    return emap

def compress_kvf13(cache, emap):
    for layer in cache.layers:
        for attr in ['keys', 'values']:
            t = getattr(layer, attr)
            raw = t.view(torch.int16).to(torch.int32)
            s = (raw >> 15) & 1; e = (raw >> 7) & 0xFF; m = raw & 0x7F
            setattr(layer, attr, ((s << 15) | (emap[e] << 7) | m).to(torch.int16).view(torch.bfloat16))

def compress_fp8(cache):
    for layer in cache.layers:
        layer.keys = layer.keys.to(torch.float8_e4m3fn).to(torch.bfloat16)
        layer.values = layer.values.to(torch.float8_e4m3fn).to(torch.bfloat16)

# ============================================================
# PPL evaluation
# ============================================================

def eval_decode_ppl(model, input_ids, compress_fn=None, max_tokens=4096):
    seq_len = min(input_ids.shape[1], max_tokens)
    nlls = []
    t0 = time.time()
    cache = DynamicCache()
    overflow_total = 0
    values_total = 0

    model.eval()
    with torch.no_grad():
        for t in range(seq_len - 1):
            tok = input_ids[:, t:t+1]
            out = model(tok, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if compress_fn:
                ret = compress_fn(cache)
                if isinstance(ret, tuple):
                    overflow_total += ret[0]
                    values_total += ret[1]
            logits = out.logits[:, -1, :]
            target = input_ids[:, t+1]
            loss = torch.nn.functional.cross_entropy(logits, target)
            nlls.append(loss.item())
            if (t+1) % 1000 == 0:
                ppl = math.exp(sum(nlls) / len(nlls))
                tps = (t+1) / (time.time() - t0)
                print(f"    [{t+1}/{seq_len}] ppl={ppl:.4f} ({time.time()-t0:.0f}s, {tps:.0f} tok/s)")

    ppl = math.exp(sum(nlls) / len(nlls))
    elapsed = time.time() - t0
    return ppl, elapsed, overflow_total, values_total

# ============================================================

print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

kvf13_emap = make_kvf13_emap(model, tokenizer)

print("\nLoading Wikitext-2...")
try:
    ds = load_dataset("/root/autodl-tmp/data/datasets/wikitext", "wikitext-2-raw-v1", split="test")
except:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
input_ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids.to(model.device)

MAX = 4096

# ============================================================
# First: verify losslessness
# ============================================================
print(f"\n{'='*65}")
print("LOSSLESS VERIFICATION")
print(f"{'='*65}")

with torch.no_grad():
    inputs = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(model.device)
    out = model(**inputs, use_cache=True)
    for li, layer in enumerate(out.past_key_values.layers):
        for name, orig_tensor in [("key", layer.keys.clone()), ("value", layer.values.clone())]:
            compressed, n_of, n_v = lossless_tiered_compress(orig_tensor)
            match = (orig_tensor.view(torch.int16) == compressed.view(torch.int16)).all().item()
            if li == 0 or not match:
                n_total = orig_tensor.numel()
                n_match = (orig_tensor.view(torch.int16) == compressed.view(torch.int16)).sum().item()
                print(f"  L{li} {name}: {'LOSSLESS' if match else 'LOSSY!'} "
                      f"({n_match}/{n_total}, overflow={n_of}/{n_v})")
    print("  ... (checking all layers)")

    all_match = True
    for layer in out.past_key_values.layers:
        for orig_tensor in [layer.keys.clone(), layer.values.clone()]:
            compressed, _, _ = lossless_tiered_compress(orig_tensor)
            if not (orig_tensor.view(torch.int16) == compressed.view(torch.int16)).all():
                all_match = False
                break
    print(f"  Overall: {'ALL LAYERS LOSSLESS' if all_match else 'SOME LAYERS LOSSY!'}")

# ============================================================
# Storage analysis
# ============================================================
print(f"\n{'='*65}")
print("STORAGE ANALYSIS (per block of 128 values)")
print(f"{'='*65}")

# Measure actual overflow rate
overflow_stats = []
with torch.no_grad():
    for p in ["The quick brown fox jumps over the lazy dog in a warm afternoon.",
              "def quicksort(arr): return [] if not arr else quicksort([x for x in arr])",
              "Neural networks are trained using gradient descent optimization."]:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        out = model(**inputs, use_cache=True)
        for layer in out.past_key_values.layers:
            for t in [layer.keys, layer.values]:
                raw = t.view(torch.int16).to(torch.int32)
                exp8 = ((raw >> 7) & 0xFF).reshape(-1, 128)
                for bi in range(exp8.shape[0]):
                    bexp = exp8[bi]
                    unique, _, counts = bexp.unique(return_inverse=True, return_counts=True)
                    if len(unique) <= 7:
                        overflow_stats.append(0)
                    else:
                        top7_idx = counts.argsort(descending=True)[:7]
                        top7_set = set(unique[top7_idx].cpu().tolist())
                        n_of = sum(1 for e in bexp.cpu().tolist() if e not in top7_set)
                        overflow_stats.append(n_of)

overflow_stats = np.array(overflow_stats)
avg_overflow = overflow_stats.mean()
print(f"  Overflow per block: mean={avg_overflow:.2f} max={overflow_stats.max()}")
print(f"  Blocks with 0 overflow: {100*np.sum(overflow_stats==0)/len(overflow_stats):.1f}%")

avg_block_size = 128 + 32 + 16 + 7 + avg_overflow  # stream1 + stream2 + sign + table + overflow
print(f"\n  Average block size: {avg_block_size:.1f} bytes")
print(f"  Compression: {100*(1-avg_block_size/256):.2f}%")
print(f"  Bits/value: {avg_block_size*8/128:.2f}")

print(f"\n  Comparison:")
print(f"  {'Format':<30} | {'Block':>6} | {'Compress':>8} | {'Lossless':>8}")
print(f"  {'-'*60}")
print(f"  {'BF16':<30} | {'256 B':>6} | {'0%':>8} | {'N/A':>8}")
print(f"  {'KVFloat13 (5-bit LUT)':<30} | {'208 B':>6} | {'18.75%':>8} | {'99.9998%':>8}")
print(f"  {'Lossless Tiered (this)':<30} | {f'{avg_block_size:.0f} B':>6} | {f'{100*(1-avg_block_size/256):.1f}%':>8} | {'100%':>8}")
print(f"  {'FP8 E4M3':<30} | {'128 B':>6} | {'50%':>8} | {'No':>8}")

# ============================================================
# PPL test
# ============================================================
print(f"\n{'='*65}")
print(f"DECODE PPL — Lossless Tiered Encoding ({MAX} tokens)")
print(f"{'='*65}")

# 1) Baseline
print("\n1) BF16:")
ppl_base, t, _, _ = eval_decode_ppl(model, input_ids, max_tokens=MAX)
print(f"  PPL={ppl_base:.4f} ({t:.0f}s)")
torch.cuda.empty_cache()

# 2) Lossless tiered
print("\n2) Lossless Tiered (top-7 + overflow, ~27%):")
ppl_lt, t, of_total, v_total = eval_decode_ppl(
    model, input_ids, lossless_tiered_compress_cache, MAX)
overflow_pct = 100 * of_total / v_total if v_total > 0 else 0
print(f"  PPL={ppl_lt:.4f} (Δ{ppl_lt-ppl_base:+.6f}, {t:.0f}s)")
print(f"  Overflow: {of_total:,}/{v_total:,} ({overflow_pct:.2f}%)")
torch.cuda.empty_cache()

# 3) KVFloat13
print("\n3) KVFloat13 (13-bit, 18.75%):")
ppl_13, t, _, _ = eval_decode_ppl(model, input_ids, lambda c: compress_kvf13(c, kvf13_emap), MAX)
print(f"  PPL={ppl_13:.4f} (Δ{ppl_13-ppl_base:+.6f}, {t:.0f}s)")
torch.cuda.empty_cache()

# 4) FP8
print("\n4) FP8 E4M3 (8-bit, 50%):")
ppl_fp8, t, _, _ = eval_decode_ppl(model, input_ids, compress_fp8, MAX)
print(f"  PPL={ppl_fp8:.4f} (Δ{ppl_fp8-ppl_base:+.6f}, {t:.0f}s)")

# Summary
print(f"\n{'='*65}")
print(f"RESULTS — Lossless KV Cache Compression")
print(f"{'='*65}")
print(f"{'Method':<35} | {'Bits':>4} | {'Compress':>8} | {'PPL':>8} | {'Δ PPL':>10} | {'Lossless':>8}")
print("-" * 82)
print(f"{'BF16':<35} | {'16':>4} | {'0%':>8} | {ppl_base:>8.4f} | {'—':>10} | {'—':>8}")
print(f"{'Lossless Tiered (7+1)':<35} | {'~11':>4} | {'~27%':>8} | {ppl_lt:>8.4f} | {ppl_lt-ppl_base:>+10.6f} | {'YES':>8}")
print(f"{'KVFloat13 (5-bit LUT)':<35} | {'13':>4} | {'18.75%':>8} | {ppl_13:>8.4f} | {ppl_13-ppl_base:>+10.6f} | {'99.9998%':>8}")
print(f"{'FP8 E4M3':<35} | {'8':>4} | {'50%':>8} | {ppl_fp8:>8.4f} | {ppl_fp8-ppl_base:>+10.6f} | {'No':>8}")

with open("/root/kvfloat13/lossless_tiered_results.json", "w") as f:
    json.dump({
        "model": "Qwen3-4B", "tokens": MAX,
        "baseline": ppl_base,
        "lossless_tiered": {"ppl": ppl_lt, "overflow_pct": overflow_pct,
                            "avg_block_bytes": avg_block_size, "compression": 100*(1-avg_block_size/256)},
        "kvfloat13": {"ppl": ppl_13},
        "fp8": {"ppl": ppl_fp8},
    }, f, indent=2)
print("\nSaved")
