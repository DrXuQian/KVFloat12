"""
ZipServ-inspired approach: tight bit-packing to break the 18.75% ceiling.

Key insight we missed: KVFloat13 stores 13 bits/value but uses byte-aligned
streams, which happens to be perfectly packed (13×128/8 = 208 exactly).
KVFloat12 SHOULD be 12×128/8 = 192 bytes, but our byte-aligned layout
wastes space (208 bytes = same as KVFloat13!).

The fix: TRUE bit-packing.

Schemes to test:
  A) 12-bit packed: sign(1)+exp4(4)+mant(7) = 12 bits → 192B/block = 25%
  B) 10-bit packed: exp3(3)+mant(7) = 10 bits + sign(1) separate → 176B = 31.25%
  C) 11-bit packed: exp3(3)+mant(7)+sign(1) = 11 bits → 176B = 31.25%

For (B/C), 3-bit exp = 8 entries per-block table. Need to verify per-block
top-8 coverage is sufficient.
"""
import torch, numpy as np, sys, os, time, json, math
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset
from collections import Counter

model_path = "/root/autodl-tmp/Qwen3-4B"

# ============================================================
# Bit-packing encode/decode
# ============================================================

def pack_12bit(values_12bit):
    """Pack array of 12-bit values into bytes. 2 values → 3 bytes."""
    n = len(values_12bit)
    assert n % 2 == 0
    v = values_12bit.astype(np.uint16)
    # Pair up: v[0],v[1] → 3 bytes
    even = v[0::2]
    odd = v[1::2]
    b0 = (even >> 4).astype(np.uint8)              # top 8 of even
    b1 = ((even & 0xF) << 4 | (odd >> 8)).astype(np.uint8)  # bot 4 of even + top 4 of odd
    b2 = (odd & 0xFF).astype(np.uint8)              # bot 8 of odd
    return np.stack([b0, b1, b2], axis=-1).reshape(-1)

def unpack_12bit(packed, n_values):
    """Unpack 12-bit packed bytes back to uint16 array."""
    packed = packed.reshape(-1, 3)
    b0, b1, b2 = packed[:, 0].astype(np.uint16), packed[:, 1].astype(np.uint16), packed[:, 2].astype(np.uint16)
    even = (b0 << 4) | (b1 >> 4)
    odd = ((b1 & 0xF) << 8) | b2
    result = np.empty(len(packed) * 2, dtype=np.uint16)
    result[0::2] = even
    result[1::2] = odd
    return result[:n_values]

def pack_10bit(values_10bit):
    """Pack 10-bit values. 4 values → 5 bytes."""
    n = len(values_10bit)
    assert n % 4 == 0
    v = values_10bit.astype(np.uint16).reshape(-1, 4)
    b0 = (v[:, 0] >> 2).astype(np.uint8)
    b1 = (((v[:, 0] & 0x3) << 6) | (v[:, 1] >> 4)).astype(np.uint8)
    b2 = (((v[:, 1] & 0xF) << 4) | (v[:, 2] >> 6)).astype(np.uint8)
    b3 = (((v[:, 2] & 0x3F) << 2) | (v[:, 3] >> 8)).astype(np.uint8)
    b4 = (v[:, 3] & 0xFF).astype(np.uint8)
    return np.stack([b0, b1, b2, b3, b4], axis=-1).reshape(-1)

def unpack_10bit(packed, n_values):
    """Unpack 10-bit packed bytes."""
    packed = packed.reshape(-1, 5)
    b = packed.astype(np.uint16)
    v0 = (b[:, 0] << 2) | (b[:, 1] >> 6)
    v1 = ((b[:, 1] & 0x3F) << 4) | (b[:, 2] >> 4)
    v2 = ((b[:, 2] & 0xF) << 6) | (b[:, 3] >> 2)
    v3 = ((b[:, 3] & 0x3) << 8) | b[:, 4]
    result = np.empty(len(packed) * 4, dtype=np.uint16)
    result[0::4] = v0
    result[1::4] = v1
    result[2::4] = v2
    result[3::4] = v3
    return result[:n_values]

# ============================================================
# Scheme A: 12-bit packed (sign+exp4+mant7, global LUT)
# ============================================================

def scheme_a_roundtrip(bf16_u16, compress_lut, decompress_lut):
    """12-bit packed KVFloat12. Returns decoded bf16 uint16."""
    sign = (bf16_u16 >> 15) & 1
    exp8 = (bf16_u16 >> 7) & 0xFF
    mant7 = bf16_u16 & 0x7F
    exp4 = compress_lut[exp8.astype(np.uint8)]
    # Pack: [sign(1)|exp4(4)|mant7(7)] = 12 bits
    packed_val = (sign.astype(np.uint16) << 11) | (exp4.astype(np.uint16) << 7) | mant7.astype(np.uint16)
    # Pack to bytes
    packed_bytes = pack_12bit(packed_val)
    # Unpack
    unpacked = unpack_12bit(packed_bytes, len(bf16_u16))
    # Decode
    sign_d = (unpacked >> 11) & 1
    exp4_d = (unpacked >> 7) & 0xF
    mant7_d = unpacked & 0x7F
    exp8_d = decompress_lut[exp4_d.astype(np.uint8)].astype(np.uint16)
    return ((sign_d << 15) | (exp8_d << 7) | mant7_d).astype(np.uint16)

def scheme_a_size(n_values):
    """Compressed size in bytes for scheme A."""
    return n_values * 12 // 8  # 12 bits per value, tightly packed

# ============================================================
# Scheme B: 10-bit packed (exp3+mant7, per-block table, sign separate)
# ============================================================

def scheme_b_roundtrip(bf16_u16):
    """10-bit packed per-block. Returns decoded bf16 uint16."""
    n = len(bf16_u16)
    assert n % 128 == 0
    num_blocks = n // 128
    result = np.zeros(n, dtype=np.uint16)

    for bi in range(num_blocks):
        block = bf16_u16[bi*128:(bi+1)*128]
        sign = (block >> 15) & 1
        exp8 = ((block >> 7) & 0xFF).astype(int)
        mant7 = block & 0x7F

        # Find top-8 exponents in this block
        exp_counter = Counter(exp8.tolist())
        top8 = [e for e, _ in exp_counter.most_common(8)]
        exp_table = np.array(sorted(top8), dtype=np.uint8)

        # Build local compress LUT
        local_compress = {}
        for i, e in enumerate(exp_table):
            local_compress[int(e)] = i

        # Map exponents
        exp3 = np.zeros(128, dtype=np.uint8)
        for j in range(128):
            e = int(exp8[j])
            if e in local_compress:
                exp3[j] = local_compress[e]
            else:
                # Clamp to nearest in table
                best_idx = min(range(len(exp_table)),
                               key=lambda k: abs(int(exp_table[k]) - e))
                exp3[j] = best_idx

        # Pack: [exp3(3)|mant7(7)] = 10 bits
        packed_val = (exp3.astype(np.uint16) << 7) | mant7.astype(np.uint16)
        packed_bytes = pack_10bit(packed_val)

        # Unpack
        unpacked = unpack_10bit(packed_bytes, 128)
        exp3_d = (unpacked >> 7) & 0x7
        mant7_d = unpacked & 0x7F
        exp8_d = exp_table[exp3_d].astype(np.uint16)

        result[bi*128:(bi+1)*128] = ((sign.astype(np.uint16) << 15) | (exp8_d << 7) | mant7_d)

    return result

def scheme_b_size(n_values):
    """Size: 10-bit data + 16 sign bytes + 8 table bytes per block."""
    n_blocks = n_values // 128
    return n_values * 10 // 8 + n_blocks * (16 + 8)  # packed data + signs + exp_table

# ============================================================
# Scheme B-fast: vectorized per-block (no Python loop)
# ============================================================

def scheme_b_gpu(tensor, device='cuda'):
    """GPU-friendly 10-bit scheme: per-block top-8 + clamp."""
    shape = tensor.shape
    raw = tensor.contiguous().view(-1).view(torch.int16).to(torch.int32)
    n = raw.shape[0]
    pad = (128 - n % 128) % 128
    if pad:
        raw = torch.cat([raw, torch.zeros(pad, dtype=torch.int32, device=raw.device)])

    sign = (raw >> 15) & 1
    exp8 = (raw >> 7) & 0xFF
    mant7 = raw & 0x7F

    blocks_exp = exp8.reshape(-1, 128)
    blocks_sign = sign.reshape(-1, 128)
    blocks_mant = mant7.reshape(-1, 128)
    num_blocks = blocks_exp.shape[0]

    # Per-block: find top-8 exponents and build mapping
    result_exp = torch.zeros_like(blocks_exp)
    for bi in range(num_blocks):
        bexp = blocks_exp[bi]
        unique_exps, inverse, counts = bexp.unique(return_inverse=True, return_counts=True)
        # Top-8 by frequency
        if len(unique_exps) <= 8:
            # All fit, direct mapping
            sorted_exps = unique_exps.sort().values
            # Map inverse through sorting
            sort_map = torch.zeros(256, dtype=torch.int32, device=bexp.device)
            for i, e in enumerate(sorted_exps):
                sort_map[e] = i
            result_exp[bi] = sort_map[bexp]
        else:
            top8_idx = counts.argsort(descending=True)[:8]
            top8_exps = unique_exps[top8_idx].sort().values
            # Build mapping
            exp_map = torch.zeros(256, dtype=torch.int32, device=bexp.device)
            for i, e in enumerate(top8_exps):
                exp_map[e] = i
            # For non-top-8, find nearest
            top8_set = set(top8_exps.cpu().tolist())
            for e in unique_exps.cpu().tolist():
                if e not in top8_set:
                    # Nearest in top8
                    dists = (top8_exps - e).abs()
                    exp_map[e] = dists.argmin()
            result_exp[bi] = exp_map[bexp]

    # Reconstruct with top-8 table
    # We need to store the tables to decode, but for PPL test we can
    # just decode immediately using the same mapping
    # For each block, get the actual top-8 exponents used
    decoded_exp = torch.zeros_like(blocks_exp)
    for bi in range(num_blocks):
        bexp = blocks_exp[bi]
        unique_exps, _, counts = bexp.unique(return_inverse=True, return_counts=True)
        if len(unique_exps) <= 8:
            sorted_exps = unique_exps.sort().values
        else:
            top8_idx = counts.argsort(descending=True)[:8]
            sorted_exps = unique_exps[top8_idx].sort().values
        decoded_exp[bi] = sorted_exps[result_exp[bi].clamp(0, len(sorted_exps)-1)]

    bf16_out = (blocks_sign << 15) | (decoded_exp << 7) | blocks_mant
    return bf16_out.reshape(-1)[:n].to(torch.int16).view(torch.bfloat16).reshape(shape)

# ============================================================

print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

# ============================================================
# Part 1: Per-block top-K coverage analysis
# ============================================================
print(f"\n{'='*65}")
print("PER-BLOCK TOP-K EXPONENT COVERAGE (KV cache)")
print(f"{'='*65}")

all_block_coverages = {4: [], 6: [], 8: [], 12: [], 16: []}
cal_prompts = [
    "The quick brown fox jumps over the lazy dog in a warm summer afternoon.",
    "def quicksort(arr): return [] if not arr else quicksort([x for x in arr[1:] if x < arr[0]])",
    "In machine learning, neural networks are trained using backpropagation.",
    "The stock market experienced significant volatility today.",
    "Once upon a time in a distant kingdom there lived a wise wizard.",
]

with torch.no_grad():
    for p in cal_prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        outputs = model(**inputs, use_cache=True)
        for layer in outputs.past_key_values.layers:
            for t in [layer.keys, layer.values]:
                raw = t.view(torch.int16).cpu().numpy().astype(np.uint16).flatten()
                n = len(raw)
                pad = (128 - n % 128) % 128
                if pad:
                    raw = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)])
                for bi in range(len(raw) // 128):
                    block = raw[bi*128:(bi+1)*128]
                    exps = ((block >> 7) & 0xFF).astype(int)
                    exp_counter = Counter(exps.tolist())
                    total = len(exps)
                    for k in all_block_coverages:
                        topk = [e for e, _ in exp_counter.most_common(k)]
                        cov = sum(exp_counter[e] for e in topk) / total
                        all_block_coverages[k].append(cov)

print(f"{'Top-K':>6} | {'Mean cov':>8} | {'Min cov':>8} | {'P5 cov':>8} | {'Bits needed':>11} | {'Block size':>10} | {'Compress':>8}")
print("-" * 80)
for k in sorted(all_block_coverages.keys()):
    covs = np.array(all_block_coverages[k])
    bits = max(1, int(math.ceil(math.log2(k))))
    # Block size: packed(bits+7 per value) + sign(16B) + table(k bytes)
    packed_bits = bits + 7  # exp_idx + mant
    packed_bytes = 128 * packed_bits // 8
    block_bytes = packed_bytes + 16 + k  # data + signs + exp_table
    comp = 100 * (1 - block_bytes / 256)
    print(f"  {k:>4} | {covs.mean():>7.4f} | {covs.min():>7.4f} | {np.percentile(covs, 5):>7.4f} | "
          f"{bits:>5} bits | {block_bytes:>5} B | {comp:>7.2f}%")

# ============================================================
# Part 2: Packing efficiency comparison
# ============================================================
print(f"\n{'='*65}")
print("PACKING EFFICIENCY")
print(f"{'='*65}")

layouts = [
    ("BF16 (baseline)", 256, 0),
    ("KVFloat13 (byte-aligned)", 208, 18.75),
    ("KVFloat12 (byte-aligned)", 208, 18.75),  # same due to byte alignment!
    ("12-bit packed (4-bit exp)", 192, 25.00),
    ("11-bit packed (4-bit exp, sign merged)", 176, 31.25),
    ("10-bit packed (3-bit exp) + sign + table", 160+16+8, 28.125),
    ("Huffman (theoretical)", int(128*11.34/8), 29.1),
]

print(f"{'Layout':<42} | {'Block size':>10} | {'Compress':>8} | {'bits/val':>8}")
print("-" * 75)
for name, size, comp in layouts:
    bpv = size * 8 / 128
    print(f"{name:<42} | {size:>7} B | {comp:>7.2f}% | {bpv:>7.2f}")

# ============================================================
# Part 3: Verify packing correctness
# ============================================================
print(f"\n{'='*65}")
print("PACKING ROUNDTRIP VERIFICATION")
print(f"{'='*65}")

# Test 12-bit packing
rng = np.random.RandomState(42)
test_vals = rng.randint(0, 4096, 256).astype(np.uint16)
packed = pack_12bit(test_vals)
unpacked = unpack_12bit(packed, 256)
assert np.all(test_vals == unpacked), "12-bit pack/unpack FAILED"
print(f"  12-bit: {len(test_vals)} values → {len(packed)} bytes → roundtrip OK")
print(f"    Ratio: {len(packed)/len(test_vals)/2:.4f} ({100*(1-len(packed)/len(test_vals)/2):.2f}% compression)")

# Test 10-bit packing
test_vals10 = rng.randint(0, 1024, 256).astype(np.uint16)
packed10 = pack_10bit(test_vals10)
unpacked10 = unpack_10bit(packed10, 256)
assert np.all(test_vals10 == unpacked10), "10-bit pack/unpack FAILED"
print(f"  10-bit: {len(test_vals10)} values → {len(packed10)} bytes → roundtrip OK")
print(f"    Ratio: {len(packed10)/len(test_vals10)/2:.4f} ({100*(1-len(packed10)/len(test_vals10)/2):.2f}% compression)")

# ============================================================
# Part 4: PPL test — 10-bit scheme on KV cache
# ============================================================
print(f"\n{'='*65}")
print("DECODE PPL — 10-bit packed (3-bit exp per-block)")
print(f"{'='*65}")

# Build global LUT for comparison
kv_counter = Counter()
with torch.no_grad():
    for p in cal_prompts[:3]:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        out = model(**inputs, use_cache=True)
        for layer in out.past_key_values.layers:
            for t in [layer.keys, layer.values]:
                exps = ((t.view(torch.int16).to(torch.int32) >> 7) & 0xFF)
                kv_counter.update(exps.cpu().numpy().flatten().tolist())

top32 = [e for e, _ in kv_counter.most_common(32)]
kvf13_decompress = np.array(sorted(top32), dtype=np.uint8)
kvf13_compress = np.zeros(256, dtype=np.uint8)
idx = {e: i for i, e in enumerate(sorted(top32))}
for e, i in idx.items():
    kvf13_compress[e] = i
for e in range(256):
    if e in idx: continue
    best_i, best_d = 0, float('inf')
    for se in sorted(top32):
        d = abs(2.0**(e-127) - 2.0**(se-127))
        if d < best_d:
            best_d, best_i = d, idx[se]
    kvf13_compress[e] = best_i

kvf13_emap = torch.zeros(256, dtype=torch.int32, device='cuda')
for e in range(256):
    kvf13_emap[e] = kvf13_decompress[kvf13_compress[e]]

def compress_kvf13_gpu(cache):
    for layer in cache.layers:
        for attr in ['keys', 'values']:
            t = getattr(layer, attr)
            raw = t.view(torch.int16).to(torch.int32)
            sign = (raw >> 15) & 1
            exp8 = (raw >> 7) & 0xFF
            mant7 = raw & 0x7F
            exp_new = kvf13_emap[exp8]
            setattr(layer, attr, ((sign << 15) | (exp_new << 7) | mant7).to(torch.int16).view(torch.bfloat16))

def compress_10bit_gpu(cache):
    for layer in cache.layers:
        layer.keys = scheme_b_gpu(layer.keys)
        layer.values = scheme_b_gpu(layer.values)

# Load data
print("\nLoading Wikitext-2...")
try:
    ds = load_dataset("/root/autodl-tmp/data/datasets/wikitext", "wikitext-2-raw-v1", split="test")
except:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"])
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

MAX_TOKENS = 4096

def eval_decode_ppl(model, input_ids, compress_fn=None, max_tokens=4096):
    seq_len = min(input_ids.shape[1], max_tokens)
    nlls = []
    t0 = time.time()
    cache = DynamicCache()
    model.eval()
    with torch.no_grad():
        for t in range(seq_len - 1):
            tok = input_ids[:, t:t+1]
            out = model(tok, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if compress_fn is not None:
                compress_fn(cache)
            logits = out.logits[:, -1, :]
            target = input_ids[:, t+1]
            loss = torch.nn.functional.cross_entropy(logits, target)
            nlls.append(loss.item())
            if (t+1) % 1000 == 0:
                ppl = math.exp(sum(nlls) / len(nlls))
                print(f"    [{t+1}/{seq_len}] ppl={ppl:.4f} ({time.time()-t0:.0f}s)")
    return math.exp(sum(nlls) / len(nlls)), time.time() - t0

# Baseline
print("\n1) BF16 baseline:")
ppl_base, t = eval_decode_ppl(model, input_ids, max_tokens=MAX_TOKENS)
print(f"  PPL={ppl_base:.4f} ({t:.0f}s)")
torch.cuda.empty_cache()

# KVFloat13
print("\n2) KVFloat13 (13-bit, 18.75%):")
ppl_13, t = eval_decode_ppl(model, input_ids, compress_kvf13_gpu, MAX_TOKENS)
print(f"  PPL={ppl_13:.4f} (Δ{ppl_13-ppl_base:+.4f}, {t:.0f}s)")
torch.cuda.empty_cache()

# FP8
print("\n3) FP8 E4M3 (8-bit, 50%):")
def compress_fp8(cache):
    for layer in cache.layers:
        layer.keys = layer.keys.to(torch.float8_e4m3fn).to(torch.bfloat16)
        layer.values = layer.values.to(torch.float8_e4m3fn).to(torch.bfloat16)
ppl_fp8, t = eval_decode_ppl(model, input_ids, compress_fp8, MAX_TOKENS)
print(f"  PPL={ppl_fp8:.4f} (Δ{ppl_fp8-ppl_base:+.4f}, {t:.0f}s)")
torch.cuda.empty_cache()

# 10-bit per-block
print("\n4) 10-bit packed per-block top-8 (10-bit, ~28%):")
ppl_10, t = eval_decode_ppl(model, input_ids, compress_10bit_gpu, MAX_TOKENS)
print(f"  PPL={ppl_10:.4f} (Δ{ppl_10-ppl_base:+.4f}, {t:.0f}s)")

# Summary
print(f"\n{'='*65}")
print(f"RESULTS — KV Cache Compression ({MAX_TOKENS} tokens)")
print(f"{'='*65}")
print(f"{'Method':<35} | {'bits':>4} | {'Compress':>8} | {'PPL':>8} | {'Δ PPL':>8}")
print("-" * 70)
print(f"{'BF16 (baseline)':<35} | {'16':>4} | {'0%':>8} | {ppl_base:>8.4f} | {'—':>8}")
print(f"{'KVFloat13 (5-bit exp)':<35} | {'13':>4} | {'18.75%':>8} | {ppl_13:>8.4f} | {ppl_13-ppl_base:>+8.4f}")
print(f"{'10-bit packed (3-bit exp/block)':<35} | {'10':>4} | {'~28%':>8} | {ppl_10:>8.4f} | {ppl_10-ppl_base:>+8.4f}")
print(f"{'FP8 E4M3':<35} | {'8':>4} | {'50%':>8} | {ppl_fp8:>8.4f} | {ppl_fp8-ppl_base:>+8.4f}")

with open("/root/kvfloat13/zipserv_results.json", "w") as f:
    json.dump({
        "model": "Qwen3-4B", "tokens": MAX_TOKENS,
        "baseline": ppl_base,
        "kvfloat13": {"ppl": ppl_13, "bits": 13, "compress": "18.75%"},
        "packed_10bit": {"ppl": ppl_10, "bits": 10, "compress": "~28%"},
        "fp8_e4m3": {"ppl": ppl_fp8, "bits": 8, "compress": "50%"},
    }, f, indent=2)
print("\nSaved to zipserv_results.json")
