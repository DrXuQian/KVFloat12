"""
Part 2 continued: per-block analysis for KV cache + per-head analysis
"""
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

def extract_exp8(t):
    return ((t.view(torch.int16).to(torch.int32) >> 7) & 0xFF).cpu().numpy().flatten()

model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
model.eval()

prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "def quicksort(arr): return [] if not arr else quicksort([x for x in arr[1:] if x < arr[0]])",
    "The integral of x^2 from 0 to 1 equals 1/3. Consider the Taylor expansion",
    "xkcd 1729 αβγδ ∫∫∫ 0xDEADBEEF !!!??? ===",
    "In the beginning, there was nothing but void and darkness. Then came light, and with it, the first stars formed in the cosmic dawn. Billions of years passed as galaxies collided.",
]

# ============================================================
# KV cache: per-HEAD-VECTOR analysis (head_dim=64)
# Each KV cache entry for one head at one position = 64 values
# ============================================================
print("=" * 70)
print("KV CACHE: PER-HEAD-VECTOR EXPONENT RANGE (head_dim=64)")
print("=" * 70)

head_spans = []
head_data = []  # (layer, kv, head, pos, min, max, span, unique)

with torch.no_grad():
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs, use_cache=True)
        seq_len = inputs["input_ids"].shape[1]

        for layer_idx, kv in enumerate(outputs.past_key_values):
            for kv_idx, kv_name in enumerate(["key", "value"]):
                tensor = kv[kv_idx].to(torch.bfloat16)
                # [batch, heads, seq, head_dim]
                _, h, s, d = tensor.shape
                for hi in range(h):
                    for si in range(s):
                        vec = tensor[0, hi, si, :]
                        exps = extract_exp8(vec)
                        nonzero = exps[exps != 0]
                        if len(nonzero) > 0:
                            mn, mx = int(nonzero.min()), int(nonzero.max())
                            span = mx - mn
                        else:
                            mn, mx, span = 0, 0, 0
                        head_spans.append(span)
                        head_data.append((layer_idx, kv_name, hi, si, mn, mx, span, len(np.unique(exps))))

head_spans = np.array(head_spans)
print(f"  Head dim: {d}")
print(f"  Total vectors: {len(head_spans):,}")
print(f"  Span: min={head_spans.min()} max={head_spans.max()} mean={head_spans.mean():.1f} median={np.median(head_spans):.0f}")
for threshold in [4, 8, 12, 15, 16, 20, 24]:
    pct = 100.0 * np.sum(head_spans <= threshold) / len(head_spans)
    print(f"  span ≤ {threshold:>2}: {pct:.2f}% of vectors")

# Show worst-case vectors
worst_idx = np.argsort(head_spans)[-10:][::-1]
print(f"\n  Worst-span vectors:")
for idx in worst_idx:
    d = head_data[idx]
    print(f"    L{d[0]:>2}.{d[1]:<5} head={d[2]} pos={d[3]:>2}: [{d[4]}-{d[5]}] span={d[6]} unique={d[7]}")

# ============================================================
# Weight: per-block detailed analysis
# ============================================================
print("\n" + "=" * 70)
print("WEIGHTS: PER-BLOCK EXPONENT SPAN DETAILS")
print("=" * 70)

target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

block_spans = []
block_info = []

for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    bf16 = param.data.to(torch.bfloat16)
    raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    exps = ((raw >> 7) & 0xFF).astype(int)
    n = len(exps)
    pad = (128 - n % 128) % 128
    if pad:
        exps = np.concatenate([exps, np.zeros(pad, dtype=int)])
    blocks = exps.reshape(-1, 128)
    for bi, b in enumerate(blocks):
        mn, mx = int(b.min()), int(b.max())
        span = mx - mn
        block_spans.append(span)
        if span > 15:
            block_info.append((name, bi, mn, mx, span, len(np.unique(b))))

block_spans = np.array(block_spans)
print(f"Total blocks: {len(block_spans):,}")
print(f"Span: min={block_spans.min()} max={block_spans.max()} mean={block_spans.mean():.1f}")
for t in [8, 12, 15, 16, 20]:
    print(f"  span ≤ {t:>2}: {100*np.sum(block_spans<=t)/len(block_spans):.3f}%")

print(f"\nBlocks with span > 15: {len(block_info)} ({100*len(block_info)/len(block_spans):.3f}%)")
if block_info:
    # Group by tensor
    from collections import defaultdict
    by_tensor = defaultdict(list)
    for name, bi, mn, mx, span, uniq in block_info:
        tname = name.split('.')[-2]
        layer = name.split('.')[2]
        by_tensor[f"L{layer}.{tname}"].append(span)

    print(f"  Distribution by tensor type:")
    for tname in sorted(by_tensor.keys())[:20]:
        spans_list = by_tensor[tname]
        print(f"    {tname:<20}: {len(spans_list)} blocks, max_span={max(spans_list)}")

# ============================================================
# Strategy B: base+offset encode/decode + PPL test
# ============================================================
print("\n" + "=" * 70)
print("STRATEGY B: PER-BLOCK BASE+OFFSET (4-bit, adaptive)")
print("=" * 70)
print("""
Layout per block of 128 values:
  base_exp:  1 byte (shared for block)
  signs:     16 bytes (128 bits)
  offsets:   64 bytes (128 × 4-bit nibbles)  exp4 = exp8 - base_exp
  mantissa:  128 bytes (128 × 7-bit, padded to byte)
  Total: 209 bytes / 128 values  (vs 208 for KVFloat13, vs 256 for BF16)
  Compression: 18.36% — almost same as KVFloat13 but ADAPTIVE

For blocks with span > 15: clamp offset to [0,15], losing precision
on a few outlier exponents. Or use 5-bit offset (same as KVFloat13).

Alternative: skip base_exp, use same 208 bytes, but interpret
exp5 as base_exp(stored in block header reusing 1 sign byte?) + offset
""")

# Implement and test
def encode_base_offset(bf16_uint16, block_size=128):
    """Encode with per-block base + 4-bit offset. Fallback: clamp."""
    n = len(bf16_uint16)
    assert n % block_size == 0
    num_blocks = n // block_size

    sign = ((bf16_uint16 >> 15) & 1).astype(np.uint8)
    exp8 = ((bf16_uint16 >> 7) & 0xFF).astype(np.uint8)
    mant7 = (bf16_uint16 & 0x7F).astype(np.uint8)

    sign_r = sign.reshape(num_blocks, block_size)
    exp_r = exp8.reshape(num_blocks, block_size).astype(np.int16)
    mant_r = mant7.reshape(num_blocks, block_size)

    # Per-block: choose base_exp to minimize clamping error
    # Strategy: base = max(0, max_exp - 15) so that the TOP is always covered
    bases = np.zeros(num_blocks, dtype=np.uint8)
    offsets = np.zeros((num_blocks, block_size), dtype=np.uint8)

    for bi in range(num_blocks):
        block_exps = exp_r[bi]
        # Weight by magnitude: prioritize covering large exponents
        mx = int(block_exps.max())
        base = max(0, mx - 15)
        bases[bi] = base

        off = block_exps - base
        off = np.clip(off, 0, 15)  # clamp outliers
        offsets[bi] = off.astype(np.uint8)

    # Pack signs
    signs_packed = np.packbits(sign_r, axis=1, bitorder='little')

    # Pack offsets as nibbles
    even = offsets[:, 0::2]
    odd = offsets[:, 1::2]
    offsets_packed = (even | (odd << 4)).astype(np.uint8)

    return bases, signs_packed, offsets_packed, mant_r

def decode_base_offset(bases, signs_packed, offsets_packed, mant, block_size=128):
    """Decode per-block base+offset."""
    num_blocks = len(bases)

    sign = np.unpackbits(signs_packed, axis=1, bitorder='little')[:, :block_size].astype(np.uint16)

    even = (offsets_packed & 0x0F).astype(np.uint16)
    odd = ((offsets_packed >> 4) & 0x0F).astype(np.uint16)
    offsets = np.empty((num_blocks, block_size), dtype=np.uint16)
    offsets[:, 0::2] = even
    offsets[:, 1::2] = odd

    exp8 = bases[:, None].astype(np.uint16) + offsets
    exp8 = np.clip(exp8, 0, 255)

    bf16 = (sign << 15) | (exp8 << 7) | mant.astype(np.uint16)
    return bf16.reshape(-1).astype(np.uint16)

# Test on weights
print("Testing per-block base+offset on weights...")
total_match = 0
total_n = 0
saved = {}

for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    bf16 = param.data.to(torch.bfloat16)
    shape = bf16.shape
    raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    n = len(raw)
    pad = (128 - n % 128) % 128
    raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw

    bases, sp, op, mant = encode_base_offset(raw_p)
    decoded = decode_base_offset(bases, sp, op, mant)[:n]

    total_match += np.sum(raw == decoded)
    total_n += n

    # Swap weights
    dec_t = torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(param.device)
    saved[name] = param.data.clone()
    param.data = dec_t

match_pct = 100.0 * total_match / total_n
print(f"  Bit-exact: {total_match:,}/{total_n:,} ({match_pct:.6f}%)")

# PPL
test_texts = [
    "The meaning of life is a question that has puzzled philosophers for centuries.",
    "In machine learning, neural networks are trained using backpropagation algorithms.",
    "The quick brown fox jumps over the lazy dog in the warm summer afternoon.",
    "Python is a high-level programming language known for its simplicity and readability.",
    "Climate change is one of the most pressing challenges facing humanity today.",
]
losses = []
with torch.no_grad():
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())
ppl = np.exp(np.mean(losses))

inputs_gen = tokenizer("Once upon a time", return_tensors="pt").to(model.device)
with torch.no_grad():
    gen = model.generate(**inputs_gen, max_new_tokens=50, do_sample=False)
gen_text = tokenizer.decode(gen[0], skip_special_tokens=True)

print(f"  PPL: {ppl:.4f}")
print(f"  Gen: {gen_text[:150]}")

# Restore
for name, orig in saved.items():
    parts = name.split('.')
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    getattr(obj, parts[-1]).data = orig

# Baseline for comparison
orig_losses = []
with torch.no_grad():
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        orig_losses.append(outputs.loss.item())
orig_ppl = np.exp(np.mean(orig_losses))

# Storage calculation
total_blocks = total_n // 128
storage = total_blocks * (1 + 16 + 64 + 128)  # base + signs + offsets + mant
bf16_storage = total_n * 2
print(f"\n  Storage: {storage:,} bytes vs BF16 {bf16_storage:,} bytes")
print(f"  Compression: {100*(1 - storage/bf16_storage):.2f}%")
print(f"  Original PPL: {orig_ppl:.4f}")
