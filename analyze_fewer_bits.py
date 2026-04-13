"""
Analyze: can we use fewer exponent bits than 5?
Check coverage at 4-bit (16 entries), 3-bit (8 entries), 2-bit (4 entries)
"""
import json
import numpy as np

with open('/root/kvfloat13/exponent_frequencies.json') as f:
    freq = {int(k): int(v) for k, v in json.load(f).items()}

total = sum(freq.values())
sorted_exps = sorted(freq.items(), key=lambda x: -x[1])

print("=" * 65)
print("EXPONENT BIT-WIDTH ANALYSIS")
print("=" * 65)

for bits in [5, 4, 3, 2, 1]:
    k = 2 ** bits
    top_k = sorted_exps[:k]
    covered = sum(c for _, c in top_k)
    coverage = 100.0 * covered / total
    missed = total - covered

    # Format size
    total_bits = 1 + bits + 7  # sign + exp + mantissa
    compression = 100.0 * (1 - total_bits / 16)

    # Storage layout analysis
    print(f"\n--- {bits}-bit exponent ({k} entries) ---")
    print(f"  Logical format: sign(1) + exp({bits}) + mant(7) = {total_bits} bits")
    print(f"  Compression vs BF16: {compression:.2f}%")
    print(f"  Coverage: {coverage:.6f}% ({covered:,}/{total:,})")
    print(f"  Missed values: {missed:,} ({100*missed/total:.6f}%)")

    # Show what's covered vs missed
    covered_exps = sorted([e for e, _ in top_k])
    print(f"  Covered exponents: {covered_exps}")

    if missed > 0:
        missed_exps = [(e, c) for e, c in sorted_exps[k:]]
        print(f"  Missed exponents: ", end="")
        for e, c in missed_exps[:10]:
            print(f"exp={e}({c})", end=" ")
        print()

# Deeper analysis: what about separate weight vs KV cache?
# Reload per-source data
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

print("\n" + "=" * 65)
print("PER-SOURCE COVERAGE AT 4-BIT (16 entries)")
print("=" * 65)

model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")

def extract_exp8(t):
    return ((t.view(torch.int16).to(torch.int32) >> 7) & 0xFF).cpu().numpy().flatten()

# Weight exponents
w_counter = Counter()
for name, param in model.named_parameters():
    if any(name.endswith(s) for s in ['q_proj.weight','k_proj.weight','v_proj.weight',
            'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']):
        w_counter.update(extract_exp8(param.data.to(torch.bfloat16)).tolist())

# KV cache exponents
kv_counter = Counter()
prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "In machine learning, neural networks are trained using backpropagation.",
]
model.eval()
with torch.no_grad():
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs, use_cache=True)
        for kv in outputs.past_key_values:
            kv_counter.update(extract_exp8(kv[0].to(torch.bfloat16)).tolist())
            kv_counter.update(extract_exp8(kv[1].to(torch.bfloat16)).tolist())

w_total = sum(w_counter.values())
kv_total = sum(kv_counter.values())

for bits in [4, 3]:
    k = 2 ** bits
    print(f"\n--- {bits}-bit exponent ({k} entries) ---")

    # Weight-only top-k
    w_topk = [e for e, _ in w_counter.most_common(k)]
    w_cov = 100.0 * sum(w_counter[e] for e in w_topk) / w_total

    # KV-only top-k
    kv_topk = [e for e, _ in kv_counter.most_common(k)]
    kv_cov = 100.0 * sum(kv_counter[e] for e in kv_topk) / kv_total

    # Combined top-k
    combined = w_counter + kv_counter
    c_topk = [e for e, _ in combined.most_common(k)]
    w_with_global = 100.0 * sum(w_counter.get(e, 0) for e in c_topk) / w_total
    kv_with_global = 100.0 * sum(kv_counter.get(e, 0) for e in c_topk) / kv_total

    print(f"  Weight-specific LUT:  {w_cov:.4f}%  exps={sorted(w_topk)}")
    print(f"  KV-specific LUT:      {kv_cov:.4f}%  exps={sorted(kv_topk)}")
    print(f"  Global LUT → weights: {w_with_global:.4f}%")
    print(f"  Global LUT → KV:      {kv_with_global:.4f}%")
    print(f"  Global top-{k}: {sorted(c_topk)}")

# Proposed new format: KVFloat12 with 4-bit exponent
print("\n" + "=" * 65)
print("PROPOSED STORAGE LAYOUTS")
print("=" * 65)

layouts = [
    ("KVFloat13 (current)", 5, "sign(1)+exp_hi(4) | exp_lo(1)+mant(7)",
     "16 + 64 + 128 = 208 bytes/128vals", 18.75),
    ("KVFloat12", 4, "sign(1)+exp(4)+mant(7) = 12 bits",
     "16 + 64 + 112 = 192 bytes/128vals (needs redesign)", 25.0),
    ("KVFloat11", 3, "sign(1)+exp(3)+mant(7) = 11 bits",
     "TBD - harder to byte-align", 31.25),
]

for name, bits, fmt, storage, comp in layouts:
    print(f"\n{name}: {1+bits+7} bits/value, {comp:.2f}% compression")
    print(f"  Format: {fmt}")
    print(f"  Storage: {storage}")

# KVFloat12 concrete layout
print("\n" + "=" * 65)
print("KVFloat12: 4-bit exponent layout (byte-aligned)")
print("=" * 65)
print("""
Option A: 1+4+8 split (same structure, fewer exp bits)
  Stream 1: sign(1) bits      → 16 bytes / 128 vals
  Stream 2: exp(4) nibbles     → 64 bytes / 128 vals
  Stream 3: mantissa(7) bytes  → 128 bytes / 128 vals (pad 1 bit)
  Total: 208 bytes — NO SAVINGS (mantissa byte wastes 1 bit)

Option B: 12-bit packed (3 values per 4.5 bytes)
  Pack 2 values into 3 bytes: [exp4|mant7|sign1] [exp4|mant7|sign1] [0000|0000]
  Awkward, not byte-aligned for individual access.

Option C: sign(1)+exp(4) nibble + mant(7) byte
  Stream 1: [sign(1)|exp(4)] packed as 5-bit → 80 bytes/128 vals (5-bit packing, messy)
  Stream 2: mantissa(7) → 128 bytes
  Total: 208 bytes — still no real savings

Option D: Combine sign+exp into 5-bit field, keep mant as 7-bit byte
  [sign(1)|exp(4)] = 5 bits → need 5-bit packing or waste bits

  Better: [sign(1)|exp(4)|mant_hi(3)] byte + mant_lo(4) nibble
  Stream 1: 128 bytes (8 bits each)
  Stream 2: 64 bytes (4-bit nibbles)
  Total: 192 bytes / 128 vals = 1.5 bytes/val = 12 bits
  Compression: 25% vs BF16

  Decode:
    se_byte = stream1[i]
    sign = se_byte >> 7
    exp4 = (se_byte >> 3) & 0xF
    mant_hi3 = se_byte & 0x7
    mant_lo4 = extract_nibble(stream2, i)
    mant7 = (mant_hi3 << 4) | mant_lo4
    exp8 = lut[exp4]
    bf16 = (sign<<15) | (exp8<<7) | mant7
""")

print("\n" + "=" * 65)
print("SUMMARY: FORMAT COMPARISON")
print("=" * 65)
print(f"{'Format':<16} {'Bits':>4} {'Compress':>9} {'Coverage':>10} {'Byte-aligned':>13}")
print("-" * 55)
print(f"{'BF16':<16} {'16':>4} {'0%':>9} {'100%':>10} {'Yes':>13}")
print(f"{'KVFloat13':<16} {'13':>4} {'18.75%':>9} {'100.000%':>10} {'Yes':>13}")
print(f"{'KVFloat12':<16} {'12':>4} {'25.00%':>9} {'~99.999%':>10} {'Yes (Opt D)':>13}")
print(f"{'KVFloat11':<16} {'11':>4} {'31.25%':>9} {'~99.9%':>10} {'Tricky':>13}")
