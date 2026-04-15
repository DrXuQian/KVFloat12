"""
Huffman compression on BF16 KV cache exponents.
Since exponent distribution is highly concentrated, Huffman could
compress more than KVFloat13's fixed 18.75%.

Test:
1. Shannon entropy → theoretical compression limit
2. Huffman coding of exponents → actual compression ratio
3. Compare: full BF16 Huffman vs exponent-only Huffman + raw mantissa
"""
import torch, numpy as np, sys, os, time, json, math, heapq
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset
from collections import Counter

model_path = "/root/autodl-tmp/Qwen3-4B"

# ============================================================
# Huffman tree
# ============================================================

class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    heap = [HuffmanNode(sym, freq) for sym, freq in freq_dict.items() if freq > 0]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, parent)
    return heap[0] if heap else None

def build_codebook(tree, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if tree.symbol is not None:
        codebook[tree.symbol] = prefix if prefix else "0"
    else:
        if tree.left:
            build_codebook(tree.left, prefix + "0", codebook)
        if tree.right:
            build_codebook(tree.right, prefix + "1", codebook)
    return codebook

def shannon_entropy(freq_dict):
    total = sum(freq_dict.values())
    entropy = 0
    for count in freq_dict.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy

def avg_code_length(codebook, freq_dict):
    total = sum(freq_dict.values())
    avg = 0
    for sym, code in codebook.items():
        if sym in freq_dict:
            avg += (freq_dict[sym] / total) * len(code)
    return avg

# ============================================================

print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

# Collect KV cache exponent distribution from calibration
print("\nCollecting KV cache exponent distribution...")
kv_exp_counter = Counter()
kv_full_counter = Counter()  # full uint16 values

cal_prompts = [
    "The quick brown fox jumps over the lazy dog in a warm summer afternoon.",
    "def quicksort(arr): return [] if not arr else quicksort([x for x in arr[1:] if x < arr[0]])",
    "In machine learning, neural networks are trained using backpropagation algorithms.",
    "The stock market experienced significant volatility as investors reacted to the latest news.",
    "Once upon a time in a distant kingdom there lived a wise old wizard who could predict the future.",
]

total_kv_values = 0
with torch.no_grad():
    for p in cal_prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        outputs = model(**inputs, use_cache=True)
        for layer in outputs.past_key_values.layers:
            for t in [layer.keys, layer.values]:
                raw = t.view(torch.int16).cpu().numpy().astype(np.uint16).flatten()
                total_kv_values += len(raw)
                exps = ((raw >> 7) & 0xFF).astype(int)
                kv_exp_counter.update(exps.tolist())
                # Sample full values (too many to count all)
                if len(kv_full_counter) < 100000:
                    kv_full_counter.update(raw[:1000].tolist())

print(f"  Total KV values: {total_kv_values:,}")
print(f"  Unique exponents: {len(kv_exp_counter)}")

# ============================================================
# Analysis
# ============================================================
print(f"\n{'='*65}")
print("HUFFMAN COMPRESSION ANALYSIS — KV Cache Exponents")
print(f"{'='*65}")

# 1) Shannon entropy of exponent distribution
exp_entropy = shannon_entropy(kv_exp_counter)
print(f"\n1) Exponent distribution entropy: {exp_entropy:.4f} bits")
print(f"   BF16 exponent: 8 bits (fixed)")
print(f"   KVFloat13:     5 bits (fixed)")
print(f"   Huffman limit:  {exp_entropy:.2f} bits (variable)")

# 2) Build Huffman tree for exponents
tree = build_huffman_tree(kv_exp_counter)
codebook = build_codebook(tree)
avg_len = avg_code_length(codebook, kv_exp_counter)

print(f"\n2) Huffman codebook ({len(codebook)} symbols):")
print(f"   Average code length: {avg_len:.4f} bits")
total_exp = sum(kv_exp_counter.values())
for exp_val, count in kv_exp_counter.most_common(15):
    code = codebook.get(exp_val, "N/A")
    pct = 100 * count / total_exp
    print(f"   exp={exp_val:>3d}: {pct:>6.2f}%  code={code:<20s} ({len(code)} bits)")
print(f"   ... ({len(codebook) - 15} more symbols)")

# 3) Compression ratio comparison
bf16_bits = 16
kvf13_bits = 13  # sign(1) + exp(5) + mant(7)

# Huffman exponent + raw mantissa + raw sign
huffman_exp_bits = avg_len  # variable-length exponent
huffman_total_bits = 1 + huffman_exp_bits + 7  # sign + huffman_exp + mantissa

# Full BF16 Huffman (on entire uint16 value)
full_entropy = shannon_entropy(kv_full_counter)

print(f"\n3) Compression ratio comparison:")
print(f"   {'Format':<30} | {'Bits/value':>10} | {'Compression':>11} | {'Parallel?':>9}")
print(f"   {'-'*65}")
print(f"   {'BF16 (baseline)':<30} | {bf16_bits:>10.2f} | {'0%':>11} | {'Yes':>9}")
print(f"   {'KVFloat13 (fixed 5-bit exp)':<30} | {kvf13_bits:>10.2f} | {100*(1-kvf13_bits/bf16_bits):>10.2f}% | {'Yes':>9}")
print(f"   {'Huffman exp + raw mant':<30} | {huffman_total_bits:>10.2f} | {100*(1-huffman_total_bits/bf16_bits):>10.2f}% | {'No':>9}")
print(f"   {'Full BF16 Huffman':<30} | {full_entropy:>10.2f} | {100*(1-full_entropy/bf16_bits):>10.2f}% | {'No':>9}")

# 4) Practical block-level compression
print(f"\n4) Practical block-level analysis (128-value blocks):")
print(f"   Testing actual compression on real KV cache data...")

# Collect actual KV cache tensors for block-level analysis
block_ratios_huff = []
block_ratios_kvf13 = []

with torch.no_grad():
    inputs = tokenizer(cal_prompts[0] + " " + cal_prompts[1], return_tensors="pt").to(model.device)
    outputs = model(**inputs, use_cache=True)
    for layer in outputs.past_key_values.layers[:5]:  # first 5 layers
        for t in [layer.keys, layer.values]:
            raw = t.view(torch.int16).cpu().numpy().astype(np.uint16).flatten()
            n = len(raw)
            pad = (128 - n % 128) % 128
            if pad:
                raw = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)])

            for bi in range(len(raw) // 128):
                block = raw[bi*128:(bi+1)*128]
                # Per-block exponent Huffman
                block_exps = ((block >> 7) & 0xFF).astype(int)
                block_exp_counter = Counter(block_exps.tolist())

                # Use GLOBAL codebook (practical scenario)
                total_exp_bits = sum(len(codebook.get(e, "0"*8)) for e in block_exps)
                total_bits_huffman = 128 + total_exp_bits + 128 * 7  # signs + huffman exps + mantissas
                total_bytes_huffman = math.ceil(total_bits_huffman / 8)

                # KVFloat13
                total_bytes_kvf13 = 208  # fixed: 16 + 64 + 128

                # BF16
                total_bytes_bf16 = 256

                block_ratios_huff.append(total_bytes_huffman / total_bytes_bf16)
                block_ratios_kvf13.append(total_bytes_kvf13 / total_bytes_bf16)

huff_avg = np.mean(block_ratios_huff)
kvf13_avg = np.mean(block_ratios_kvf13)
print(f"   Huffman avg compression: {100*(1-huff_avg):.2f}% (ratio={huff_avg:.4f})")
print(f"   KVFloat13 compression:  {100*(1-kvf13_avg):.2f}% (ratio={kvf13_avg:.4f})")
print(f"   Huffman min block ratio: {min(block_ratios_huff):.4f} ({100*(1-min(block_ratios_huff)):.1f}%)")
print(f"   Huffman max block ratio: {max(block_ratios_huff):.4f} ({100*(1-max(block_ratios_huff)):.1f}%)")

# 5) Key tradeoffs
print(f"\n{'='*65}")
print("TRADEOFF ANALYSIS")
print(f"{'='*65}")
print(f"""
                    KVFloat13           Huffman (exp only)
Compression:        18.75% (fixed)      ~{100*(1-huffman_total_bits/bf16_bits):.1f}% (variable)
Lossless:           99.9998%            100% (truly lossless)
Decode:             LUT (O(1))          Serial bitstream (O(n))
GPU-friendly:       Yes (SIMD)          No (serial dependency)
Random access:      Yes                 No (must decode from start)
Block size:         Fixed 208B          Variable
Implementation:     Simple              Complex (codebook + bitstream)

Key insight: Huffman gives ~{100*(1-huffman_total_bits/bf16_bits):.0f}% vs KVFloat13's 18.75%.
The {abs(100*(1-huffman_total_bits/bf16_bits) - 18.75):.1f}pp difference {'favors Huffman' if huffman_total_bits < kvf13_bits else 'favors KVFloat13'}.
But Huffman's serial decode makes it impractical for GPU inference.

DFloat11 (ZipServ) uses Huffman-like coding and gets ~30% compression,
but requires special hardware-aware decompression (TCA-TBE) to avoid
the serial decode bottleneck.
""")

# 6) What about sign+exponent Huffman, mantissa raw?
# The sign bit has very little entropy (roughly 50/50), so Huffman on
# combined sign+exponent would be ~1 + entropy(exp) bits
sign_exp_counter = Counter()
with torch.no_grad():
    inputs = tokenizer(cal_prompts[0], return_tensors="pt").to(model.device)
    outputs = model(**inputs, use_cache=True)
    for layer in outputs.past_key_values.layers:
        for t in [layer.keys, layer.values]:
            raw = t.view(torch.int16).cpu().numpy().astype(np.uint16).flatten()
            sign_exp = ((raw >> 7) & 0x1FF).astype(int)  # 9 bits: sign + exp
            sign_exp_counter.update(sign_exp.tolist())

se_entropy = shannon_entropy(sign_exp_counter)
se_total = 1 + se_entropy + 7  # not exactly, sign is included in se_entropy
print(f"Sign+Exponent (9-bit) entropy: {se_entropy:.4f} bits")
print(f"Total with raw mantissa: {se_entropy + 7:.2f} bits ({100*(1-(se_entropy+7)/16):.2f}% compression)")

with open("/root/kvfloat13/huffman_analysis.json", "w") as f:
    json.dump({
        "model": "Qwen3-4B",
        "exp_entropy": exp_entropy,
        "huffman_avg_code_len": avg_len,
        "huffman_total_bits": huffman_total_bits,
        "kvfloat13_bits": kvf13_bits,
        "bf16_bits": bf16_bits,
        "huffman_compression": 100*(1-huffman_total_bits/bf16_bits),
        "kvfloat13_compression": 100*(1-kvf13_bits/bf16_bits),
        "full_bf16_entropy": full_entropy,
        "sign_exp_entropy": se_entropy,
    }, f, indent=2)
print("\nSaved to huffman_analysis.json")
