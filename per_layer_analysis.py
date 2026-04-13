"""
Per-layer exponent analysis: are the same exponents used everywhere,
or does each layer have its own range?
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

def extract_exp8(t):
    return ((t.view(torch.int16).to(torch.int32) >> 7) & 0xFF).cpu().numpy().flatten()

model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")

target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

# ============================================================
# Part 1: Per-layer exponent RANGE (min exp, max exp, span)
# ============================================================
print("=" * 90)
print("PER-LAYER EXPONENT RANGE (weights)")
print("=" * 90)
print(f"{'Layer':>5} | {'Tensor':<12} | {'Min exp':>7} | {'Max exp':>7} | {'Span':>4} | {'Unique':>6} | Top exponents (by freq)")
print("-" * 90)

layer_ranges = {}  # layer_num -> (min_exp, max_exp)
layer_counters = {}  # layer_num -> Counter

for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue

    parts = name.split('.')
    layer_num = None
    for i, p in enumerate(parts):
        if p == 'layers' and i+1 < len(parts):
            layer_num = int(parts[i+1])
    tensor_name = name.rsplit('.', 1)[0].rsplit('.', 1)[-1]

    exps = extract_exp8(param.data.to(torch.bfloat16))
    counter = Counter(exps.tolist())

    # Exclude zero-count
    used_exps = sorted(counter.keys())
    min_e, max_e = used_exps[0], used_exps[-1]
    span = max_e - min_e + 1

    # Top 5 exponents
    top5 = counter.most_common(5)
    top5_str = " ".join([f"{e}({100*c/len(exps):.1f}%)" for e, c in top5])

    print(f"{layer_num:>5} | {tensor_name:<12} | {min_e:>7} | {max_e:>7} | {span:>4} | {len(counter):>6} | {top5_str}")

    if layer_num not in layer_counters:
        layer_counters[layer_num] = Counter()
    layer_counters[layer_num].update(counter)

# ============================================================
# Part 2: Per-layer summary — range and overlap
# ============================================================
print("\n" + "=" * 90)
print("PER-LAYER SUMMARY (all tensors combined)")
print("=" * 90)
print(f"{'Layer':>5} | {'Unique':>6} | {'Min':>4} | {'Max':>4} | {'Span':>4} | {'Center':>6} | Used exponents")
print("-" * 90)

all_layer_exps = {}
for layer_num in sorted(layer_counters.keys()):
    c = layer_counters[layer_num]
    used = sorted(c.keys())
    min_e, max_e = used[0], used[-1]

    # Weighted center
    total = sum(c.values())
    center = sum(e * cnt for e, cnt in c.items()) / total

    all_layer_exps[layer_num] = set(used)
    print(f"{layer_num:>5} | {len(used):>6} | {min_e:>4} | {max_e:>4} | {max_e-min_e+1:>4} | {center:>6.1f} | {used}")

# ============================================================
# Part 3: KV cache per-layer
# ============================================================
print("\n" + "=" * 90)
print("PER-LAYER KV CACHE EXPONENT RANGE")
print("=" * 90)

prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "In machine learning, neural networks are trained using backpropagation.",
    "The stock market experienced significant volatility as investors reacted to news.",
    "Once upon a time in a distant kingdom there lived a wise old wizard.",
]

kv_layer_counters = {}  # (layer, 'key'/'value') -> Counter
model.eval()
with torch.no_grad():
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs, use_cache=True)
        for layer_idx, kv in enumerate(outputs.past_key_values):
            for kv_name, tensor in [("key", kv[0]), ("value", kv[1])]:
                exps = extract_exp8(tensor.to(torch.bfloat16))
                key = (layer_idx, kv_name)
                if key not in kv_layer_counters:
                    kv_layer_counters[key] = Counter()
                kv_layer_counters[key].update(exps.tolist())

print(f"{'Layer':>5} | {'KV':<6} | {'Unique':>6} | {'Min':>4} | {'Max':>4} | {'Span':>4} | {'Center':>6} | Used exponents")
print("-" * 90)

for (layer_idx, kv_name) in sorted(kv_layer_counters.keys()):
    c = kv_layer_counters[(layer_idx, kv_name)]
    used = sorted(c.keys())
    min_e, max_e = used[0], used[-1]
    total = sum(c.values())
    center = sum(e * cnt for e, cnt in c.items()) / total
    print(f"{layer_idx:>5} | {kv_name:<6} | {len(used):>6} | {min_e:>4} | {max_e:>4} | {max_e-min_e+1:>4} | {center:>6.1f} | {used}")

# ============================================================
# Part 4: Key insight — can we use base_exp + offset?
# ============================================================
print("\n" + "=" * 90)
print("KEY INSIGHT: PER-LAYER BASE + OFFSET APPROACH")
print("=" * 90)
print("""
If every layer's exponents fit in a contiguous range of ≤16,
we can store: base_exp (8-bit, per block/layer) + 4-bit offset
→ exp8 = base_exp + offset

This eliminates the LUT entirely!
""")

# Check contiguity
print("Weight layers — contiguous range check:")
for layer_num in sorted(layer_counters.keys()):
    used = sorted(layer_counters[layer_num].keys())
    span = used[-1] - used[0] + 1
    is_contiguous = (len(used) == span)
    fits_4bit = span <= 16
    fits_3bit = span <= 8
    print(f"  Layer {layer_num:>2}: range [{used[0]}-{used[-1]}] span={span:>2} unique={len(used):>2} "
          f"contiguous={'✓' if is_contiguous else '✗'} fits_4bit={'✓' if fits_4bit else '✗'} "
          f"fits_3bit={'✓' if fits_3bit else '✗'}")

print("\nKV cache layers — contiguous range check:")
for (layer_idx, kv_name) in sorted(kv_layer_counters.keys()):
    c = kv_layer_counters[(layer_idx, kv_name)]
    used = sorted(c.keys())
    # Check if 0 (zero/denorm) is an outlier
    used_nonzero = [e for e in used if e != 0]
    if used_nonzero:
        span = used_nonzero[-1] - used_nonzero[0] + 1
    else:
        span = 0
    has_zero = 0 in used
    fits_4bit = span <= 16
    fits_3bit = span <= 8
    if kv_name == 'key':  # only print key to save space, or every 5th layer
        if layer_idx % 5 == 0:
            zero_note = " (has exp=0)" if has_zero else ""
            print(f"  Layer {layer_idx:>2} {kv_name}: range [{used_nonzero[0] if used_nonzero else 'N/A'}-{used_nonzero[-1] if used_nonzero else 'N/A'}] "
                  f"span={span:>2} unique={len(used):>2} fits_4bit={'✓' if fits_4bit else '✗'}{zero_note}")

# ============================================================
# Part 5: Global union analysis
# ============================================================
print("\n" + "=" * 90)
print("GLOBAL UNION: ALL LAYERS COMBINED")
print("=" * 90)

all_weight_exps = set()
for c in layer_counters.values():
    all_weight_exps.update(c.keys())

all_kv_exps = set()
for c in kv_layer_counters.values():
    all_kv_exps.update(c.keys())

all_exps = all_weight_exps | all_kv_exps
all_nonzero = sorted(e for e in all_exps if e != 0)

print(f"Weight exponents (union):  {sorted(all_weight_exps)}")
print(f"KV cache exponents (union):{sorted(all_kv_exps)}")
print(f"Combined (non-zero): [{all_nonzero[0]}-{all_nonzero[-1]}] span={all_nonzero[-1]-all_nonzero[0]+1}")
print(f"Combined total unique: {len(all_exps)} (including exp=0: {'yes' if 0 in all_exps else 'no'})")

if all_nonzero[-1] - all_nonzero[0] + 1 <= 32:
    print(f"\n→ All non-zero exponents fit in a CONTIGUOUS range of {all_nonzero[-1]-all_nonzero[0]+1}")
    print(f"  This means: exp5 = exp8 - {all_nonzero[0]}  (simple subtraction, NO LUT needed!)")
    print(f"  Plus one special code for exp=0 (zero/denorm)")
