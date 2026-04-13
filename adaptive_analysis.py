"""
1) KV cache exponent distribution stability across different inputs
2) Per-block exponent range analysis — can we do base+offset per block?
"""
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter, defaultdict

def extract_exp8(t):
    return ((t.view(torch.int16).to(torch.int32) >> 7) & 0xFF).cpu().numpy().flatten()

model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
model.eval()

# ============================================================
# Part 1: KV cache stability across different inputs
# ============================================================
print("=" * 70)
print("PART 1: KV CACHE EXPONENT DISTRIBUTION vs INPUT")
print("=" * 70)

prompt_groups = {
    "english": [
        "The quick brown fox jumps over the lazy dog.",
        "She sold seashells by the seashore on a warm summer day.",
    ],
    "code": [
        "def quicksort(arr): return [] if not arr else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]]",
        "for i in range(100): print(sum(range(i)))",
    ],
    "math": [
        "The integral of x^2 from 0 to 1 equals 1/3. Consider the Taylor expansion",
        "If f(x) = e^x, then f'(x) = e^x. The Fourier transform of a Gaussian is",
    ],
    "random_tokens": [
        "xkcd 1729 🎭 §§§ αβγδ ∫∫∫ ████ 0xDEADBEEF",
        "!!!??? ... === +++ --- *** @@@ ### $$$ %%% ^^^",
    ],
    "long_context": [
        "In the beginning, there was nothing but void and darkness. Then came light, and with it, the first stars formed in the cosmic dawn. Billions of years passed as galaxies collided and merged, creating the vast structures we observe today. On a small rocky planet orbiting an unremarkable yellow star, something remarkable happened: life emerged from the primordial soup.",
    ],
}

# Per-input, per-layer exponent stats
layer_input_stats = defaultdict(list)  # (layer, kv) -> list of (input_name, min_exp, max_exp, unique, top3)

with torch.no_grad():
    for group_name, prompts in prompt_groups.items():
        for pi, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            seq_len = inputs["input_ids"].shape[1]
            outputs = model(**inputs, use_cache=True)

            for layer_idx, kv in enumerate(outputs.past_key_values):
                for kv_name, tensor in [("key", kv[0]), ("value", kv[1])]:
                    exps = extract_exp8(tensor.to(torch.bfloat16))
                    counter = Counter(exps.tolist())
                    used = sorted(counter.keys())
                    nonzero = [e for e in used if e != 0]

                    if nonzero:
                        mn, mx = nonzero[0], nonzero[-1]
                    else:
                        mn, mx = 0, 0

                    top3 = [e for e, _ in counter.most_common(3)]
                    layer_input_stats[(layer_idx, kv_name)].append({
                        'input': f"{group_name}_{pi}",
                        'seq_len': seq_len,
                        'min': mn, 'max': mx,
                        'span': mx - mn + 1 if nonzero else 0,
                        'unique': len(used),
                        'has_zero': 0 in used,
                        'top3': top3,
                    })

# Show variation per layer
print(f"\n{'Layer':>5} | {'KV':<5} | {'Min range':>10} | {'Max range':>10} | {'Span range':>11} | Top-3 exps across inputs")
print("-" * 85)

for layer_idx in [0, 5, 10, 15, 20, 25, 29]:
    for kv_name in ['key', 'value']:
        stats = layer_input_stats[(layer_idx, kv_name)]
        mins = [s['min'] for s in stats]
        maxs = [s['max'] for s in stats]
        spans = [s['span'] for s in stats]
        all_top3 = set()
        for s in stats:
            all_top3.update(s['top3'])

        print(f"{layer_idx:>5} | {kv_name:<5} | [{min(mins)}-{max(mins)}] | [{min(maxs)}-{max(maxs)}] | "
              f"[{min(spans):>2}-{max(spans):>2}]     | {sorted(all_top3)}")

# Detailed: show each input for layer 0 key
print(f"\nDetailed: Layer 0, Key")
for s in layer_input_stats[(0, 'key')]:
    print(f"  {s['input']:<20} seq={s['seq_len']:>3} range=[{s['min']}-{s['max']}] span={s['span']:>2} "
          f"unique={s['unique']:>2} zero={'Y' if s['has_zero'] else 'N'} top3={s['top3']}")

print(f"\nDetailed: Layer 15, Value")
for s in layer_input_stats[(15, 'value')]:
    print(f"  {s['input']:<20} seq={s['seq_len']:>3} range=[{s['min']}-{s['max']}] span={s['span']:>2} "
          f"unique={s['unique']:>2} zero={'Y' if s['has_zero'] else 'N'} top3={s['top3']}")

# ============================================================
# Part 2: Per-BLOCK exponent range (128 values per block)
# ============================================================
print("\n" + "=" * 70)
print("PART 2: PER-BLOCK EXPONENT RANGE (block=128 values)")
print("=" * 70)
print("If per-block span ≤ 16, we can use: base_exp(8-bit) + offset(4-bit)")
print()

# Weights
target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

block_spans_w = []
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
    for b in blocks:
        mn, mx = b.min(), b.max()
        block_spans_w.append(mx - mn)

block_spans_w = np.array(block_spans_w)
print("WEIGHTS per-block span:")
print(f"  Total blocks: {len(block_spans_w)}")
print(f"  Span: min={block_spans_w.min()} max={block_spans_w.max()} "
      f"mean={block_spans_w.mean():.1f} median={np.median(block_spans_w):.0f}")
for threshold in [8, 16, 24, 32]:
    pct = 100.0 * np.sum(block_spans_w <= threshold) / len(block_spans_w)
    print(f"  span ≤ {threshold:>2}: {pct:.2f}%")

# KV cache
block_spans_kv = []
with torch.no_grad():
    for group_name, prompts in prompt_groups.items():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model(**inputs, use_cache=True)
            for kv in outputs.past_key_values:
                for tensor in [kv[0], kv[1]]:
                    bf16 = tensor.to(torch.bfloat16)
                    raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
                    exps = ((raw >> 7) & 0xFF).astype(int)
                    n = len(exps)
                    if n < 128:
                        exps = np.concatenate([exps, np.zeros(128 - n, dtype=int)])
                    blocks = exps.reshape(-1, 128)
                    for b in blocks:
                        block_spans_kv.append(b.max() - b.min())

block_spans_kv = np.array(block_spans_kv)
print(f"\nKV CACHE per-block span:")
print(f"  Total blocks: {len(block_spans_kv)}")
print(f"  Span: min={block_spans_kv.min()} max={block_spans_kv.max()} "
      f"mean={block_spans_kv.mean():.1f} median={np.median(block_spans_kv):.0f}")
for threshold in [8, 16, 24, 32]:
    pct = 100.0 * np.sum(block_spans_kv <= threshold) / len(block_spans_kv)
    print(f"  span ≤ {threshold:>2}: {pct:.2f}%")

# ============================================================
# Part 3: Per-HEAD exponent range (for KV cache, natural boundary)
# ============================================================
print("\n" + "=" * 70)
print("PART 3: PER-HEAD EXPONENT RANGE (head_dim=64, natural KV boundary)")
print("=" * 70)

head_spans = []
head_details = []
with torch.no_grad():
    for prompt in ["The quick brown fox", "def fibonacci(n):", "∫∫ αβγ 0xDEAD"]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs, use_cache=True)
        seq_len = inputs["input_ids"].shape[1]

        for layer_idx, kv in enumerate(outputs.past_key_values):
            for kv_name, tensor in [("key", kv[0]), ("value", kv[1])]:
                # tensor: [batch, heads, seq, head_dim]
                t = tensor.to(torch.bfloat16)
                b, h, s, d = t.shape
                for hi in range(h):
                    for si in range(s):
                        head_vec = t[0, hi, si, :]  # one head, one position
                        exps = extract_exp8(head_vec)
                        nonzero = exps[exps != 0]
                        if len(nonzero) > 0:
                            span = nonzero.max() - nonzero.min()
                        else:
                            span = 0
                        head_spans.append(span)
                        if layer_idx in [0, 15, 29] and hi == 0 and si == 0:
                            head_details.append((layer_idx, kv_name, hi, si, span,
                                                 int(nonzero.min()) if len(nonzero) > 0 else 0,
                                                 int(nonzero.max()) if len(nonzero) > 0 else 0))

head_spans = np.array(head_spans)
print(f"Per-head-per-position span (head_dim={d}):")
print(f"  Total vectors: {len(head_spans)}")
print(f"  Span: min={head_spans.min()} max={head_spans.max()} "
      f"mean={head_spans.mean():.1f} median={np.median(head_spans):.0f}")
for threshold in [4, 8, 12, 16]:
    pct = 100.0 * np.sum(head_spans <= threshold) / len(head_spans)
    print(f"  span ≤ {threshold:>2}: {pct:.2f}%")

# ============================================================
# Part 4: Proposed adaptive schemes
# ============================================================
print("\n" + "=" * 70)
print("PART 4: ALTERNATIVE MAPPING STRATEGIES")
print("=" * 70)
print("""
Strategy A: Fixed global LUT (current KVFloat13)
  - 32-entry LUT, set once at model load
  - Pro: simple, no per-block overhead
  - Con: can't adapt, needs 5 bits

Strategy B: Per-block base+offset (NO LUT)
  - Store base_exp (1 byte) per block of 128
  - Each value stores 4-bit offset: exp8 = base_exp + offset
  - Overhead: +1 byte per 128 values (0.8% extra)
  - Pro: adapts to each block, no global LUT
  - Con: only works if per-block span ≤ 15

Strategy C: Per-block base+offset with wider offset
  - base_exp (1 byte) + 5-bit offset → span ≤ 31
  - Same as current KVFloat13 but adaptive base
  - Same compression ratio, but handles shifting distributions

Strategy D: Two-level: coarse global + fine per-block
  - Global LUT maps exp8 → exp5 (coarse, like KVFloat13)
  - Per-block correction for outliers (1-2 bytes overhead)

Strategy E: Per-block min/max with uniform quantization
  - Store min_exp, max_exp per block (2 bytes)
  - Quantize each exp into the [min, max] range with N bits
  - Like block floating point (BFP)
""")

# Check feasibility of Strategy B
print("Strategy B feasibility check:")
print(f"  Weights: {100*np.sum(block_spans_w <= 15)/len(block_spans_w):.2f}% blocks fit in 4-bit offset (span ≤ 15)")
print(f"  KV cache: {100*np.sum(block_spans_kv <= 15)/len(block_spans_kv):.2f}% blocks fit")
print(f"  Weight blocks with span > 15: {np.sum(block_spans_w > 15)} / {len(block_spans_w)}")

# Show the outlier blocks
outlier_mask = block_spans_w > 15
if np.any(outlier_mask):
    outlier_spans = block_spans_w[outlier_mask]
    print(f"  Outlier spans: {sorted(np.unique(outlier_spans).tolist())}")
    print(f"  Span distribution of outliers: mean={outlier_spans.mean():.1f} max={outlier_spans.max()}")
