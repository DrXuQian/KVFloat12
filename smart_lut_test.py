"""
Test non-contiguous LUT strategies for 4-bit (16 entries).
Can we beat [115-130] by picking smarter 16 exponents?
"""
import torch, numpy as np, sys
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from collections import Counter
sys.path.insert(0, '/root/kvfloat13')
from split_lut_kvfloat12 import encode_kvf12, decode_kvf12, extract_exp8

model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
model.eval()

target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

# Collect (exp, count, sum_abs_val) for weights
w_exp_stats = {}  # exp -> (count, sum_abs_val)
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    bf16 = param.data.to(torch.bfloat16)
    raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    exps = ((raw >> 7) & 0xFF).astype(int)
    vals = np.abs(bf16.contiguous().view(-1).float().cpu().numpy())
    for e in np.unique(exps):
        mask = exps == e
        if e not in w_exp_stats:
            w_exp_stats[e] = [0, 0.0]
        w_exp_stats[e][0] += int(mask.sum())
        w_exp_stats[e][1] += float(vals[mask].sum())

# Also collect KV cache stats
kv_exp_stats = {}
prompts_cal = ["The quick brown fox jumps over the lazy dog.",
               "def fibonacci(n): return n",
               "In machine learning, neural networks are trained."]
with torch.no_grad():
    for prompt in prompts_cal:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs, use_cache=True)
        for layer in outputs.past_key_values.layers:
            for tensor in [layer.keys, layer.values]:
                bf16 = tensor.to(torch.bfloat16)
                raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
                exps = ((raw >> 7) & 0xFF).astype(int)
                vals = np.abs(bf16.contiguous().view(-1).float().cpu().numpy())
                for e in np.unique(exps):
                    mask = exps == e
                    if e not in kv_exp_stats:
                        kv_exp_stats[e] = [0, 0.0]
                    kv_exp_stats[e][0] += int(mask.sum())
                    kv_exp_stats[e][1] += float(vals[mask].sum())

# Combined
all_exp_stats = {}
for e in set(list(w_exp_stats.keys()) + list(kv_exp_stats.keys())):
    wc, wv = w_exp_stats.get(e, [0, 0.0])
    kc, kv_val = kv_exp_stats.get(e, [0, 0.0])
    all_exp_stats[e] = (wc + kc, wv + kv_val)

def build_lut_from_set(exp_set):
    """Build compress/decompress LUTs from arbitrary set of exponents."""
    exps = sorted(exp_set)
    decompress = np.array(exps, dtype=np.uint8)
    compress = np.zeros(256, dtype=np.uint8)
    exp_to_idx = {e: i for i, e in enumerate(exps)}
    for e, i in exp_to_idx.items():
        compress[e] = i
    for e in range(256):
        if e in exp_to_idx:
            continue
        best_i, best_d = 0, float('inf')
        for se in exps:
            d = abs(2.0**(e-127) - 2.0**(se-127))
            if d < best_d:
                best_d, best_i = d, exp_to_idx[se]
        compress[e] = best_i
    return compress, decompress

def compute_total_error(exp_set, stats):
    """Compute total absolute reconstruction error for a given LUT."""
    supported = set(exp_set)
    sorted_exps = sorted(exp_set)
    total_err = 0.0
    for e, (count, sum_val) in stats.items():
        if e in supported:
            continue  # no error
        # Find nearest supported
        best_se = min(sorted_exps, key=lambda se: abs(2.0**(e-127) - 2.0**(se-127)))
        # Error per value: |2^e - 2^nearest| * (avg_mantissa_factor)
        # Approximate: each value has magnitude ~2^(e-127) * 1.5 (average mantissa)
        # After mapping: magnitude ~2^(best_se-127) * 1.5
        # Error = |1 - 2^(best_se-e)| * original_magnitude
        ratio = 2.0**(best_se - e)
        rel_err = abs(1.0 - ratio)
        # Total error contribution = rel_err * sum_abs_val
        total_err += rel_err * sum_val
    return total_err

# ============================================================
# Strategy 1: Contiguous [115-130] (baseline)
# ============================================================
s1 = list(range(115, 131))
err1 = compute_total_error(s1, all_exp_stats)

# ============================================================
# Strategy 2: Top-16 by sum|val| (value-weighted frequency)
# ============================================================
sorted_by_val = sorted(all_exp_stats.items(), key=lambda x: -x[1][1])
s2 = [e for e, _ in sorted_by_val[:16]]
err2 = compute_total_error(s2, all_exp_stats)

# ============================================================
# Strategy 3: Greedy minimize total error
# ============================================================
# Start with the most valuable exponent, greedily add the one that reduces error most
candidates = sorted(all_exp_stats.keys())
s3 = []
remaining = set(candidates)
for _ in range(16):
    best_e, best_err = None, float('inf')
    for e in remaining:
        trial = s3 + [e]
        err = compute_total_error(trial, all_exp_stats)
        if err < best_err:
            best_err, best_e = err, e
    s3.append(best_e)
    remaining.discard(best_e)
err3 = compute_total_error(s3, all_exp_stats)

# ============================================================
# Strategy 4: Cover both tails — [112-127] + swap lowest for 128,129,130
# ============================================================
s4 = list(range(115, 131))  # same as s1 for now
# Try: drop least valuable from bottom, add from top
# E.g., [113, 114, ..., 127, 128, 129, 130] — skip some low ones
s4a = [113, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
err4a = compute_total_error(s4a, all_exp_stats)

s4b = [114, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130]
err4b = compute_total_error(s4b, all_exp_stats)

s4c = [113, 114, 116, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130]
err4c = compute_total_error(s4c, all_exp_stats)

# ============================================================
# Strategy 5: Brute-force best 16 from [110-131] (C(22,16) = 74613)
# ============================================================
print("Brute-force searching best 16 from [110-131]...")
candidate_range = list(range(110, 132))
best_bf_set, best_bf_err = None, float('inf')
count = 0
for combo in combinations(candidate_range, 16):
    err = compute_total_error(list(combo), all_exp_stats)
    if err < best_bf_err:
        best_bf_err, best_bf_set = err, list(combo)
    count += 1
    if count % 10000 == 0:
        print(f"  checked {count}...")
s5 = best_bf_set
err5 = best_bf_err

print(f"\n{'Strategy':<50} | {'Error':>12} | Exponents")
print("-" * 100)
print(f"{'1. Contiguous [115-130]':<50} | {err1:>12.2f} | {sorted(s1)}")
print(f"{'2. Top-16 by sum|val|':<50} | {err2:>12.2f} | {sorted(s2)}")
print(f"{'3. Greedy min error':<50} | {err3:>12.2f} | {sorted(s3)}")
print(f"{'4a. [113,115-129]':<50} | {err4a:>12.2f} | {sorted(s4a)}")
print(f"{'4b. [114,116-130]':<50} | {err4b:>12.2f} | {sorted(s4b)}")
print(f"{'4c. [113,114,116,118-130]':<50} | {err4c:>12.2f} | {sorted(s4c)}")
print(f"{'5. Brute-force best from [110-131]':<50} | {err5:>12.2f} | {sorted(s5)}")

# ============================================================
# PPL test for top strategies
# ============================================================
def compress_tensor(tensor, cl, dl):
    shape, device = tensor.shape, tensor.device
    raw = tensor.to(torch.bfloat16).contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    n = len(raw)
    pad = (128 - n % 128) % 128
    raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw
    s1_, s2_ = encode_kvf12(raw_p, cl)
    decoded = decode_kvf12(s1_, s2_, dl)[:n]
    return torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(device)

def compress_cache(cache, cl, dl):
    for layer in cache.layers:
        layer.keys = compress_tensor(layer.keys, cl, dl)
        layer.values = compress_tensor(layer.values, cl, dl)
    return cache

test_texts = [
    "The meaning of life is a question that has puzzled philosophers for centuries.",
    "In machine learning, neural networks are trained using backpropagation algorithms.",
    "The quick brown fox jumps over the lazy dog in the warm summer afternoon.",
    "Python is a high-level programming language known for its simplicity and readability.",
    "Climate change is one of the most pressing challenges facing humanity today.",
]

# Baseline
orig_losses = []
with torch.no_grad():
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        orig_losses.append(outputs.loss.item())
orig_ppl = np.exp(np.mean(orig_losses))

def test_full_ppl(exp_set, label):
    cl, dl = build_lut_from_set(exp_set)

    # Compress weights
    saved = {}
    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        saved[name] = param.data.clone()
        param.data = compress_tensor(param.data, cl, dl)

    # PPL with compressed weights + KV cache
    losses = []
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            total_loss = 0.0
            cache = DynamicCache()
            for t in range(seq_len - 1):
                tok = input_ids[:, t:t+1]
                out = model(tok, past_key_values=cache, use_cache=True)
                cache = compress_cache(out.past_key_values, cl, dl)
                logits = out.logits[:, -1, :]
                target = input_ids[:, t+1]
                loss = torch.nn.functional.cross_entropy(logits, target)
                total_loss += loss.item()
            losses.append(total_loss / (seq_len - 1))
    ppl = np.exp(np.mean(losses))

    # Restore
    for name, orig in saved.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        getattr(obj, parts[-1]).data = orig

    print(f"  [{label}] PPL={ppl:.4f} (Δ{ppl-orig_ppl:+.4f}) exps={sorted(exp_set)}")
    return ppl

print(f"\nBaseline PPL: {orig_ppl:.4f}")
print("\nFull PPL (weights + KV cache compressed):")
strategies = [
    (s1, "contiguous [115-130]"),
    (s2, "top-16 by val"),
    (s3, "greedy min error"),
    (s5, "brute-force best"),
]
# Add 4a/4b/4c if they're different enough
if sorted(s4a) != sorted(s1):
    strategies.append((s4a, "skip-hole 4a"))
if sorted(s4c) != sorted(s1):
    strategies.append((s4c, "skip-hole 4c"))

for exp_set, label in strategies:
    test_full_ppl(exp_set, label)
