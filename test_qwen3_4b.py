"""
Test all KVFloat12 approaches on Qwen3-4B (larger model).
1) Exponent distribution analysis
2) Per-block base+offset stats
3) PPL comparison: LUT [115-130] vs base+offset vs hybrid
"""
import torch, numpy as np, sys, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from collections import Counter
sys.path.insert(0, '/root/kvfloat13')
from split_lut_kvfloat12 import encode_kvf12, decode_kvf12
from per_block_base_offset import encode_block_offset, decode_block_offset

model_path = "/root/autodl-tmp/Qwen3-4B"
print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Config: {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads, "
      f"hidden={model.config.hidden_size}")

target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

# ============================================================
# Part 1: Exponent distribution
# ============================================================
print("\n" + "=" * 70)
print("PART 1: EXPONENT DISTRIBUTION")
print("=" * 70)

w_counter = Counter()
w_weighted = Counter()
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    bf16 = param.data.to(torch.bfloat16)
    raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    exps = ((raw >> 7) & 0xFF).astype(int)
    vals = np.abs(bf16.contiguous().view(-1).float().cpu().numpy())
    for e in np.unique(exps):
        mask = exps == e
        w_counter[e] += int(mask.sum())
        w_weighted[e] += float(vals[mask].sum())

total_w = sum(w_counter.values())
print(f"\nWeight exponents ({len(w_counter)} unique):")
print(f"{'exp':>4} | {'count':>12} | {'count%':>8} | {'sum|val|':>12} | {'val%':>8}")
print("-" * 55)
total_val = sum(w_weighted.values())
for e in sorted(w_counter.keys()):
    c = w_counter[e]
    v = w_weighted[e]
    if c / total_w > 0.0001 or v / total_val > 0.001:  # only show significant
        print(f"{e:>4} | {c:>12,} | {100*c/total_w:>7.4f}% | {v:>12.1f} | {100*v/total_val:>7.4f}%")

# KV cache
kv_counter = Counter()
prompts_cal = [
    "The quick brown fox jumps over the lazy dog.",
    "In machine learning, neural networks are trained using backpropagation.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
]
with torch.no_grad():
    for prompt in prompts_cal:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs, use_cache=True)
        for layer in outputs.past_key_values.layers:
            for tensor in [layer.keys, layer.values]:
                exps = ((tensor.to(torch.bfloat16).view(torch.int16).to(torch.int32) >> 7) & 0xFF)
                kv_counter.update(exps.cpu().numpy().flatten().tolist())

total_kv = sum(kv_counter.values())
print(f"\nKV cache exponents ({len(kv_counter)} unique):")
for e in sorted(kv_counter.keys()):
    c = kv_counter[e]
    if c / total_kv > 0.001:
        print(f"  exp={e}: {100*c/total_kv:.3f}%")

# Top-32 coverage
for k in [16, 32]:
    w_topk = [e for e, _ in sorted(w_counter.items(), key=lambda x: -x[1])[:k]]
    kv_topk = [e for e, _ in sorted(kv_counter.items(), key=lambda x: -x[1])[:k]]
    w_cov = 100 * sum(w_counter[e] for e in w_topk) / total_w
    kv_cov = 100 * sum(kv_counter[e] for e in kv_topk) / total_kv
    print(f"\nTop-{k} coverage: weights={w_cov:.4f}%, KV={kv_cov:.4f}%")
    print(f"  W top-{k}: {sorted(w_topk)}")
    print(f"  KV top-{k}: {sorted(kv_topk)}")

# ============================================================
# Part 2: Per-block span analysis
# ============================================================
print("\n" + "=" * 70)
print("PART 2: PER-BLOCK SPAN")
print("=" * 70)

block_spans = []
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    raw = param.data.to(torch.bfloat16).contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    exps = ((raw >> 7) & 0xFF).astype(int)
    n = len(exps)
    pad = (128 - n % 128) % 128
    if pad:
        exps = np.concatenate([exps, np.zeros(pad, dtype=int)])
    blocks = exps.reshape(-1, 128)
    spans = blocks.max(axis=1) - blocks.min(axis=1)
    block_spans.extend(spans.tolist())

block_spans = np.array(block_spans)
print(f"Weight blocks: {len(block_spans):,}")
print(f"Span: min={block_spans.min()} max={block_spans.max()} mean={block_spans.mean():.1f} median={np.median(block_spans):.0f}")
for t in [8, 15, 16, 20, 24, 31]:
    print(f"  span ≤ {t:>2}: {100*np.sum(block_spans<=t)/len(block_spans):.3f}%")

# ============================================================
# Part 3: Sliding window — find best LUT for this model
# ============================================================
print("\n" + "=" * 70)
print("PART 3: FIND BEST FIXED LUT WINDOW")
print("=" * 70)

combined = w_counter + kv_counter
all_exps = sorted(combined.keys())
print(f"Combined exp range: [{all_exps[0]}-{all_exps[-1]}]")

# Quick error estimate for different windows
def estimate_error(window, stats):
    supported = set(window)
    total_err = 0.0
    for e, (count, sum_val) in stats.items():
        if e in supported:
            continue
        best_se = min(window, key=lambda se: abs(2.0**(e-127) - 2.0**(se-127)))
        ratio = 2.0**(best_se - e)
        total_err += abs(1.0 - ratio) * sum_val
    return total_err

# Build combined stats
combined_stats = {}
for e in set(list(w_counter.keys()) + list(kv_counter.keys())):
    combined_stats[e] = (w_counter.get(e, 0) + kv_counter.get(e, 0),
                         w_weighted.get(e, 0))

for base in range(110, 122):
    window = list(range(base, base + 16))
    err = estimate_error(window, combined_stats)
    print(f"  [{base}-{base+15}]: error={err:.2f}")

# ============================================================
# Part 4: PPL tests
# ============================================================
print("\n" + "=" * 70)
print("PART 4: PPL COMPARISON")
print("=" * 70)

def build_contiguous_lut(base, width=16):
    exps = list(range(base, base + width))
    decompress = np.array(exps, dtype=np.uint8)
    compress = np.zeros(256, dtype=np.uint8)
    for i, e in enumerate(exps):
        compress[e] = i
    for e in range(256):
        if e in set(exps):
            continue
        best_i, best_d = 0, float('inf')
        for se in exps:
            d = abs(2.0**(e-127) - 2.0**(se-127))
            if d < best_d:
                best_d, best_i = d, exps.index(se)
        compress[e] = best_i
    return compress, decompress

def compress_tensor_lut(tensor, cl, dl):
    shape, device = tensor.shape, tensor.device
    raw = tensor.to(torch.bfloat16).contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    n = len(raw)
    pad = (128 - n % 128) % 128
    raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw
    s1, s2 = encode_kvf12(raw_p, cl)
    decoded = decode_kvf12(s1, s2, dl)[:n]
    return torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(device)

def compress_tensor_bo(tensor):
    shape, device = tensor.shape, tensor.device
    raw = tensor.to(torch.bfloat16).contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    n = len(raw)
    pad = (128 - n % 128) % 128
    raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw
    bases, sp, op, mant = encode_block_offset(raw_p)
    decoded = decode_block_offset(bases, sp, op, mant)[:n]
    return torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(device)

def compress_cache_lut(cache, cl, dl):
    for layer in cache.layers:
        layer.keys = compress_tensor_lut(layer.keys, cl, dl)
        layer.values = compress_tensor_lut(layer.values, cl, dl)
    return cache

def compress_cache_bo(cache):
    for layer in cache.layers:
        layer.keys = compress_tensor_bo(layer.keys)
        layer.values = compress_tensor_bo(layer.values)
    return cache

test_texts = [
    "The meaning of life is a question that has puzzled philosophers for centuries.",
    "In machine learning, neural networks are trained using backpropagation algorithms.",
    "The quick brown fox jumps over the lazy dog in the warm summer afternoon.",
    "Python is a high-level programming language known for its simplicity and readability.",
    "Climate change is one of the most pressing challenges facing humanity today.",
]

def compute_ppl_baseline():
    losses = []
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())
    return np.exp(np.mean(losses))

def compute_ppl_with_kv_compress(compress_kv_fn):
    """PPL with KV cache compression (token-by-token)."""
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
                cache = compress_kv_fn(out.past_key_values)
                logits = out.logits[:, -1, :]
                target = input_ids[:, t+1]
                loss = torch.nn.functional.cross_entropy(logits, target)
                total_loss += loss.item()
            losses.append(total_loss / (seq_len - 1))
    return np.exp(np.mean(losses))

def compress_weights(compress_fn, *args):
    saved = {}
    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        saved[name] = param.data.clone()
        param.data = compress_fn(param.data, *args)
    return saved

def restore_weights(saved):
    for name, orig in saved.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        getattr(obj, parts[-1]).data = orig

# Baseline
orig_ppl = compute_ppl_baseline()
print(f"Baseline PPL: {orig_ppl:.4f}")

# Find best window first with weights-only quick test
print("\nSliding window weights-only PPL:")
best_base, best_ppl = 115, 999
for base in [113, 114, 115, 116, 117, 118]:
    cl, dl = build_contiguous_lut(base, 16)
    saved = compress_weights(compress_tensor_lut, cl, dl)
    ppl = compute_ppl_baseline()
    restore_weights(saved)
    marker = ""
    if ppl < best_ppl:
        best_ppl, best_base = ppl, base
        marker = " ← best"
    print(f"  [{base}-{base+15}] PPL={ppl:.4f} (Δ{ppl-orig_ppl:+.4f}){marker}")

print(f"\nBest window: [{best_base}-{best_base+15}]")
best_cl, best_dl = build_contiguous_lut(best_base, 16)

# Full tests with best window
print(f"\n--- Full tests with [{best_base}-{best_base+15}] ---")

# A) LUT both
saved = compress_weights(compress_tensor_lut, best_cl, best_dl)
ppl_lut_both = compute_ppl_with_kv_compress(lambda c: compress_cache_lut(c, best_cl, best_dl))
restore_weights(saved)
print(f"LUT both:        PPL={ppl_lut_both:.4f} (Δ{ppl_lut_both-orig_ppl:+.4f})")

# B) base+offset both
saved = compress_weights(compress_tensor_bo)
ppl_bo_both = compute_ppl_with_kv_compress(compress_cache_bo)
restore_weights(saved)
print(f"base+offset both: PPL={ppl_bo_both:.4f} (Δ{ppl_bo_both-orig_ppl:+.4f})")

# C) Hybrid: weights=LUT, KV=base+offset
saved = compress_weights(compress_tensor_lut, best_cl, best_dl)
ppl_hybrid = compute_ppl_with_kv_compress(compress_cache_bo)
restore_weights(saved)
print(f"Hybrid W=LUT KV=BO: PPL={ppl_hybrid:.4f} (Δ{ppl_hybrid-orig_ppl:+.4f})")

# D) Hybrid: weights=base+offset, KV=LUT
saved = compress_weights(compress_tensor_bo)
ppl_hybrid2 = compute_ppl_with_kv_compress(lambda c: compress_cache_lut(c, best_cl, best_dl))
restore_weights(saved)
print(f"Hybrid W=BO KV=LUT: PPL={ppl_hybrid2:.4f} (Δ{ppl_hybrid2-orig_ppl:+.4f})")

# Generation test with best approach
print("\n--- Generation test ---")
saved = compress_weights(compress_tensor_lut, best_cl, best_dl)
cache = DynamicCache()
prompt = "Once upon a time"
generated = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
with torch.no_grad():
    out = model(generated, past_key_values=cache, use_cache=True)
    cache = compress_cache_bo(out.past_key_values)
    for _ in range(50):
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
        out = model(generated[:, -1:], past_key_values=cache, use_cache=True)
        cache = compress_cache_bo(out.past_key_values)
gen_text = tokenizer.decode(generated[0], skip_special_tokens=True)
restore_weights(saved)

with torch.no_grad():
    g = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device),
                       max_new_tokens=50, do_sample=False)
orig_text = tokenizer.decode(g[0], skip_special_tokens=True)

print(f"Hybrid: {gen_text[:200]}")
print(f"Orig:   {orig_text[:200]}")
print(f"Match:  {gen_text == orig_text}")

# ============================================================
print("\n" + "=" * 70)
print(f"FINAL SUMMARY: Qwen3-4B")
print("=" * 70)
print(f"""
{'Approach':<40} | {'PPL':>8} | {'Δ PPL':>8} | {'W comp':>7} | {'KV comp':>7}
{'-'*78}
{'BF16 baseline':<40} | {orig_ppl:>8.4f} | {'—':>8} | {'0%':>7} | {'0%':>7}
{'LUT [{0}-{1}] both'.format(best_base, best_base+15):<40} | {ppl_lut_both:>8.4f} | {ppl_lut_both-orig_ppl:>+8.4f} | {'25.0%':>7} | {'25.0%':>7}
{'base+offset both':<40} | {ppl_bo_both:>8.4f} | {ppl_bo_both-orig_ppl:>+8.4f} | {'18.4%':>7} | {'18.4%':>7}
{'Hybrid W=LUT KV=base+offset':<40} | {ppl_hybrid:>8.4f} | {ppl_hybrid-orig_ppl:>+8.4f} | {'25.0%':>7} | {'18.4%':>7}
{'Hybrid W=base+offset KV=LUT':<40} | {ppl_hybrid2:>8.4f} | {ppl_hybrid2-orig_ppl:>+8.4f} | {'18.4%':>7} | {'25.0%':>7}
""")
