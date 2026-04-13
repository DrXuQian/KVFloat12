"""
Fast Qwen3-4B test: vectorized encode/decode, skip token-by-token for KV.
"""
import torch, numpy as np, sys, time
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from collections import Counter
sys.path.insert(0, '/root/kvfloat13')

# ============================================================
# FAST vectorized encode/decode (no Python loops over blocks)
# ============================================================

def encode_decode_lut_fast(bf16_uint16, compress_lut, decompress_lut):
    """Round-trip through LUT compression. Fully vectorized."""
    sign = (bf16_uint16 >> 15) & 1
    exp8 = (bf16_uint16 >> 7) & 0xFF
    mant7 = bf16_uint16 & 0x7F
    exp4 = compress_lut[exp8.astype(np.uint8)]
    exp8_new = decompress_lut[exp4].astype(np.uint16)
    return ((sign << 15) | (exp8_new << 7) | mant7).astype(np.uint16)

def encode_decode_bo_fast(bf16_uint16):
    """Round-trip through per-block base+offset. Vectorized."""
    n = len(bf16_uint16)
    pad = (128 - n % 128) % 128
    if pad:
        bf16_uint16 = np.concatenate([bf16_uint16, np.zeros(pad, dtype=np.uint16)])

    sign = (bf16_uint16 >> 15) & 1
    exp8 = ((bf16_uint16 >> 7) & 0xFF).astype(np.int16)
    mant7 = bf16_uint16 & 0x7F

    blocks_exp = exp8.reshape(-1, 128)
    blocks_sign = sign.reshape(-1, 128)
    blocks_mant = mant7.reshape(-1, 128)

    # Vectorized base computation: base = max(0, max_per_block - 15)
    maxes = blocks_exp.max(axis=1)  # (num_blocks,)
    bases = np.maximum(0, maxes - 15).astype(np.int16)

    # offset = exp - base, clamp to [0, 15]
    offsets = blocks_exp - bases[:, None]
    offsets = np.clip(offsets, 0, 15)

    # Reconstruct
    exp8_new = (bases[:, None] + offsets).astype(np.uint16)
    result = (blocks_sign.astype(np.uint16) << 15) | (exp8_new << 7) | blocks_mant.astype(np.uint16)
    return result.reshape(-1)[:n]

def compress_tensor_lut_fast(tensor, cl, dl):
    shape, device = tensor.shape, tensor.device
    raw = tensor.to(torch.bfloat16).contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    decoded = encode_decode_lut_fast(raw, cl, dl)
    return torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(device)

def compress_tensor_bo_fast(tensor):
    shape, device = tensor.shape, tensor.device
    raw = tensor.to(torch.bfloat16).contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    decoded = encode_decode_bo_fast(raw)
    return torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(device)

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

# ============================================================

model_path = "/root/autodl-tmp/Qwen3-4B"
print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")
print(f"Layers: {model.config.num_hidden_layers}, Heads: {model.config.num_attention_heads}, "
      f"Hidden: {model.config.hidden_size}")

target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

# ============================================================
# Part 1: Exponent distribution (fast)
# ============================================================
print("\n" + "=" * 60)
print("EXPONENT DISTRIBUTION")
print("=" * 60)

t0 = time.time()
w_counter = Counter()
w_weighted = Counter()
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    raw = param.data.to(torch.bfloat16).contiguous().view(-1).view(torch.int16)
    exps = ((raw.to(torch.int32) >> 7) & 0xFF)
    # Use torch for speed
    unique_exps, counts = exps.unique(return_counts=True)
    for e, c in zip(unique_exps.cpu().numpy(), counts.cpu().numpy()):
        w_counter[int(e)] += int(c)

total_w = sum(w_counter.values())
print(f"Weight scan: {time.time()-t0:.1f}s, {total_w:,} values, {len(w_counter)} unique exps")
print(f"Exp range: [{min(w_counter.keys())}-{max(w_counter.keys())}]")
print(f"Top-10: {[e for e,_ in w_counter.most_common(10)]}")

# Top-k coverage
for k in [16, 32]:
    topk = [e for e, _ in w_counter.most_common(k)]
    cov = 100 * sum(w_counter[e] for e in topk) / total_w
    print(f"Top-{k} coverage: {cov:.4f}% exps={sorted(topk)}")

# Per-block span (sample 10% of blocks for speed)
t0 = time.time()
all_spans = []
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    raw = param.data.to(torch.bfloat16).contiguous().view(-1).view(torch.int16).to(torch.int32)
    exps = ((raw >> 7) & 0xFF)
    n = exps.shape[0]
    pad = (128 - n % 128) % 128
    if pad:
        exps = torch.cat([exps, torch.zeros(pad, dtype=torch.int32, device=exps.device)])
    blocks = exps.reshape(-1, 128)
    spans = blocks.max(dim=1).values - blocks.min(dim=1).values
    all_spans.append(spans.cpu().numpy())

all_spans = np.concatenate(all_spans)
print(f"\nPer-block span ({len(all_spans):,} blocks, {time.time()-t0:.1f}s):")
print(f"  mean={all_spans.mean():.1f} median={np.median(all_spans):.0f} max={all_spans.max()}")
for t in [8, 15, 16, 20, 31]:
    print(f"  span ≤ {t:>2}: {100*np.sum(all_spans<=t)/len(all_spans):.3f}%")

# ============================================================
# Part 2: PPL tests
# ============================================================
print("\n" + "=" * 60)
print("PPL COMPARISON")
print("=" * 60)

test_texts = [
    "The meaning of life is a question that has puzzled philosophers for centuries.",
    "In machine learning, neural networks are trained using backpropagation algorithms.",
    "The quick brown fox jumps over the lazy dog in the warm summer afternoon.",
    "Python is a high-level programming language known for its simplicity and readability.",
    "Climate change is one of the most pressing challenges facing humanity today.",
]

def compute_ppl():
    losses = []
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())
    return np.exp(np.mean(losses))

def compress_all_weights(compress_fn, *args):
    saved = {}
    t0 = time.time()
    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        saved[name] = param.data.clone()
        param.data = compress_fn(param.data, *args)
    print(f"  Weight compression: {time.time()-t0:.1f}s")
    return saved

def restore_weights(saved):
    for name, orig in saved.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        getattr(obj, parts[-1]).data = orig

# Baseline
t0 = time.time()
orig_ppl = compute_ppl()
print(f"Baseline PPL: {orig_ppl:.4f} ({time.time()-t0:.1f}s)")

# Sliding window (weights only, fast)
print("\nSliding window (weights only):")
best_base, best_wppl = 115, 999
for base in [113, 114, 115, 116, 117, 118, 119]:
    cl, dl = build_contiguous_lut(base, 16)
    t0 = time.time()
    saved = compress_all_weights(compress_tensor_lut_fast, cl, dl)
    ppl = compute_ppl()
    restore_weights(saved)
    marker = ""
    if ppl < best_wppl:
        best_wppl, best_base = ppl, base
        marker = " ← best"
    print(f"  [{base}-{base+15}] PPL={ppl:.4f} (Δ{ppl-orig_ppl:+.4f}) {time.time()-t0:.1f}s{marker}")

best_cl, best_dl = build_contiguous_lut(best_base, 16)
print(f"Best window: [{best_base}-{best_base+15}]")

# base+offset weights only
print("\nbase+offset (weights only):")
t0 = time.time()
saved = compress_all_weights(compress_tensor_bo_fast)
ppl_bo_w = compute_ppl()
restore_weights(saved)
print(f"  base+offset PPL={ppl_bo_w:.4f} (Δ{ppl_bo_w-orig_ppl:+.4f}) {time.time()-t0:.1f}s")

# KV cache compression: use single-pass (compress after full forward, re-eval)
# This approximates token-by-token compression with much less cost
print("\nKV cache tests (single forward, compress KV, re-eval next-token):")

def test_kv_compression(compress_kv_fn, label):
    """Approximate KV cache compression test using 2-pass approach."""
    losses = []
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]

            # Token-by-token with KV compression
            cache = DynamicCache()
            total_loss = 0.0
            for t in range(seq_len - 1):
                tok = input_ids[:, t:t+1]
                out = model(tok, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                compress_kv_fn(cache)
                logits = out.logits[:, -1, :]
                target = input_ids[:, t+1]
                loss = torch.nn.functional.cross_entropy(logits, target)
                total_loss += loss.item()
            losses.append(total_loss / (seq_len - 1))
    ppl = np.exp(np.mean(losses))
    print(f"  [{label}] PPL={ppl:.4f} (Δ{ppl-orig_ppl:+.4f})")
    return ppl

def compress_kv_lut_inplace(cache):
    for layer in cache.layers:
        layer.keys = compress_tensor_lut_fast(layer.keys, best_cl, best_dl)
        layer.values = compress_tensor_lut_fast(layer.values, best_cl, best_dl)

def compress_kv_bo_inplace(cache):
    for layer in cache.layers:
        layer.keys = compress_tensor_bo_fast(layer.keys)
        layer.values = compress_tensor_bo_fast(layer.values)

t0 = time.time()
ppl_kv_lut = test_kv_compression(compress_kv_lut_inplace, f"KV=LUT[{best_base}-{best_base+15}]")
print(f"  ({time.time()-t0:.1f}s)")

t0 = time.time()
ppl_kv_bo = test_kv_compression(compress_kv_bo_inplace, "KV=base+offset")
print(f"  ({time.time()-t0:.1f}s)")

# Full: weights + KV
print("\nFull (weights + KV cache):")

# A) LUT both
t0 = time.time()
saved = compress_all_weights(compress_tensor_lut_fast, best_cl, best_dl)
ppl_lut_both = test_kv_compression(compress_kv_lut_inplace, f"LUT[{best_base}-{best_base+15}] both")
restore_weights(saved)
print(f"  ({time.time()-t0:.1f}s)")

# B) base+offset both
t0 = time.time()
saved = compress_all_weights(compress_tensor_bo_fast)
ppl_bo_both = test_kv_compression(compress_kv_bo_inplace, "base+offset both")
restore_weights(saved)
print(f"  ({time.time()-t0:.1f}s)")

# C) Hybrid: W=LUT, KV=BO
t0 = time.time()
saved = compress_all_weights(compress_tensor_lut_fast, best_cl, best_dl)
ppl_hybrid = test_kv_compression(compress_kv_bo_inplace, "Hybrid W=LUT KV=BO")
restore_weights(saved)
print(f"  ({time.time()-t0:.1f}s)")

# Generation
print("\n--- Generation ---")
saved = compress_all_weights(compress_tensor_lut_fast, best_cl, best_dl)
cache = DynamicCache()
prompt = "Once upon a time"
generated = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
with torch.no_grad():
    out = model(generated, past_key_values=cache, use_cache=True)
    compress_kv_bo_inplace(out.past_key_values)
    cache = out.past_key_values
    for _ in range(50):
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
        out = model(generated[:, -1:], past_key_values=cache, use_cache=True)
        compress_kv_bo_inplace(out.past_key_values)
        cache = out.past_key_values
hybrid_gen = tokenizer.decode(generated[0], skip_special_tokens=True)
restore_weights(saved)

with torch.no_grad():
    g = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device),
                       max_new_tokens=50, do_sample=False)
orig_gen = tokenizer.decode(g[0], skip_special_tokens=True)
print(f"Hybrid: {hybrid_gen[:200]}")
print(f"Orig:   {orig_gen[:200]}")

# ============================================================
print("\n" + "=" * 70)
print(f"FINAL SUMMARY: Qwen3-4B ({n_params/1e9:.1f}B params)")
print("=" * 70)
print(f"""
{'Approach':<40} | {'PPL':>8} | {'Δ PPL':>8} | {'W comp':>7} | {'KV comp':>7}
{'-'*78}
{'BF16 baseline':<40} | {orig_ppl:>8.4f} | {'—':>8} | {'0%':>7} | {'0%':>7}
{'W=LUT[{0}-{1}] only'.format(best_base,best_base+15):<40} | {best_wppl:>8.4f} | {best_wppl-orig_ppl:>+8.4f} | {'25.0%':>7} | {'0%':>7}
{'W=base+offset only':<40} | {ppl_bo_w:>8.4f} | {ppl_bo_w-orig_ppl:>+8.4f} | {'18.4%':>7} | {'0%':>7}
{'KV=LUT[{0}-{1}] only'.format(best_base,best_base+15):<40} | {ppl_kv_lut:>8.4f} | {ppl_kv_lut-orig_ppl:>+8.4f} | {'0%':>7} | {'25.0%':>7}
{'KV=base+offset only':<40} | {ppl_kv_bo:>8.4f} | {ppl_kv_bo-orig_ppl:>+8.4f} | {'0%':>7} | {'18.4%':>7}
{'LUT both':<40} | {ppl_lut_both:>8.4f} | {ppl_lut_both-orig_ppl:>+8.4f} | {'25.0%':>7} | {'25.0%':>7}
{'base+offset both':<40} | {ppl_bo_both:>8.4f} | {ppl_bo_both-orig_ppl:>+8.4f} | {'18.4%':>7} | {'18.4%':>7}
{'Hybrid W=LUT KV=BO':<40} | {ppl_hybrid:>8.4f} | {ppl_hybrid-orig_ppl:>+8.4f} | {'25.0%':>7} | {'18.4%':>7}
""")
