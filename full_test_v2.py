"""
Test [115-130] LUT on weights AND KV cache, using DynamicCache.
"""
import torch, numpy as np, sys
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
sys.path.insert(0, '/root/kvfloat13')
from split_lut_kvfloat12 import encode_kvf12, decode_kvf12

model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
model.eval()

target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

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

def compress_tensor(tensor, cl, dl):
    shape, device = tensor.shape, tensor.device
    raw = tensor.to(torch.bfloat16).contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    n = len(raw)
    pad = (128 - n % 128) % 128
    raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw
    s1, s2 = encode_kvf12(raw_p, cl)
    decoded = decode_kvf12(s1, s2, dl)[:n]
    return torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(device)

def compress_cache(cache, cl, dl):
    """Compress all KV tensors in a DynamicCache."""
    for layer in cache.layers:
        layer.keys = compress_tensor(layer.keys, cl, dl)
        layer.values = compress_tensor(layer.values, cl, dl)
    return cache

c, d = build_contiguous_lut(115, 16)

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
print(f"Baseline PPL: {orig_ppl:.4f}")

# ============================================================
# Test: KV cache only
# ============================================================
print("\n--- KV CACHE ONLY ---")
losses_kv = []
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
            cache = out.past_key_values
            cache = compress_cache(cache, c, d)
            logits = out.logits[:, -1, :]
            target = input_ids[:, t+1]
            loss = torch.nn.functional.cross_entropy(logits, target)
            total_loss += loss.item()
        losses_kv.append(total_loss / (seq_len - 1))
ppl_kv = np.exp(np.mean(losses_kv))
print(f"KV-only PPL: {ppl_kv:.4f}")

# ============================================================
# Test: Weights only (confirm)
# ============================================================
print("\n--- WEIGHTS ONLY ---")
saved = {}
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    saved[name] = param.data.clone()
    param.data = compress_tensor(param.data, c, d)

losses_w = []
with torch.no_grad():
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        losses_w.append(outputs.loss.item())
ppl_w = np.exp(np.mean(losses_w))
print(f"Weights-only PPL: {ppl_w:.4f}")

# ============================================================
# Test: Both (weights still compressed, add KV compression)
# ============================================================
print("\n--- WEIGHTS + KV CACHE ---")
losses_both = []
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
            cache = out.past_key_values
            cache = compress_cache(cache, c, d)
            logits = out.logits[:, -1, :]
            target = input_ids[:, t+1]
            loss = torch.nn.functional.cross_entropy(logits, target)
            total_loss += loss.item()
        losses_both.append(total_loss / (seq_len - 1))
ppl_both = np.exp(np.mean(losses_both))
print(f"Both PPL: {ppl_both:.4f}")

# Generation
cache = DynamicCache()
generated = tokenizer("Once upon a time", return_tensors="pt")["input_ids"].to(model.device)
with torch.no_grad():
    # First forward with full prompt
    out = model(generated, past_key_values=cache, use_cache=True)
    cache = compress_cache(out.past_key_values, c, d)
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_tok], dim=1)

    for _ in range(49):
        out = model(generated[:, -1:], past_key_values=cache, use_cache=True)
        cache = compress_cache(out.past_key_values, c, d)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)

gen_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"Gen: {gen_text[:200]}")

# Restore weights
for name, orig in saved.items():
    parts = name.split('.')
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    getattr(obj, parts[-1]).data = orig

# Original generation for comparison
with torch.no_grad():
    inputs_gen = tokenizer("Once upon a time", return_tensors="pt").to(model.device)
    gen_orig = model.generate(**inputs_gen, max_new_tokens=50, do_sample=False)
orig_gen_text = tokenizer.decode(gen_orig[0], skip_special_tokens=True)
print(f"Orig: {orig_gen_text[:200]}")

# ============================================================
print("\n" + "=" * 50)
print("SUMMARY: [115-130] KVFloat12 (25% compression)")
print("=" * 50)
print(f"{'Config':<25} | {'PPL':>8} | {'vs baseline':>12}")
print("-" * 50)
print(f"{'BF16 baseline':<25} | {orig_ppl:>8.4f} | {'—':>12}")
print(f"{'Weights only':<25} | {ppl_w:>8.4f} | {ppl_w-orig_ppl:>+11.4f}")
print(f"{'KV cache only':<25} | {ppl_kv:>8.4f} | {ppl_kv-orig_ppl:>+11.4f}")
print(f"{'Weights + KV cache':<25} | {ppl_both:>8.4f} | {ppl_both-orig_ppl:>+11.4f}")
