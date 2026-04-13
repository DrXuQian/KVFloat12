"""
Hybrid: weights = fixed LUT [115-130] (25% compression)
        KV cache = per-block base+offset (18.36%, adaptive)
"""
import torch, numpy as np, sys
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
sys.path.insert(0, '/root/kvfloat13')
from split_lut_kvfloat12 import encode_kvf12, decode_kvf12
from per_block_base_offset import encode_block_offset, decode_block_offset

model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
model.eval()

target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

# Fixed LUT for weights
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

w_compress, w_decompress = build_contiguous_lut(115, 16)

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
# Hybrid: weights=LUT, KV=base+offset
# ============================================================
print("\n--- HYBRID: weights=LUT[115-130], KV=base+offset ---")
saved = {}
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    saved[name] = param.data.clone()
    param.data = compress_tensor_lut(param.data, w_compress, w_decompress)

losses_hybrid = []
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
            cache = compress_cache_bo(out.past_key_values)
            logits = out.logits[:, -1, :]
            target = input_ids[:, t+1]
            loss = torch.nn.functional.cross_entropy(logits, target)
            total_loss += loss.item()
        losses_hybrid.append(total_loss / (seq_len - 1))
ppl_hybrid = np.exp(np.mean(losses_hybrid))

# Generation
cache = DynamicCache()
generated = tokenizer("Once upon a time", return_tensors="pt")["input_ids"].to(model.device)
with torch.no_grad():
    out = model(generated, past_key_values=cache, use_cache=True)
    cache = compress_cache_bo(out.past_key_values)
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_tok], dim=1)
    for _ in range(49):
        out = model(generated[:, -1:], past_key_values=cache, use_cache=True)
        cache = compress_cache_bo(out.past_key_values)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
hybrid_gen = tokenizer.decode(generated[0], skip_special_tokens=True)

# Restore
for name, orig in saved.items():
    parts = name.split('.')
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    getattr(obj, parts[-1]).data = orig

# Original gen
with torch.no_grad():
    g = model.generate(**tokenizer("Once upon a time", return_tensors="pt").to(model.device),
                       max_new_tokens=50, do_sample=False)
orig_gen = tokenizer.decode(g[0], skip_special_tokens=True)

print(f"Hybrid PPL:  {ppl_hybrid:.4f} (Δ{ppl_hybrid-orig_ppl:+.4f})")
print(f"Gen hybrid:  {hybrid_gen[:150]}")
print(f"Gen orig:    {orig_gen[:150]}")
print(f"Identical:   {hybrid_gen == orig_gen}")

# ============================================================
# Compare all approaches
# ============================================================
print("\n" + "=" * 70)
print("FINAL COMPARISON: ALL APPROACHES (weights + KV cache)")
print("=" * 70)
print(f"""
{'Approach':<45} | {'PPL':>7} | {'Δ PPL':>7} | {'W comp':>6} | {'KV comp':>7}
{'-'*80}
{'BF16 baseline':<45} | {orig_ppl:>7.4f} | {'—':>7} | {'0%':>6} | {'0%':>7}
{'KVFloat13 5-bit global LUT':<45} | {6.6122:>7.4f} | {'+0.000':>7} | {'18.75%':>6} | {'18.75%':>7}
{'KVFloat12 LUT[115-130] both':<45} | {6.6290:>7.4f} | {'+0.017':>7} | {'25.0%':>6} | {'25.0%':>7}
{'Per-block base+offset both':<45} | {6.6507:>7.4f} | {'+0.038':>7} | {'18.4%':>6} | {'18.4%':>7}
{'HYBRID: W=LUT[115-130], KV=base+offset':<45} | {ppl_hybrid:>7.4f} | {ppl_hybrid-orig_ppl:>+7.4f} | {'25.0%':>6} | {'18.4%':>7}
""")
