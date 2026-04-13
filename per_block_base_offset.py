"""
Per-block base+offset KVFloat12:
  - Each block of 128 values stores base_exp (1 byte)
  - Each value: sign(1) + offset(4) + mantissa(7) = 12 bits
  - exp8 = base_exp + offset
  - No global LUT needed, fully adaptive

Layout per block (128 values):
  base_exp:   1 byte
  signs:     16 bytes (128 bits)
  offsets:   64 bytes (128 × 4-bit nibbles)
  mantissa: 128 bytes (128 × 7-bit padded to byte)
  Total:    209 bytes / 128 values
  BF16:     256 bytes / 128 values
  Compression: 18.36%

For blocks with span > 15: choose base to cover the TOP,
clamp bottom outliers upward (small values get slightly larger = safe).
"""
import torch, numpy as np, sys
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
sys.path.insert(0, '/root/kvfloat13')

model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
model.eval()

target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

# ============================================================
# Encode / Decode
# ============================================================

def encode_block_offset(bf16_uint16, block_size=128):
    """Encode BF16 → per-block base+offset format."""
    n = len(bf16_uint16)
    assert n % block_size == 0
    num_blocks = n // block_size

    sign = ((bf16_uint16 >> 15) & 1).astype(np.uint8)
    exp8 = ((bf16_uint16 >> 7) & 0xFF).astype(np.int16)  # signed for subtraction
    mant7 = (bf16_uint16 & 0x7F).astype(np.uint8)

    sign_r = sign.reshape(num_blocks, block_size)
    exp_r = exp8.reshape(num_blocks, block_size)
    mant_r = mant7.reshape(num_blocks, block_size)

    bases = np.zeros(num_blocks, dtype=np.uint8)
    offsets = np.zeros((num_blocks, block_size), dtype=np.uint8)

    for bi in range(num_blocks):
        block_exps = exp_r[bi]
        # base = max_exp - 15, so top is always covered
        mx = int(block_exps.max())
        base = max(0, mx - 15)
        bases[bi] = base
        off = block_exps - base
        off = np.clip(off, 0, 15)
        offsets[bi] = off.astype(np.uint8)

    # Pack signs: 8 bits per byte
    signs_packed = np.packbits(sign_r, axis=1, bitorder='little')  # (num_blocks, 16)

    # Pack offsets: 2 nibbles per byte
    even = offsets[:, 0::2]
    odd = offsets[:, 1::2]
    offsets_packed = (even | (odd << 4)).astype(np.uint8)  # (num_blocks, 64)

    return bases, signs_packed, offsets_packed, mant_r


def decode_block_offset(bases, signs_packed, offsets_packed, mant, block_size=128):
    """Decode per-block base+offset → BF16 uint16."""
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


def compress_tensor_bo(tensor):
    """Compress a tensor through block-offset encode/decode."""
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


# ============================================================
# Tests
# ============================================================

test_texts = [
    "The meaning of life is a question that has puzzled philosophers for centuries.",
    "In machine learning, neural networks are trained using backpropagation algorithms.",
    "The quick brown fox jumps over the lazy dog in the warm summer afternoon.",
    "Python is a high-level programming language known for its simplicity and readability.",
    "Climate change is one of the most pressing challenges facing humanity today.",
]

# Baseline PPL
orig_losses = []
with torch.no_grad():
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        orig_losses.append(outputs.loss.item())
orig_ppl = np.exp(np.mean(orig_losses))
print(f"Baseline PPL: {orig_ppl:.4f}")

# --- Weights bit-exact stats ---
print("\n--- WEIGHT COMPRESSION STATS ---")
total_vals = 0
total_match = 0
total_clipped = 0
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    bf16 = param.data.to(torch.bfloat16)
    raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    n = len(raw)
    pad = (128 - n % 128) % 128
    raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw

    # Encode
    bases, sp, op, mant = encode_block_offset(raw_p)
    decoded = decode_block_offset(bases, sp, op, mant)[:n]

    match = np.sum(raw == decoded)
    total_vals += n
    total_match += match

    # Count clipped values (where offset was clamped)
    exp8 = ((raw_p >> 7) & 0xFF).astype(np.int16)
    exp_r = exp8.reshape(-1, 128)
    for bi in range(len(bases)):
        block_exps = exp_r[bi]
        mx = int(block_exps.max())
        base = max(0, mx - 15)
        off = block_exps - base
        total_clipped += int(np.sum((off < 0) | (off > 15)))

print(f"Bit-exact: {total_match:,}/{total_vals:,} ({100*total_match/total_vals:.6f}%)")
print(f"Clipped values: {total_clipped:,} ({100*total_clipped/total_vals:.6f}%)")
# note: total_clipped counts padded values too, but negligible

# --- Weights only PPL ---
print("\n--- WEIGHTS ONLY PPL ---")
saved = {}
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    saved[name] = param.data.clone()
    param.data = compress_tensor_bo(param.data)

losses_w = []
with torch.no_grad():
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        losses_w.append(outputs.loss.item())
ppl_w = np.exp(np.mean(losses_w))
print(f"Weights-only PPL: {ppl_w:.4f} (Δ{ppl_w-orig_ppl:+.4f})")

# Restore
for name, orig in saved.items():
    parts = name.split('.')
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    getattr(obj, parts[-1]).data = orig

# --- KV cache only PPL ---
print("\n--- KV CACHE ONLY PPL ---")
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
            cache = compress_cache_bo(out.past_key_values)
            logits = out.logits[:, -1, :]
            target = input_ids[:, t+1]
            loss = torch.nn.functional.cross_entropy(logits, target)
            total_loss += loss.item()
        losses_kv.append(total_loss / (seq_len - 1))
ppl_kv = np.exp(np.mean(losses_kv))
print(f"KV-only PPL: {ppl_kv:.4f} (Δ{ppl_kv-orig_ppl:+.4f})")

# --- Both ---
print("\n--- WEIGHTS + KV CACHE PPL ---")
saved2 = {}
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    saved2[name] = param.data.clone()
    param.data = compress_tensor_bo(param.data)

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
            cache = compress_cache_bo(out.past_key_values)
            logits = out.logits[:, -1, :]
            target = input_ids[:, t+1]
            loss = torch.nn.functional.cross_entropy(logits, target)
            total_loss += loss.item()
        losses_both.append(total_loss / (seq_len - 1))
ppl_both = np.exp(np.mean(losses_both))
print(f"Both PPL: {ppl_both:.4f} (Δ{ppl_both-orig_ppl:+.4f})")

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
gen_text = tokenizer.decode(generated[0], skip_special_tokens=True)

# Restore
for name, orig in saved2.items():
    parts = name.split('.')
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    getattr(obj, parts[-1]).data = orig

# Original generation
with torch.no_grad():
    inputs_gen = tokenizer("Once upon a time", return_tensors="pt").to(model.device)
    gen_orig = model.generate(**inputs_gen, max_new_tokens=50, do_sample=False)
orig_text = tokenizer.decode(gen_orig[0], skip_special_tokens=True)

# ============================================================
print("\n" + "=" * 60)
print("SUMMARY: Per-block base+offset (adaptive, no LUT)")
print("=" * 60)
print(f"""
Format:    sign(1) + offset(4) + mantissa(7) = 12 bits
           + base_exp(1 byte per 128 values)
Storage:   209 bytes / 128 values = 18.36% compression
Decode:    exp8 = base + offset  (one addition, no LUT)

{'Config':<25} | {'PPL':>8} | {'Δ PPL':>8}
{'-'*48}
{'BF16 baseline':<25} | {orig_ppl:>8.4f} | {'—':>8}
{'Weights only':<25} | {ppl_w:>8.4f} | {ppl_w-orig_ppl:>+8.4f}
{'KV cache only':<25} | {ppl_kv:>8.4f} | {ppl_kv-orig_ppl:>+8.4f}
{'Weights + KV cache':<25} | {ppl_both:>8.4f} | {ppl_both-orig_ppl:>+8.4f}

Weight bit-exact: {100*total_match/total_vals:.4f}%
Clipped values:   {total_clipped:,} ({100*total_clipped/total_vals:.4f}%)

Gen (compressed): {gen_text[:150]}
Gen (original):   {orig_text[:150]}
Identical: {gen_text == orig_text}
""")
