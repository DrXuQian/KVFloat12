"""
Test [115-130] LUT on BOTH weights AND KV cache simultaneously.
"""
import torch, numpy as np, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    exp_set = set(exps)
    for i, e in enumerate(exps):
        compress[e] = i
    for e in range(256):
        if e in exp_set:
            continue
        best_i, best_d = 0, float('inf')
        for se in exps:
            d = abs(2.0**(e-127) - 2.0**(se-127))
            if d < best_d:
                best_d = d
                best_i = exps.index(se)
        compress[e] = best_i
    return compress, decompress

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
# Test 1: Weights only (confirm previous result)
# ============================================================
print("\n" + "=" * 65)
print("TEST 1: WEIGHTS ONLY with [115-130]")
print("=" * 65)

c, d = build_contiguous_lut(115, 16)
saved_w = {}
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    bf16 = param.data.to(torch.bfloat16)
    shape = bf16.shape
    raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    n = len(raw)
    pad = (128 - n % 128) % 128
    raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw
    s1, s2 = encode_kvf12(raw_p, c)
    decoded = decode_kvf12(s1, s2, d)[:n]
    dec_t = torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(param.device)
    saved_w[name] = param.data.clone()
    param.data = dec_t

losses = []
with torch.no_grad():
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())
ppl_w = np.exp(np.mean(losses))
print(f"Weights-only PPL: {ppl_w:.4f}")

# Restore
for name, orig in saved_w.items():
    parts = name.split('.')
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    getattr(obj, parts[-1]).data = orig

# ============================================================
# Test 2: KV cache only — hook into model to compress KV on-the-fly
# ============================================================
print("\n" + "=" * 65)
print("TEST 2: KV CACHE ONLY with [115-130]")
print("=" * 65)

def compress_kv_tensor(tensor, compress_lut, decompress_lut):
    """Compress a KV cache tensor through KVFloat12."""
    shape = tensor.shape
    device = tensor.device
    bf16 = tensor.to(torch.bfloat16)
    raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    n = len(raw)
    pad = (128 - n % 128) % 128
    raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw
    s1, s2 = encode_kvf12(raw_p, compress_lut)
    decoded = decode_kvf12(s1, s2, decompress_lut)[:n]
    return torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(device)

# Run inference, compress KV cache, then re-run with compressed KV
losses_kv = []
with torch.no_grad():
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        # First pass: get KV cache
        outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values

        # Compress KV cache
        compressed_kv = []
        for layer_kv in past_kv:
            ck = compress_kv_tensor(layer_kv[0], c, d)
            cv = compress_kv_tensor(layer_kv[1], c, d)
            compressed_kv.append((ck, cv))

        # Re-run with compressed KV to get loss
        # We need to compute loss with the compressed KV cache
        # Run a forward pass where we provide the compressed past_kv
        # But for PPL we need the full sequence loss, not just next-token
        # So let's do it token by token
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        total_loss = 0.0
        past = None
        for t in range(seq_len - 1):
            tok = input_ids[:, t:t+1]
            out = model(tok, past_key_values=past, use_cache=True)
            # Compress the KV cache
            new_past = []
            for layer_kv in out.past_key_values:
                ck = compress_kv_tensor(layer_kv[0], c, d)
                cv = compress_kv_tensor(layer_kv[1], c, d)
                new_past.append((ck, cv))
            past = new_past

            # Loss for next token
            logits = out.logits[:, -1, :]
            target = input_ids[:, t+1]
            loss = torch.nn.functional.cross_entropy(logits, target)
            total_loss += loss.item()

        avg_loss = total_loss / (seq_len - 1)
        losses_kv.append(avg_loss)

ppl_kv = np.exp(np.mean(losses_kv))
print(f"KV-cache-only PPL: {ppl_kv:.4f}")

# ============================================================
# Test 3: BOTH weights AND KV cache compressed
# ============================================================
print("\n" + "=" * 65)
print("TEST 3: WEIGHTS + KV CACHE with [115-130]")
print("=" * 65)

# Compress weights
saved_w2 = {}
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    bf16 = param.data.to(torch.bfloat16)
    shape = bf16.shape
    raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    n = len(raw)
    pad = (128 - n % 128) % 128
    raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw
    s1, s2 = encode_kvf12(raw_p, c)
    decoded = decode_kvf12(s1, s2, d)[:n]
    dec_t = torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(param.device)
    saved_w2[name] = param.data.clone()
    param.data = dec_t

# Run with compressed weights + compressed KV
losses_both = []
with torch.no_grad():
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        total_loss = 0.0
        past = None
        for t in range(seq_len - 1):
            tok = input_ids[:, t:t+1]
            out = model(tok, past_key_values=past, use_cache=True)
            new_past = []
            for layer_kv in out.past_key_values:
                ck = compress_kv_tensor(layer_kv[0], c, d)
                cv = compress_kv_tensor(layer_kv[1], c, d)
                new_past.append((ck, cv))
            past = new_past
            logits = out.logits[:, -1, :]
            target = input_ids[:, t+1]
            loss = torch.nn.functional.cross_entropy(logits, target)
            total_loss += loss.item()
        avg_loss = total_loss / (seq_len - 1)
        losses_both.append(avg_loss)

ppl_both = np.exp(np.mean(losses_both))

# Generation with both compressed
inputs_gen = tokenizer("Once upon a time", return_tensors="pt").to(model.device)
past = None
generated = inputs_gen["input_ids"]
for _ in range(50):
    out = model(generated[:, -1:] if past else generated, past_key_values=past, use_cache=True)
    new_past = []
    for layer_kv in out.past_key_values:
        ck = compress_kv_tensor(layer_kv[0], c, d)
        cv = compress_kv_tensor(layer_kv[1], c, d)
        new_past.append((ck, cv))
    past = new_past
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_tok], dim=1)
gen_text = tokenizer.decode(generated[0], skip_special_tokens=True)

print(f"Both PPL: {ppl_both:.4f}")
print(f"Gen: {gen_text[:200]}")

# Restore
for name, orig in saved_w2.items():
    parts = name.split('.')
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    getattr(obj, parts[-1]).data = orig

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 65)
print("SUMMARY: [115-130] 4-bit LUT (KVFloat12, 25% compression)")
print("=" * 65)
print(f"{'Config':<30} | {'PPL':>10}")
print("-" * 45)
print(f"{'BF16 baseline':<30} | {orig_ppl:>10.4f}")
print(f"{'Weights only':<30} | {ppl_w:>10.4f}")
print(f"{'KV cache only':<30} | {ppl_kv:>10.4f}")
print(f"{'Weights + KV cache':<30} | {ppl_both:>10.4f}")
