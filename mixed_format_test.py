"""
1) Verify KVFloat12 Option-D encode/decode with 5-bit LUT (should be ~100% like KVFloat13)
2) Mixed: weights=KVFloat12 (4-bit, split LUT), KV cache=KVFloat13 (5-bit)
"""
import torch, numpy as np, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
sys.path.insert(0, '/root/kvfloat13')
from task3_encode_decode import encode_kvf13, decode_kvf13
from split_lut_kvfloat12 import encode_kvf12, decode_kvf12, build_lut_from_counter, extract_exp8, coverage

model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")

target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

# Collect counters
w_counter = Counter()
for name, param in model.named_parameters():
    if any(name.endswith(s) for s in target_suffixes):
        w_counter.update(extract_exp8(param.data.to(torch.bfloat16)).tolist())

# ============================================================
# Step 1: Sanity check — run KVFloat12 Option-D codec with 5-bit (32-entry) LUT
# If this still explodes PPL, the bug is in Option-D codec
# ============================================================
print("=" * 65)
print("STEP 1: KVFloat12 Option-D codec with 5-BIT LUT (32 entries)")
print("         (sanity check — should match KVFloat13 results)")
print("=" * 65)

w_compress32, w_decompress32 = build_lut_from_counter(w_counter, 32)
print(f"32-entry weight LUT: {sorted(w_decompress32.tolist())}")
print(f"Coverage: {coverage(w_counter, w_decompress32.tolist()):.6f}%")

# Encode/decode all weights with Option-D codec but 32-entry LUT
# Need to adapt — Option-D uses 4-bit exp field, won't fit 5-bit index!
# So this test doesn't make sense for Option-D. Let's just retest KVFloat13 for PPL.

# Actually, let's just verify: is the PPL issue from the 14K mismatches,
# or from a codec bug? Simplest test: use KVFloat13 codec on weights, check PPL.
print("\nRe-verifying KVFloat13 (5-bit) on weights → PPL...")

compress5 = np.load('/root/kvfloat13/compress_lut.npy')
decompress5 = np.load('/root/kvfloat13/decompress_lut.npy')

def compress_weights_and_test(model, tokenizer, compress_lut, decompress_lut, encode_fn, decode_fn, label):
    """Compress all linear weights, measure PPL, restore."""
    test_texts = [
        "The meaning of life is a question that has puzzled philosophers for centuries.",
        "In machine learning, neural networks are trained using backpropagation algorithms.",
        "The quick brown fox jumps over the lazy dog in the warm summer afternoon.",
        "Python is a high-level programming language known for its simplicity and readability.",
        "Climate change is one of the most pressing challenges facing humanity today.",
    ]

    # Original PPL
    model.eval()
    orig_losses = []
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            orig_losses.append(outputs.loss.item())
    orig_ppl = np.exp(np.mean(orig_losses))

    # Compress and swap weights
    saved = {}
    total_vals = 0
    total_match = 0
    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        bf16 = param.data.to(torch.bfloat16)
        shape = bf16.shape
        raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
        n = len(raw)
        pad = (128 - n % 128) % 128
        raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw

        encoded = encode_fn(raw_p, compress_lut)
        decoded = decode_fn(*encoded, decompress_lut)[:n]

        total_vals += n
        total_match += np.sum(raw == decoded)

        dec_t = torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(param.device)
        saved[name] = param.data.clone()
        param.data = dec_t

    match_pct = 100.0 * total_match / total_vals
    print(f"  [{label}] Bit-exact: {total_match:,}/{total_vals:,} ({match_pct:.6f}%)")

    comp_losses = []
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            comp_losses.append(outputs.loss.item())
    comp_ppl = np.exp(np.mean(comp_losses))

    # Generation
    prompt_text = "Once upon a time"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    gen_text = tokenizer.decode(gen[0], skip_special_tokens=True)

    print(f"  [{label}] PPL: {orig_ppl:.4f} → {comp_ppl:.4f} (Δ{comp_ppl-orig_ppl:+.4f}, {100*(comp_ppl/orig_ppl-1):+.4f}%)")
    print(f"  [{label}] Gen: {gen_text[:150]}")

    # Restore
    for name, orig_data in saved.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        getattr(obj, parts[-1]).data = orig_data

    return comp_ppl, match_pct

# Test 1: KVFloat13 (5-bit, should be perfect)
def encode13(data, clut):
    return encode_kvf13(data, clut)
def decode13(s, e, em, dlut):
    return decode_kvf13(s, e, em, dlut)

ppl13, match13 = compress_weights_and_test(
    model, tokenizer, compress5, decompress5, encode13, decode13, "KVFloat13-5bit"
)

# Test 2: KVFloat12 (4-bit, weight-specific LUT)
print()
w_compress4, w_decompress4 = build_lut_from_counter(w_counter, 16)
print(f"16-entry weight LUT: {sorted(w_decompress4.tolist())}")

def encode12(data, clut):
    return encode_kvf12(data, clut)
def decode12(s1, s2, dlut):
    return decode_kvf12(s1, s2, dlut)

ppl12, match12 = compress_weights_and_test(
    model, tokenizer, w_compress4, w_decompress4, encode12, decode12, "KVFloat12-4bit"
)

# ============================================================
# Step 2: Analyze the 14K mismatches more carefully
# ============================================================
print("\n" + "=" * 65)
print("STEP 2: MISMATCH ANALYSIS — WHERE ARE THE BROKEN VALUES?")
print("=" * 65)

for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    bf16 = param.data.to(torch.bfloat16)
    raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    n = len(raw)
    pad = (128 - n % 128) % 128
    raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw

    s1, s2 = encode_kvf12(raw_p, w_compress4)
    decoded = decode_kvf12(s1, s2, w_decompress4)[:n]

    mismask = raw != decoded
    nm = np.sum(mismask)
    if nm == 0:
        continue

    # What exponents are mismatched?
    orig_exp = (raw[mismask] >> 7) & 0xFF
    dec_exp_raw = (decoded[mismask] >> 7) & 0xFF

    # Actual float values
    orig_f = torch.from_numpy(raw[mismask].astype(np.int16)).view(torch.bfloat16).float().numpy()
    dec_f = torch.from_numpy(decoded[mismask].astype(np.int16)).view(torch.bfloat16).float().numpy()

    layer = name.split('.')[2] if 'layers' in name else '?'
    tname = name.rsplit('.', 1)[0].rsplit('.', 1)[-1]

    # Only print first few layers to keep output manageable
    if int(layer) > 2:
        continue

    print(f"\nL{layer}.{tname}: {nm} mismatches")
    exp_counts = Counter(orig_exp.tolist())
    for exp_val, cnt in exp_counts.most_common(10):
        dec_exps = dec_exp_raw[orig_exp == exp_val]
        print(f"  exp={exp_val} → mapped to exp={dec_exps[0]}: "
              f"{cnt} values, ratio=2^{dec_exps[0]-exp_val:+d} = {2.0**(dec_exps[0]-exp_val):.4f}x")

    # Show actual value impact
    abs_err = np.abs(orig_f - dec_f)
    rel_err = abs_err / (np.abs(orig_f) + 1e-30)
    print(f"  |error|: mean={abs_err.mean():.6e}, max={abs_err.max():.6e}")
    print(f"  |rel_err|: mean={rel_err.mean():.4f}, max={rel_err.max():.4f}")

# ============================================================
# Step 3: What if we use per-layer 4-bit LUT?
# ============================================================
print("\n" + "=" * 65)
print("STEP 3: PER-LAYER 4-BIT LUT (each layer gets its own 16-entry LUT)")
print("=" * 65)

layer_counters = {}
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    parts = name.split('.')
    layer_num = int(parts[2]) if 'layers' in name else -1
    if layer_num not in layer_counters:
        layer_counters[layer_num] = Counter()
    layer_counters[layer_num].update(extract_exp8(param.data.to(torch.bfloat16)).tolist())

total_vals = 0
total_match = 0
for layer_num in sorted(layer_counters.keys()):
    lc, ld = build_lut_from_counter(layer_counters[layer_num], 16)
    cov = coverage(layer_counters[layer_num], ld.tolist())

    # Encode/decode this layer's tensors
    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        ln = int(name.split('.')[2]) if 'layers' in name else -1
        if ln != layer_num:
            continue

        raw = param.data.to(torch.bfloat16).contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
        n = len(raw)
        pad = (128 - n % 128) % 128
        raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw

        s1, s2 = encode_kvf12(raw_p, lc)
        decoded = decode_kvf12(s1, s2, ld)[:n]

        m = np.sum(raw == decoded)
        total_vals += n
        total_match += m

    if layer_num % 10 == 0 or layer_num == 29:
        print(f"  Layer {layer_num:>2}: LUT={sorted(ld.tolist())} cov={cov:.4f}%")

per_layer_match = 100.0 * total_match / total_vals
print(f"\nPer-layer 4-bit LUT total bit-exact: {total_match:,}/{total_vals:,} ({per_layer_match:.6f}%)")
print(f"Mismatches: {total_vals - total_match:,}")
print(f"(vs global 4-bit: {match12:.6f}%)")

# Per-layer LUT PPL test
print("\nPer-layer 4-bit LUT → PPL test...")
saved = {}
for layer_num in sorted(layer_counters.keys()):
    lc, ld = build_lut_from_counter(layer_counters[layer_num], 16)
    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        ln = int(name.split('.')[2]) if 'layers' in name else -1
        if ln != layer_num:
            continue

        bf16 = param.data.to(torch.bfloat16)
        shape = bf16.shape
        raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
        n = len(raw)
        pad = (128 - n % 128) % 128
        raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw

        s1, s2 = encode_kvf12(raw_p, lc)
        decoded = decode_kvf12(s1, s2, ld)[:n]

        dec_t = torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(param.device)
        saved[name] = param.data.clone()
        param.data = dec_t

test_texts = [
    "The meaning of life is a question that has puzzled philosophers for centuries.",
    "In machine learning, neural networks are trained using backpropagation algorithms.",
    "The quick brown fox jumps over the lazy dog in the warm summer afternoon.",
    "Python is a high-level programming language known for its simplicity and readability.",
    "Climate change is one of the most pressing challenges facing humanity today.",
]
model.eval()
losses = []
with torch.no_grad():
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())
ppl_perlayer = np.exp(np.mean(losses))

inputs = tokenizer("Once upon a time", return_tensors="pt").to(model.device)
with torch.no_grad():
    gen = model.generate(**inputs, max_new_tokens=50, do_sample=False)
gen_text = tokenizer.decode(gen[0], skip_special_tokens=True)

print(f"  Per-layer 4-bit PPL: {ppl_perlayer:.4f}")
print(f"  Gen: {gen_text[:150]}")

# Restore
for name, orig_data in saved.items():
    parts = name.split('.')
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    getattr(obj, parts[-1]).data = orig_data

# ============================================================
# Final summary
# ============================================================
print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)
print(f"{'Method':<30} {'Bit-exact':>12} {'PPL':>10} {'Compress':>10}")
print("-" * 65)
print(f"{'BF16 (baseline)':<30} {'100%':>12} {'6.6122':>10} {'0%':>10}")
print(f"{'KVFloat13 (5-bit global)':<30} {f'{match13:.4f}%':>12} {f'{ppl13:.4f}':>10} {'18.75%':>10}")
print(f"{'KVFloat12 (4-bit global W)':<30} {f'{match12:.4f}%':>12} {f'{ppl12:.4f}':>10} {'25.00%':>10}")
print(f"{'KVFloat12 (4-bit per-layer)':<30} {f'{per_layer_match:.4f}%':>12} {f'{ppl_perlayer:.4f}':>10} {'25.00%':>10}")
