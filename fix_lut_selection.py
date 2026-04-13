"""
Fix: LUT selection should minimize reconstruction error, not maximize count.
Try different 16-wide windows and smarter selection strategies.
"""
import torch, numpy as np, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
sys.path.insert(0, '/root/kvfloat13')
from split_lut_kvfloat12 import encode_kvf12, decode_kvf12, extract_exp8

model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")

target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

# Collect (exponent, absolute_value) pairs to compute weighted error
w_counter = Counter()  # exp -> count
w_weighted = Counter()  # exp -> sum of |value|

all_raw = []
for name, param in model.named_parameters():
    if any(name.endswith(s) for s in target_suffixes):
        bf16 = param.data.to(torch.bfloat16)
        raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
        exps = ((raw >> 7) & 0xFF).astype(int)
        vals_abs = np.abs(bf16.contiguous().view(-1).float().cpu().numpy())

        for e in np.unique(exps):
            mask = exps == e
            w_counter[e] += int(mask.sum())
            w_weighted[e] += float(vals_abs[mask].sum())
        all_raw.append(raw)

all_raw = np.concatenate(all_raw)
total_vals = len(all_raw)

print("Exponent stats (sorted by weighted impact):")
print(f"{'exp':>4} | {'count':>10} | {'count%':>8} | {'sum|val|':>12} | {'val%':>8} | {'mean|val|':>10}")
print("-" * 65)
total_weight = sum(w_weighted.values())
for e in sorted(w_weighted.keys()):
    c = w_counter[e]
    w = w_weighted[e]
    print(f"{e:>4} | {c:>10,} | {100*c/total_vals:>7.4f}% | {w:>12.2f} | {100*w/total_weight:>7.4f}% | {w/c:>10.6f}")

def build_lut_contiguous(base_exp, width=16):
    """Build LUT from contiguous range [base_exp, base_exp+width-1]."""
    exps = list(range(base_exp, base_exp + width))
    decompress = np.array(exps, dtype=np.uint8)
    compress = np.zeros(256, dtype=np.uint8)
    exp_set = set(exps)
    for i, e in enumerate(exps):
        compress[e] = i
    for e in range(256):
        if e in exp_set:
            continue
        # nearest
        best_i, best_d = 0, float('inf')
        for se in exps:
            d = abs(2.0**(e-127) - 2.0**(se-127))
            if d < best_d:
                best_d = d
                best_i = exps.index(se)
        compress[e] = best_i
    return compress, decompress

def test_lut(compress, decompress, label):
    """Compress all weights, measure PPL."""
    saved = {}
    total_match = 0
    total_n = 0
    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        bf16 = param.data.to(torch.bfloat16)
        shape = bf16.shape
        raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
        n = len(raw)
        pad = (128 - n % 128) % 128
        raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw

        s1, s2 = encode_kvf12(raw_p, compress)
        decoded = decode_kvf12(s1, s2, decompress)[:n]
        total_match += np.sum(raw == decoded)
        total_n += n

        dec_t = torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(param.device)
        saved[name] = param.data.clone()
        param.data = dec_t

    match_pct = 100.0 * total_match / total_n

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
    ppl = np.exp(np.mean(losses))

    inputs = tokenizer("Once upon a time", return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    gen_text = tokenizer.decode(gen[0], skip_special_tokens=True)

    # Restore
    for name, orig_data in saved.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        getattr(obj, parts[-1]).data = orig_data

    print(f"  [{label}] LUT={sorted(decompress.tolist())}")
    print(f"  [{label}] match={match_pct:.4f}%  PPL={ppl:.4f}")
    print(f"  [{label}] Gen: {gen_text[:120]}")
    return ppl, match_pct

# ============================================================
# Try different contiguous windows
# ============================================================
print("\n" + "=" * 65)
print("SLIDING WINDOW: trying all 16-wide contiguous ranges")
print("=" * 65)

# Baseline
test_texts = [
    "The meaning of life is a question that has puzzled philosophers for centuries.",
    "In machine learning, neural networks are trained using backpropagation algorithms.",
    "The quick brown fox jumps over the lazy dog in the warm summer afternoon.",
    "Python is a high-level programming language known for its simplicity and readability.",
    "Climate change is one of the most pressing challenges facing humanity today.",
]
model.eval()
orig_losses = []
with torch.no_grad():
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        orig_losses.append(outputs.loss.item())
orig_ppl = np.exp(np.mean(orig_losses))
print(f"  Baseline PPL: {orig_ppl:.4f}\n")

results = []
for base in range(110, 120):
    c, d = build_lut_contiguous(base, 16)
    ppl, match = test_lut(c, d, f"[{base}-{base+15}]")
    results.append((base, ppl, match))
    print()

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"{'Window':<12} | {'PPL':>10} | {'Bit-exact':>10}")
print("-" * 38)
print(f"{'BF16':<12} | {orig_ppl:>10.4f} | {'100%':>10}")
for base, ppl, match in results:
    print(f"[{base}-{base+15}]   | {ppl:>10.4f} | {match:>9.4f}%")
