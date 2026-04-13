"""
KVFloat12: Split LUT approach
- Weight LUT: 16 entries optimized for weights
- KV cache LUT: 16 entries optimized for KV cache
- 4-bit exponent → 12 bits total → 25% compression
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import sys
sys.path.insert(0, '/root/kvfloat13')

def extract_exp8(t):
    return ((t.view(torch.int16).to(torch.int32) >> 7) & 0xFF).cpu().numpy().flatten()

def build_lut_from_counter(counter, k=16):
    """Build compress/decompress LUTs from frequency counter."""
    sorted_exps = sorted(counter.items(), key=lambda x: -x[1])
    top_k = sorted([int(e) for e, _ in sorted_exps[:k]])

    decompress = np.array(top_k, dtype=np.uint8)
    compress = np.zeros(256, dtype=np.uint8)

    exp_to_idx = {e: i for i, e in enumerate(top_k)}
    for e, i in exp_to_idx.items():
        compress[e] = i

    # Unsupported → nearest
    supported = set(top_k)
    for e in range(256):
        if e in supported:
            continue
        best_idx = 0
        best_dist = float('inf')
        for se in top_k:
            d = abs(2.0**(e-127) - 2.0**(se-127))
            if d < best_dist:
                best_dist = d
                best_idx = exp_to_idx[se]
        compress[e] = best_idx

    return compress, decompress

def coverage(counter, lut_exps):
    total = sum(counter.values())
    covered = sum(counter.get(e, 0) for e in lut_exps)
    return 100.0 * covered / total

# ============================================================
# Encode/Decode for KVFloat12 (4-bit exp, Option D layout)
# Stream 1: [sign(1)|exp4(4)|mant_hi(3)] = 1 byte per value
# Stream 2: mant_lo(4) nibbles = 0.5 byte per value
# Total: 1.5 bytes/value = 192 bytes/128 values
# ============================================================

def encode_kvf12(bf16_uint16, compress_lut):
    """Encode BF16 → KVFloat12 (Option D layout)."""
    assert bf16_uint16.dtype == np.uint16
    n = len(bf16_uint16)
    assert n % 128 == 0
    num_blocks = n // 128

    sign = ((bf16_uint16 >> 15) & 1).astype(np.uint8)
    exp8 = ((bf16_uint16 >> 7) & 0xFF).astype(np.uint8)
    mant7 = (bf16_uint16 & 0x7F).astype(np.uint8)

    exp4 = compress_lut[exp8]
    mant_hi3 = (mant7 >> 4) & 0x07
    mant_lo4 = mant7 & 0x0F

    # Stream 1: [sign(1)|exp4(4)|mant_hi(3)] byte
    stream1 = ((sign << 7) | (exp4 << 3) | mant_hi3).astype(np.uint8)
    stream1 = stream1.reshape(num_blocks, 128)

    # Stream 2: mant_lo4 nibbles, pack 2 per byte
    mant_lo_reshaped = mant_lo4.reshape(num_blocks, 128)
    even = mant_lo_reshaped[:, 0::2]
    odd = mant_lo_reshaped[:, 1::2]
    stream2 = (even | (odd << 4)).astype(np.uint8)  # (num_blocks, 64)

    return stream1, stream2

def decode_kvf12(stream1, stream2, decompress_lut):
    """Decode KVFloat12 → BF16 uint16."""
    num_blocks = stream1.shape[0]

    s1 = stream1.astype(np.uint16)
    sign = (s1 >> 7) & 1
    exp4 = (s1 >> 3) & 0xF
    mant_hi3 = s1 & 0x7

    # Unpack mant_lo nibbles
    even = (stream2 & 0x0F).astype(np.uint16)
    odd = ((stream2 >> 4) & 0x0F).astype(np.uint16)
    mant_lo4 = np.empty((num_blocks, 128), dtype=np.uint16)
    mant_lo4[:, 0::2] = even
    mant_lo4[:, 1::2] = odd

    mant7 = (mant_hi3 << 4) | mant_lo4

    exp8 = decompress_lut[exp4.astype(np.uint8)].astype(np.uint16)

    bf16 = (sign << 15) | (exp8 << 7) | mant7
    return bf16.reshape(-1).astype(np.uint16)

# ============================================================

def main():
    model_name = "HuggingFaceTB/SmolLM-135M"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")

    target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                       'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

    # Collect weight exponents
    w_counter = Counter()
    for name, param in model.named_parameters():
        if any(name.endswith(s) for s in target_suffixes):
            w_counter.update(extract_exp8(param.data.to(torch.bfloat16)).tolist())

    # Collect KV cache exponents
    kv_counter = Counter()
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "In machine learning, neural networks are trained using backpropagation.",
        "The stock market experienced significant volatility as investors reacted.",
        "Once upon a time in a distant kingdom there lived a wise old wizard.",
    ]
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model(**inputs, use_cache=True)
            for kv in outputs.past_key_values:
                kv_counter.update(extract_exp8(kv[0].to(torch.bfloat16)).tolist())
                kv_counter.update(extract_exp8(kv[1].to(torch.bfloat16)).tolist())

    # Build separate LUTs
    w_compress, w_decompress = build_lut_from_counter(w_counter, 16)
    kv_compress, kv_decompress = build_lut_from_counter(kv_counter, 16)

    print("\n" + "=" * 65)
    print("SPLIT LUT: WEIGHT LUT (16 entries)")
    print("=" * 65)
    w_cov = coverage(w_counter, w_decompress.tolist())
    print(f"Exponents: {sorted(w_decompress.tolist())}")
    print(f"Weight coverage: {w_cov:.4f}%")
    for i, e in enumerate(w_decompress):
        freq = w_counter.get(int(e), 0)
        pct = 100.0 * freq / sum(w_counter.values())
        print(f"  exp4={i:>2d} → exp8={e:>3d} (2^{e-127:>+4d})  {pct:.3f}%")

    print("\n" + "=" * 65)
    print("SPLIT LUT: KV CACHE LUT (16 entries)")
    print("=" * 65)
    kv_cov = coverage(kv_counter, kv_decompress.tolist())
    print(f"Exponents: {sorted(kv_decompress.tolist())}")
    print(f"KV cache coverage: {kv_cov:.4f}%")
    for i, e in enumerate(kv_decompress):
        freq = kv_counter.get(int(e), 0)
        pct = 100.0 * freq / sum(kv_counter.values())
        print(f"  exp4={i:>2d} → exp8={e:>3d} (2^{e-127:>+4d})  {pct:.3f}%")

    # Cross-check: what happens if we accidentally use the wrong LUT?
    print("\n" + "=" * 65)
    print("CROSS-CHECK: WRONG LUT USAGE")
    print("=" * 65)
    w_with_kv_lut = coverage(w_counter, kv_decompress.tolist())
    kv_with_w_lut = coverage(kv_counter, w_decompress.tolist())
    print(f"Weight with KV LUT:   {w_with_kv_lut:.4f}%")
    print(f"KV with Weight LUT:   {kv_with_w_lut:.4f}%")

    # ============================================================
    # Verify on real weights
    # ============================================================
    print("\n" + "=" * 65)
    print("WEIGHT ENCODE/DECODE VERIFICATION (KVFloat12)")
    print("=" * 65)

    total_vals = 0
    total_match = 0
    per_tensor_results = []

    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue

        bf16 = param.data.to(torch.bfloat16)
        raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
        orig_len = len(raw)

        pad_len = (128 - orig_len % 128) % 128
        if pad_len:
            raw_padded = np.concatenate([raw, np.zeros(pad_len, dtype=np.uint16)])
        else:
            raw_padded = raw

        s1, s2 = encode_kvf12(raw_padded, w_compress)
        decoded = decode_kvf12(s1, s2, w_decompress)[:orig_len]

        match = np.sum(raw == decoded)
        total_vals += orig_len
        total_match += match

        if match < orig_len:
            per_tensor_results.append((name, orig_len, match))

    w_match_pct = 100.0 * total_match / total_vals
    print(f"Total: {total_match:,}/{total_vals:,} bit-exact ({w_match_pct:.6f}%)")
    print(f"Mismatches: {total_vals - total_match:,}")
    if per_tensor_results:
        print(f"\nTensors with mismatches:")
        for name, tot, m in per_tensor_results[:20]:
            short = name.rsplit('.', 1)[0].rsplit('.', 1)[-1]
            layer = name.split('.')[2] if 'layers' in name else '?'
            print(f"  L{layer:>2}.{short:<12}: {m:,}/{tot:,} ({100*m/tot:.4f}%)")
    else:
        print("ALL TENSORS BIT-EXACT!")

    # ============================================================
    # Verify on KV cache
    # ============================================================
    print("\n" + "=" * 65)
    print("KV CACHE ENCODE/DECODE VERIFICATION (KVFloat12)")
    print("=" * 65)

    kv_total = 0
    kv_match = 0
    kv_mismatches_by_exp = Counter()

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model(**inputs, use_cache=True)
            for layer_idx, kv in enumerate(outputs.past_key_values):
                for tensor in [kv[0], kv[1]]:
                    bf16 = tensor.to(torch.bfloat16)
                    raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
                    orig_len = len(raw)

                    pad_len = (128 - orig_len % 128) % 128
                    if pad_len:
                        raw_padded = np.concatenate([raw, np.zeros(pad_len, dtype=np.uint16)])
                    else:
                        raw_padded = raw

                    s1, s2 = encode_kvf12(raw_padded, kv_compress)
                    decoded = decode_kvf12(s1, s2, kv_decompress)[:orig_len]

                    match = np.sum(raw == decoded)
                    kv_total += orig_len
                    kv_match += match

                    # Track which exponents cause mismatches
                    mismask = raw != decoded
                    if np.any(mismask):
                        bad_exps = ((raw[mismask] >> 7) & 0xFF).tolist()
                        kv_mismatches_by_exp.update(bad_exps)

    kv_match_pct = 100.0 * kv_match / kv_total
    print(f"Total: {kv_match:,}/{kv_total:,} bit-exact ({kv_match_pct:.6f}%)")
    print(f"Mismatches: {kv_total - kv_match:,}")
    if kv_mismatches_by_exp:
        print(f"\nMismatches by exponent:")
        for exp, cnt in kv_mismatches_by_exp.most_common(20):
            print(f"  exp={exp}: {cnt} values")

    # ============================================================
    # Perplexity comparison
    # ============================================================
    print("\n" + "=" * 65)
    print("PERPLEXITY: BF16 vs KVFloat12 (split LUT)")
    print("=" * 65)

    test_texts = [
        "The meaning of life is a question that has puzzled philosophers for centuries.",
        "In machine learning, neural networks are trained using backpropagation algorithms.",
        "The quick brown fox jumps over the lazy dog in the warm summer afternoon.",
        "Python is a high-level programming language known for its simplicity and readability.",
        "Climate change is one of the most pressing challenges facing humanity today.",
    ]

    # Original PPL
    orig_losses = []
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            orig_losses.append(outputs.loss.item())
    orig_ppl = np.exp(np.mean(orig_losses))

    # Compress weights
    compressed_state = {}
    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue

        bf16_data = param.data.to(torch.bfloat16)
        shape = bf16_data.shape
        raw = bf16_data.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
        orig_len = len(raw)
        pad_len = (128 - orig_len % 128) % 128
        raw_padded = np.concatenate([raw, np.zeros(pad_len, dtype=np.uint16)]) if pad_len else raw

        s1, s2 = encode_kvf12(raw_padded, w_compress)
        decoded = decode_kvf12(s1, s2, w_decompress)[:orig_len]

        decoded_t = torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(param.device)
        compressed_state[name] = (param.data.clone(), decoded_t)

    # Apply and measure
    for name, (orig, comp) in compressed_state.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], torch.nn.Parameter(comp))

    comp_losses = []
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            comp_losses.append(outputs.loss.item())
    comp_ppl = np.exp(np.mean(comp_losses))

    print(f"Original PPL:   {orig_ppl:.4f}")
    print(f"KVFloat12 PPL:  {comp_ppl:.4f}")
    print(f"Delta:          {comp_ppl - orig_ppl:+.4f} ({100*(comp_ppl/orig_ppl-1):+.4f}%)")

    # Generation comparison
    for name, (orig, _) in compressed_state.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], torch.nn.Parameter(orig))

    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        orig_out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    orig_text = tokenizer.decode(orig_out[0], skip_special_tokens=True)

    for name, (_, comp) in compressed_state.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], torch.nn.Parameter(comp))

    with torch.no_grad():
        comp_out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    comp_text = tokenizer.decode(comp_out[0], skip_special_tokens=True)

    print(f"\nOriginal:   {orig_text[:200]}")
    print(f"KVFloat12:  {comp_text[:200]}")
    print(f"Identical:  {orig_text == comp_text}")

    # Restore
    for name, (orig, _) in compressed_state.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], torch.nn.Parameter(orig))

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 65)
    print("SUMMARY: KVFloat12 with Split LUT")
    print("=" * 65)
    print(f"""
Format:       sign(1) + exp(4) + mantissa(7) = 12 bits
Compression:  25.00% vs BF16
Layout:       Option D — [sign|exp4|mant_hi3] byte + mant_lo4 nibble
              = 192 bytes per 128 values

Weight LUT:   {sorted(w_decompress.tolist())}
              Coverage: {w_cov:.4f}%
              Bit-exact: {w_match_pct:.6f}%

KV cache LUT: {sorted(kv_decompress.tolist())}
              Coverage: {kv_cov:.4f}%
              Bit-exact: {kv_match_pct:.6f}%

Perplexity:   {orig_ppl:.4f} → {comp_ppl:.4f} ({comp_ppl-orig_ppl:+.4f})

vs KVFloat13: +6.25% more compression, single LUT → split LUT
""")

    # Save LUTs
    np.save('/root/kvfloat13/w_compress_lut4.npy', w_compress)
    np.save('/root/kvfloat13/w_decompress_lut4.npy', w_decompress)
    np.save('/root/kvfloat13/kv_compress_lut4.npy', kv_compress)
    np.save('/root/kvfloat13/kv_decompress_lut4.npy', kv_decompress)

if __name__ == "__main__":
    main()
