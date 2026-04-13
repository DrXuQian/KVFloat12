"""
Task 4: Correctness and Error Analysis for KVFloat13
Tests encode/decode on real model weights and KV cache, measures perplexity impact.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import sys
sys.path.insert(0, '/root/kvfloat13')
from task3_encode_decode import encode_kvf13, decode_kvf13

def verify_tensor(name, bf16_tensor, compress_lut, decompress_lut):
    """Verify encode→decode round-trip on a BF16 tensor."""
    raw = bf16_tensor.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)

    # Pad to multiple of 128
    orig_len = len(raw)
    pad_len = (128 - orig_len % 128) % 128
    if pad_len > 0:
        raw = np.concatenate([raw, np.zeros(pad_len, dtype=np.uint16)])

    # Encode → Decode
    s, e, em = encode_kvf13(raw, compress_lut)
    decoded = decode_kvf13(s, e, em, decompress_lut)

    # Compare (only original, not padding)
    raw = raw[:orig_len]
    decoded = decoded[:orig_len]

    exact_match = np.sum(raw == decoded)
    total = orig_len
    match_rate = 100.0 * exact_match / total

    # Analyze mismatches
    mismatch_mask = raw != decoded
    num_mismatch = np.sum(mismatch_mask)

    rel_errors = []
    mismatch_exps = []
    if num_mismatch > 0:
        orig_f32 = torch.from_numpy(raw[mismatch_mask].astype(np.int16)).view(torch.bfloat16).float().numpy()
        dec_f32 = torch.from_numpy(decoded[mismatch_mask].astype(np.int16)).view(torch.bfloat16).float().numpy()

        nonzero = np.abs(orig_f32) > 0
        if np.any(nonzero):
            rel_err = np.abs((orig_f32[nonzero] - dec_f32[nonzero]) / orig_f32[nonzero])
            rel_errors = rel_err.tolist()

        orig_exp8 = (raw[mismatch_mask] >> 7) & 0xFF
        mismatch_exps = orig_exp8.tolist()

    return {
        'name': name,
        'total': total,
        'exact_match': exact_match,
        'match_rate': match_rate,
        'num_mismatch': num_mismatch,
        'rel_errors': rel_errors,
        'mismatch_exps': mismatch_exps,
    }


def analyze_results(results):
    """Print summary of verification results."""
    print("=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)

    total_all = sum(r['total'] for r in results)
    match_all = sum(r['exact_match'] for r in results)
    mismatch_all = sum(r['num_mismatch'] for r in results)

    print(f"\nOverall: {match_all:,}/{total_all:,} bit-exact ({100*match_all/total_all:.6f}%)")
    print(f"Mismatches: {mismatch_all:,} ({100*mismatch_all/total_all:.6f}%)")

    if mismatch_all > 0:
        all_rel_errors = []
        all_mismatch_exps = Counter()
        for r in results:
            all_rel_errors.extend(r['rel_errors'])
            all_mismatch_exps.update(r['mismatch_exps'])

        if all_rel_errors:
            arr = np.array(all_rel_errors)
            print(f"\nRelative error distribution (mismatched values only):")
            print(f"  Min:    {arr.min():.6f}")
            print(f"  Mean:   {arr.mean():.6f}")
            print(f"  Median: {np.median(arr):.6f}")
            print(f"  Max:    {arr.max():.6f}")
            print(f"  P99:    {np.percentile(arr, 99):.6f}")

        print(f"\nMismatched exponents:")
        for exp, count in all_mismatch_exps.most_common(20):
            print(f"  exp={exp}: {count} values")
    else:
        print("\n✓ FULLY LOSSLESS — zero mismatches across all tensors!")

    # Per-tensor breakdown (show only non-100% or first/last few)
    print(f"\n{'Tensor':<30} | {'Total':>12} | {'Match%':>10} | {'Mismatches':>10}")
    print("-" * 70)
    imperfect = [r for r in results if r['match_rate'] < 100.0]
    if imperfect:
        for r in imperfect:
            print(f"{r['name']:<30} | {r['total']:>12,} | {r['match_rate']:>9.4f}% | {r['num_mismatch']:>10,}")
    else:
        # Just show a few representative ones
        for r in results[:5]:
            print(f"{r['name']:<30} | {r['total']:>12,} | {r['match_rate']:>9.4f}% | {r['num_mismatch']:>10,}")
        if len(results) > 5:
            print(f"  ... ({len(results) - 5} more tensors, all 100.0000%)")


def perplexity_comparison(model, tokenizer, compress_lut, decompress_lut):
    """Compare perplexity between original and KVFloat13-compressed weights."""
    print("\n" + "=" * 70)
    print("END-TO-END PERPLEXITY COMPARISON")
    print("=" * 70)

    test_texts = [
        "The meaning of life is a question that has puzzled philosophers for centuries.",
        "In machine learning, neural networks are trained using backpropagation algorithms.",
        "The quick brown fox jumps over the lazy dog in the warm summer afternoon.",
        "Python is a high-level programming language known for its simplicity and readability.",
        "Climate change is one of the most pressing challenges facing humanity today.",
    ]

    model.eval()

    # Measure original perplexity
    orig_losses = []
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            orig_losses.append(outputs.loss.item())

    orig_ppl = np.exp(np.mean(orig_losses))
    print(f"Original perplexity: {orig_ppl:.4f}")

    # Compress all linear weights
    compressed_state = {}
    target_suffixes = ['q_proj.weight', 'k_proj.weight', 'v_proj.weight',
                       'o_proj.weight', 'gate_proj.weight', 'up_proj.weight',
                       'down_proj.weight']

    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue

        bf16_data = param.data.to(torch.bfloat16)
        shape = bf16_data.shape
        raw = bf16_data.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)

        orig_len = len(raw)
        pad_len = (128 - orig_len % 128) % 128
        if pad_len > 0:
            raw_padded = np.concatenate([raw, np.zeros(pad_len, dtype=np.uint16)])
        else:
            raw_padded = raw

        s, e, em = encode_kvf13(raw_padded, compress_lut)
        decoded = decode_kvf13(s, e, em, decompress_lut)[:orig_len]

        # Convert back to BF16 tensor
        decoded_tensor = torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(param.device)
        compressed_state[name] = (param.data.clone(), decoded_tensor)

    # Apply compressed weights
    for name, (orig, compressed) in compressed_state.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], torch.nn.Parameter(compressed))

    # Measure compressed perplexity
    comp_losses = []
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            comp_losses.append(outputs.loss.item())

    comp_ppl = np.exp(np.mean(comp_losses))
    print(f"KVFloat13 perplexity: {comp_ppl:.4f}")
    print(f"Delta: {comp_ppl - orig_ppl:+.4f} ({100*(comp_ppl/orig_ppl - 1):+.4f}%)")

    # Restore original weights
    for name, (orig, _) in compressed_state.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], torch.nn.Parameter(orig))

    # Generation quality comparison
    print("\n--- Generation Quality ---")
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        orig_out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    orig_text = tokenizer.decode(orig_out[0], skip_special_tokens=True)

    # Apply compressed weights for generation
    for name, (_, compressed) in compressed_state.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], torch.nn.Parameter(compressed))

    with torch.no_grad():
        comp_out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    comp_text = tokenizer.decode(comp_out[0], skip_special_tokens=True)

    print(f"Original:   {orig_text[:200]}")
    print(f"Compressed: {comp_text[:200]}")
    print(f"Identical output: {orig_text == comp_text}")

    # Restore
    for name, (orig, _) in compressed_state.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], torch.nn.Parameter(orig))


def main():
    compress_lut = np.load('/root/kvfloat13/compress_lut.npy')
    decompress_lut = np.load('/root/kvfloat13/decompress_lut.npy')

    model_name = "HuggingFaceTB/SmolLM-135M"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )

    # A) Verify all weight tensors
    print("\nVerifying weight tensors...")
    results = []
    target_suffixes = ['q_proj.weight', 'k_proj.weight', 'v_proj.weight',
                       'o_proj.weight', 'gate_proj.weight', 'up_proj.weight',
                       'down_proj.weight']

    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        bf16 = param.data.to(torch.bfloat16)
        result = verify_tensor(name, bf16, compress_lut, decompress_lut)
        results.append(result)

    # B) Verify KV cache
    print("Verifying KV cache tensors...")
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "In the year 2025, artificial intelligence has transformed every aspect of daily life.",
    ]

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model(**inputs, use_cache=True)
            for layer_idx, kv in enumerate(outputs.past_key_values):
                for kv_name, tensor in [("key", kv[0]), ("value", kv[1])]:
                    bf16 = tensor.to(torch.bfloat16)
                    result = verify_tensor(
                        f"layer{layer_idx}_{kv_name}_cache",
                        bf16, compress_lut, decompress_lut
                    )
                    results.append(result)

    analyze_results(results)

    # E) End-to-end perplexity
    perplexity_comparison(model, tokenizer, compress_lut, decompress_lut)

if __name__ == "__main__":
    main()
