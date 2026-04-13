"""
Task 1: Exponent Distribution Analysis for KVFloat13
Analyzes BF16 exponent distributions in model weights and KV cache.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import sys

def extract_exp8(bf16_tensor):
    """Extract 8-bit exponents from BF16 tensor."""
    # Convert to uint16 view
    raw = bf16_tensor.view(torch.int16).to(torch.int32)
    exp8 = (raw >> 7) & 0xFF
    return exp8

def top_k_coverage(exp_counts, k=32):
    """Return coverage % of top-k most frequent exponents."""
    total = sum(exp_counts.values())
    if total == 0:
        return 0.0, []
    top_k = exp_counts.most_common(k)
    covered = sum(c for _, c in top_k)
    return 100.0 * covered / total, [e for e, _ in top_k]

def analyze_weights(model):
    """Analyze exponent distribution across all weight tensors."""
    print("=" * 60)
    print("WEIGHT EXPONENT ANALYSIS")
    print("=" * 60)

    global_counter = Counter()
    layer_results = []

    target_suffixes = ['q_proj.weight', 'k_proj.weight', 'v_proj.weight',
                       'o_proj.weight', 'gate_proj.weight', 'up_proj.weight',
                       'down_proj.weight']

    for name, param in model.named_parameters():
        is_target = any(name.endswith(s) for s in target_suffixes)
        if not is_target:
            continue

        bf16_param = param.data.to(torch.bfloat16)
        exps = extract_exp8(bf16_param).cpu().numpy().flatten()

        counter = Counter(exps.tolist())
        global_counter.update(counter)

        unique = len(counter)
        cov, top_exps = top_k_coverage(counter, 32)

        # Parse layer number and tensor name
        parts = name.split('.')
        layer_num = None
        tensor_name = name.split('.')[-1].replace('.weight', '')
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                layer_num = int(parts[i + 1])
                break
        if layer_num is None:
            layer_num = -1
        tensor_name = name.rsplit('.', 1)[0].rsplit('.', 1)[-1]

        layer_results.append({
            'layer': layer_num, 'tensor': tensor_name,
            'unique': unique, 'coverage': cov
        })

    global_unique = len(global_counter)
    global_cov, global_top32 = top_k_coverage(global_counter, 32)

    coverages = [r['coverage'] for r in layer_results]

    print(f"Global unique exponents: {global_unique} / 256")
    print(f"Global top-32 coverage:  {global_cov:.4f}%")
    if coverages:
        print(f"Per-tensor coverage (min/max/mean): {min(coverages):.2f}% / {max(coverages):.2f}% / {np.mean(coverages):.2f}%")

    print(f"\n{'Layer':>5} | {'Tensor':<12} | {'Unique':>6} | {'Top-32 Coverage':>15}")
    print("-" * 50)
    for r in layer_results:
        print(f"{r['layer']:>5} | {r['tensor']:<12} | {r['unique']:>6} | {r['coverage']:>14.4f}%")

    print(f"\nGlobal top-32 exponents (sorted): {sorted(global_top32)}")

    # Show full distribution for reference
    print(f"\nFull exponent distribution (top 40):")
    for exp_val, count in global_counter.most_common(40):
        pct = 100.0 * count / sum(global_counter.values())
        print(f"  exp={exp_val:>3d} (2^{exp_val-127:>+4d}): {pct:>7.3f}%  count={count}")

    return global_counter, global_top32, layer_results

def analyze_kv_cache(model, tokenizer):
    """Analyze exponent distribution in KV cache during inference."""
    print("\n" + "=" * 60)
    print("KV CACHE EXPONENT ANALYSIS")
    print("=" * 60)

    # Sample prompts for diverse KV cache values
    prompts = [
        "The quick brown fox jumps over the lazy dog. In a world where technology advances rapidly,",
        "Mathematics is the language of the universe. Consider the equation E = mc^2, which tells us",
        "Once upon a time in a distant kingdom, there lived a wise old wizard who could predict",
        "The stock market experienced significant volatility today as investors reacted to the latest",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# This function",
    ]

    key_counter = Counter()
    value_counter = Counter()
    layer_results = []

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model(**inputs, use_cache=True)
            past_kv = outputs.past_key_values

            for layer_idx, kv in enumerate(past_kv):
                # kv is typically (key, value), each shape [batch, heads, seq, head_dim]
                key_tensor = kv[0].to(torch.bfloat16)
                val_tensor = kv[1].to(torch.bfloat16)

                key_exps = extract_exp8(key_tensor).cpu().numpy().flatten()
                val_exps = extract_exp8(val_tensor).cpu().numpy().flatten()

                kc = Counter(key_exps.tolist())
                vc = Counter(val_exps.tolist())
                key_counter.update(kc)
                value_counter.update(vc)

                # Per-layer stats (aggregate across heads for this prompt)
                k_unique = len(kc)
                k_cov, _ = top_k_coverage(kc, 32)
                v_unique = len(vc)
                v_cov, _ = top_k_coverage(vc, 32)

                layer_results.append({
                    'layer': layer_idx, 'kv': 'key',
                    'unique': k_unique, 'coverage': k_cov
                })
                layer_results.append({
                    'layer': layer_idx, 'kv': 'value',
                    'unique': v_unique, 'coverage': v_cov
                })

    # Aggregate per-layer across prompts
    from collections import defaultdict
    agg = defaultdict(list)
    for r in layer_results:
        agg[(r['layer'], r['kv'])].append(r['coverage'])

    print(f"Key global unique exponents:   {len(key_counter)} / 256")
    key_cov, key_top32 = top_k_coverage(key_counter, 32)
    print(f"Key global top-32 coverage:    {key_cov:.4f}%")

    print(f"Value global unique exponents: {len(value_counter)} / 256")
    val_cov, val_top32 = top_k_coverage(value_counter, 32)
    print(f"Value global top-32 coverage:  {val_cov:.4f}%")

    print(f"\n{'Layer':>5} | {'KV':<6} | {'Avg Unique':>10} | {'Avg Top-32 Cov':>15}")
    print("-" * 50)
    for (layer, kv), covs in sorted(agg.items()):
        avg_cov = np.mean(covs)
        print(f"{layer:>5} | {kv:<6} | {'':>10} | {avg_cov:>14.4f}%")

    print(f"\nKey top-32 exponents: {sorted(key_top32)}")
    print(f"Value top-32 exponents: {sorted(val_top32)}")

    # Full distribution
    all_kv = key_counter + value_counter
    print(f"\nKV cache full exponent distribution (top 40):")
    for exp_val, count in all_kv.most_common(40):
        pct = 100.0 * count / sum(all_kv.values())
        print(f"  exp={exp_val:>3d} (2^{exp_val-127:>+4d}): {pct:>7.3f}%  count={count}")

    return key_counter, value_counter, key_top32, val_top32

def unified_lut_analysis(weight_counter, kv_counter, weight_top32, key_top32, val_top32):
    """Check if a single global LUT can cover both weights and KV cache."""
    print("\n" + "=" * 60)
    print("UNIFIED LUT FEASIBILITY")
    print("=" * 60)

    combined = weight_counter + kv_counter
    combined_cov, combined_top32 = top_k_coverage(combined, 32)

    # Check weight coverage with combined top-32
    w_total = sum(weight_counter.values())
    w_covered = sum(weight_counter.get(e, 0) for e in combined_top32)
    w_cov = 100.0 * w_covered / w_total if w_total > 0 else 0

    kv_total = sum(kv_counter.values())
    kv_covered = sum(kv_counter.get(e, 0) for e in combined_top32)
    kv_cov = 100.0 * kv_covered / kv_total if kv_total > 0 else 0

    print(f"Combined top-32 coverage (overall): {combined_cov:.4f}%")
    print(f"  - Weights with unified LUT:       {w_cov:.4f}%")
    print(f"  - KV cache with unified LUT:      {kv_cov:.4f}%")
    print(f"  - Unified top-32 exponents:        {sorted(combined_top32)}")

    # Check overlap
    w_set = set(weight_top32)
    k_set = set(key_top32)
    v_set = set(val_top32)
    print(f"\nOverlap analysis:")
    print(f"  Weight ∩ Key top-32:   {len(w_set & k_set)}/32")
    print(f"  Weight ∩ Value top-32: {len(w_set & v_set)}/32")
    print(f"  Key ∩ Value top-32:    {len(k_set & v_set)}/32")
    print(f"  All three ∩:           {len(w_set & k_set & v_set)}/32")

    if combined_cov >= 99.5 and w_cov >= 99.5 and kv_cov >= 99.5:
        print(f"\n✓ Recommendation: GLOBAL LUT — single LUT covers all at ≥99.5%")
    elif w_cov >= 99.5 and kv_cov < 99.5:
        print(f"\n⚠ Recommendation: SEPARATE LUTs — KV cache needs its own LUT")
    else:
        print(f"\n⚠ Recommendation: PER-LAYER LUTs — coverage too low for global approach")

    return combined_top32

def main():
    model_name = "HuggingFaceTB/SmolLM-135M"
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture: {model.config.architectures}")
    print(f"Layers: {model.config.num_hidden_layers}, Heads: {model.config.num_attention_heads}")
    print(f"Hidden: {model.config.hidden_size}, Head dim: {model.config.hidden_size // model.config.num_attention_heads}")

    # Task 1A: Weight analysis
    weight_counter, weight_top32, weight_results = analyze_weights(model)

    # Task 1B: KV cache analysis
    key_counter, value_counter, key_top32, val_top32 = analyze_kv_cache(model, tokenizer)

    # Task 1C: Unified LUT feasibility
    kv_counter = key_counter + value_counter
    combined_top32 = unified_lut_analysis(weight_counter, kv_counter, weight_top32, key_top32, val_top32)

    # Save the combined top-32 for Task 2
    np.save('/root/kvfloat13/combined_top32.npy', np.array(sorted(combined_top32), dtype=np.uint8))

    # Save full frequency data
    all_counter = weight_counter + kv_counter
    freq_data = {int(k): int(v) for k, v in all_counter.items()}
    import json
    with open('/root/kvfloat13/exponent_frequencies.json', 'w') as f:
        json.dump(freq_data, f, indent=2)

    print("\n\nSaved: combined_top32.npy, exponent_frequencies.json")

if __name__ == "__main__":
    main()
