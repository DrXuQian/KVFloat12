"""
同一层、同一次推理中，不同 token position 的 KV cache 指数分布是否不同？
"""
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_exp8(t):
    return ((t.view(torch.int16).to(torch.int32) >> 7) & 0xFF).cpu().numpy().flatten()

model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
model.eval()

prompt = "The quick brown fox jumps over the lazy dog in a warm summer afternoon by the river"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
seq_len = len(tokens)
print(f"Prompt: {prompt}")
print(f"Tokens ({seq_len}): {tokens}\n")

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

# ============================================================
# Per-token, per-head exponent distribution
# ============================================================
for layer_idx in [0, 10, 20, 29]:
    for kv_name, kv_idx in [("key", 0), ("value", 1)]:
        tensor = outputs.past_key_values[layer_idx][kv_idx].to(torch.bfloat16)
        # [batch=1, heads, seq, head_dim=64]
        _, n_heads, s, d = tensor.shape

        print("=" * 80)
        print(f"Layer {layer_idx}, {kv_name}  (heads={n_heads}, seq={s}, head_dim={d})")
        print("=" * 80)

        # Show head 0 in detail: every token position
        hi = 0
        print(f"\n  Head {hi}, per-token breakdown:")
        print(f"  {'Pos':>3} | {'Token':<15} | {'Min':>4} | {'Max':>4} | {'Span':>4} | {'Uniq':>4} | {'Top-3 exponents'}")
        print("  " + "-" * 70)

        pos_ranges = []
        for si in range(s):
            vec = tensor[0, hi, si, :]  # [head_dim]
            exps = extract_exp8(vec)
            nonzero = exps[exps != 0]
            if len(nonzero) > 0:
                mn, mx = int(nonzero.min()), int(nonzero.max())
            else:
                mn, mx = 0, 0
            span = mx - mn
            unique = len(np.unique(exps))
            from collections import Counter
            top3 = Counter(exps.tolist()).most_common(3)
            top3_str = ", ".join(f"{e}({c})" for e, c in top3)

            tok = tokens[si] if si < len(tokens) else "?"
            print(f"  {si:>3} | {tok:<15} | {mn:>4} | {mx:>4} | {span:>4} | {unique:>4} | {top3_str}")
            pos_ranges.append((mn, mx, span))

        # Summary across all heads
        print(f"\n  All heads summary:")
        all_per_pos_spans = []
        for si in range(s):
            pos_spans = []
            for hi2 in range(n_heads):
                vec = tensor[0, hi2, si, :]
                exps = extract_exp8(vec)
                nonzero = exps[exps != 0]
                if len(nonzero) > 0:
                    pos_spans.append(int(nonzero.max()) - int(nonzero.min()))
                else:
                    pos_spans.append(0)
            all_per_pos_spans.append(pos_spans)

        # Per-position: range across heads
        print(f"  {'Pos':>3} | {'Token':<15} | {'Head spans (all heads)':>30}")
        print("  " + "-" * 55)
        for si in range(min(s, 20)):
            spans = all_per_pos_spans[si]
            tok = tokens[si] if si < len(tokens) else "?"
            spans_str = " ".join(f"{sp:>2}" for sp in spans)
            print(f"  {si:>3} | {tok:<15} | {spans_str}")

        print()

# ============================================================
# Key question: does position 0 (BOS/first token) differ from others?
# ============================================================
print("\n" + "=" * 80)
print("POSITION 0 vs OTHER POSITIONS (attention sink effect?)")
print("=" * 80)

for layer_idx in [0, 15, 29]:
    key_tensor = outputs.past_key_values[layer_idx][0].to(torch.bfloat16)
    val_tensor = outputs.past_key_values[layer_idx][1].to(torch.bfloat16)
    _, h, s, d = key_tensor.shape

    # Position 0
    k0_exps = extract_exp8(key_tensor[0, :, 0, :])
    v0_exps = extract_exp8(val_tensor[0, :, 0, :])
    # All other positions
    k_rest = extract_exp8(key_tensor[0, :, 1:, :])
    v_rest = extract_exp8(val_tensor[0, :, 1:, :])

    k0_nz = k0_exps[k0_exps != 0]
    kr_nz = k_rest[k_rest != 0]
    v0_nz = v0_exps[v0_exps != 0]
    vr_nz = v_rest[v_rest != 0]

    print(f"\n  Layer {layer_idx}:")
    print(f"    Key  pos=0: range=[{k0_nz.min()}-{k0_nz.max()}] span={k0_nz.max()-k0_nz.min()} mean_exp={k0_nz.mean():.1f}")
    print(f"    Key  rest:  range=[{kr_nz.min()}-{kr_nz.max()}] span={kr_nz.max()-kr_nz.min()} mean_exp={kr_nz.mean():.1f}")
    print(f"    Value pos=0: range=[{v0_nz.min()}-{v0_nz.max()}] span={v0_nz.max()-v0_nz.min()} mean_exp={v0_nz.mean():.1f}")
    print(f"    Value rest:  range=[{vr_nz.min()}-{vr_nz.max()}] span={vr_nz.max()-vr_nz.min()} mean_exp={vr_nz.mean():.1f}")
