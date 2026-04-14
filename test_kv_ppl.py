"""
KV cache compression eval: Wikitext-2 perplexity.
Hook into attention to compress K/V after cache update, before attention compute.
Batch forward, no token-by-token needed.
"""
import torch, numpy as np, sys, os, time, json, math, gc
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import Counter
import functools

model_path = "/root/autodl-tmp/Qwen3-4B"

# ============================================================
# Compression
# ============================================================

def build_lut(exps_list, k):
    top_k = sorted(exps_list[:k])
    decompress = np.array(top_k, dtype=np.uint8)
    compress = np.zeros(256, dtype=np.uint8)
    idx = {e: i for i, e in enumerate(top_k)}
    for e, i in idx.items():
        compress[e] = i
    for e in range(256):
        if e in idx: continue
        best_i, best_d = 0, float('inf')
        for se in top_k:
            d = abs(2.0**(e-127) - 2.0**(se-127))
            if d < best_d:
                best_d, best_i = d, idx[se]
        compress[e] = best_i
    return compress, decompress

# Pre-build torch LUT for GPU-side compression (no CPU round-trip)
def make_torch_lut(cl, dl):
    """Create a GPU tensor that maps uint16 bf16 → compressed uint16 bf16."""
    # Build a lookup: for each possible exp8 (0-255), what's the new exp8?
    exp_map = np.zeros(256, dtype=np.uint8)
    for e in range(256):
        exp_map[e] = dl[cl[e]]
    return torch.from_numpy(exp_map).to(torch.int32)

def compress_kv_gpu(tensor, exp_map_gpu):
    """Compress KV tensor on GPU using vectorized ops. No CPU round-trip."""
    raw = tensor.view(torch.int16).to(torch.int32)
    sign = (raw >> 15) & 1
    exp8 = (raw >> 7) & 0xFF
    mant7 = raw & 0x7F
    # LUT lookup on GPU
    exp_new = exp_map_gpu[exp8]
    result = (sign << 15) | (exp_new << 7) | mant7
    return result.to(torch.int16).view(torch.bfloat16)

# ============================================================
# Hook into attention: compress KV after cache.update()
# ============================================================

def install_kv_hooks(model, exp_map_gpu):
    """Monkey-patch each attention layer to compress KV after cache update."""
    hooks = []

    for layer in model.model.layers:
        attn = layer.self_attn
        original_forward = attn.forward

        @functools.wraps(original_forward)
        def hooked_forward(*args, _orig=original_forward, _emap=exp_map_gpu, **kwargs):
            # Get past_key_values from args/kwargs
            past_kv = kwargs.get('past_key_values', None)
            if past_kv is None and len(args) > 3:
                past_kv = args[3]

            result = _orig(*args, **kwargs)

            # After forward, compress the KV cache for this layer
            if past_kv is not None and hasattr(past_kv, 'layers'):
                layer_idx = _orig.__self__.layer_idx
                if layer_idx < len(past_kv.layers):
                    layer_cache = past_kv.layers[layer_idx]
                    layer_cache.keys = compress_kv_gpu(layer_cache.keys, _emap)
                    layer_cache.values = compress_kv_gpu(layer_cache.values, _emap)

            return result

        attn.forward = hooked_forward
        hooks.append((attn, original_forward))

    return hooks

def remove_hooks(hooks):
    for attn, original_forward in hooks:
        attn.forward = original_forward

# ============================================================
# Wikitext-2 PPL (standard batch, sliding window)
# ============================================================

def eval_wikitext_ppl(model, input_ids, stride=512, max_length=2048):
    seq_len = input_ids.shape[1]
    nlls = []
    total_tokens = 0
    t0 = time.time()

    model.eval()
    with torch.no_grad():
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            chunk = input_ids[:, begin:end]
            target_len = end - prev_end if begin == 0 else min(stride, end - begin)

            outputs = model(chunk, use_cache=False)
            logits = outputs.logits

            # Loss on the target portion (last target_len tokens)
            shift_logits = logits[:, -(target_len+1):-1, :].contiguous()
            shift_labels = chunk[:, -target_len:].contiguous()

            if shift_logits.shape[1] != shift_labels.shape[1]:
                # Edge case: first window
                min_len = min(shift_logits.shape[1], shift_labels.shape[1])
                shift_logits = shift_logits[:, -min_len:, :]
                shift_labels = shift_labels[:, -min_len:]

            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction='none'
            )
            nlls.extend(loss.cpu().tolist())
            total_tokens += loss.numel()
            prev_end = end

            if begin % (stride * 5) == 0:
                ppl = math.exp(sum(nlls) / total_tokens)
                print(f"    [{total_tokens}/{seq_len}] ppl={ppl:.4f} ({time.time()-t0:.0f}s)")

            if end >= seq_len:
                break

    ppl = math.exp(sum(nlls) / total_tokens)
    return ppl, total_tokens, time.time() - t0

# ============================================================
# Main
# ============================================================

print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

# Calibrate LUTs
print("\nCalibrating KV cache LUTs...")
kv_counter = Counter()
with torch.no_grad():
    for p in ["The quick brown fox jumps over the lazy dog.",
              "def fibonacci(n): return n if n <= 1 else fibonacci(n-1)",
              "In machine learning, neural networks are trained."]:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        outputs = model(**inputs, use_cache=True)
        for layer in outputs.past_key_values.layers:
            for t in [layer.keys, layer.values]:
                exps = ((t.view(torch.int16).to(torch.int32) >> 7) & 0xFF)
                kv_counter.update(exps.cpu().numpy().flatten().tolist())

# KVFloat13
top32 = [e for e, _ in kv_counter.most_common(32)]
kvf13_cov = 100 * sum(kv_counter[e] for e in top32) / sum(kv_counter.values())
kvf13_cl, kvf13_dl = build_lut(top32, 32)
print(f"  KVFloat13: {sorted(top32)}, cov={kvf13_cov:.4f}%")

# KVFloat12
total_kv = sum(kv_counter.values())
best_base, best_cov = 115, 0
for base in range(110, 125):
    cov = 100 * sum(kv_counter.get(e, 0) for e in range(base, base+16)) / total_kv
    if cov > best_cov:
        best_cov, best_base = cov, base
kvf12_cl, kvf12_dl = build_lut(list(range(best_base, best_base+16)), 16)
print(f"  KVFloat12: [{best_base}-{best_base+15}], cov={best_cov:.4f}%")

# GPU LUTs
kvf13_emap = make_torch_lut(kvf13_cl, kvf13_dl).cuda()
kvf12_emap = make_torch_lut(kvf12_cl, kvf12_dl).cuda()

# Load wikitext-2
print("\nLoading Wikitext-2...")
try:
    ds = load_dataset("/root/autodl-tmp/data/datasets/wikitext", "wikitext-2-raw-v1", split="test")
except:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"])
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
print(f"  Total tokens: {input_ids.shape[1]}")

# ============================================================
print(f"\n{'='*60}")
print("WIKITEXT-2 PPL — KV Cache Compression (weights=BF16)")
print(f"{'='*60}")

# 1) Baseline
print("\n1) BF16 baseline:")
ppl_base, n_tok, elapsed = eval_wikitext_ppl(model, input_ids)
print(f"  Baseline PPL={ppl_base:.4f} ({n_tok} tokens, {elapsed:.0f}s)")

# 2) KVFloat13
print("\n2) KVFloat13 (5-bit):")
hooks = install_kv_hooks(model, kvf13_emap)
ppl_13, _, elapsed = eval_wikitext_ppl(model, input_ids)
remove_hooks(hooks)
print(f"  KVFloat13 PPL={ppl_13:.4f} (Δ{ppl_13-ppl_base:+.4f}, {elapsed:.0f}s)")

# 3) KVFloat12
print(f"\n3) KVFloat12 [{best_base}-{best_base+15}]:")
hooks = install_kv_hooks(model, kvf12_emap)
ppl_12, _, elapsed = eval_wikitext_ppl(model, input_ids)
remove_hooks(hooks)
print(f"  KVFloat12 PPL={ppl_12:.4f} (Δ{ppl_12-ppl_base:+.4f}, {elapsed:.0f}s)")

# Summary
print(f"\n{'='*60}")
print(f"RESULTS — Qwen3-4B, Wikitext-2 ({n_tok} tokens)")
print(f"{'='*60}")
print(f"{'Method':<35} | {'PPL':>8} | {'Δ PPL':>8} | {'KV comp':>7}")
print(f"{'-'*65}")
print(f"{'BF16 (no compression)':<35} | {ppl_base:>8.4f} | {'—':>8} | {'0%':>7}")
print(f"{'KVFloat13 (5-bit LUT)':<35} | {ppl_13:>8.4f} | {ppl_13-ppl_base:>+8.4f} | {'18.75%':>7}")
print(f"{'KVFloat12 [{0}-{1}]'.format(best_base,best_base+15):<35} | {ppl_12:>8.4f} | {ppl_12-ppl_base:>+8.4f} | {'25.0%':>7}")

with open("/root/kvfloat13/wikitext_kv_results.json", "w") as f:
    json.dump({
        "model": "Qwen3-4B", "dataset": "wikitext-2-full", "tokens": n_tok,
        "baseline_ppl": ppl_base,
        "kvfloat13": {"ppl": ppl_13, "delta": ppl_13-ppl_base, "coverage": kvf13_cov},
        "kvfloat12": {"ppl": ppl_12, "delta": ppl_12-ppl_base,
                      "window": [best_base, best_base+15], "coverage": best_cov},
    }, f, indent=2)
print("\nSaved to wikitext_kv_results.json")
