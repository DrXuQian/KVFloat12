"""
KV cache compression with per-layer LUT windows for KVFloat12.
Each layer gets its own best 16-wide window, calibrated from KV cache data.
"""
import torch, numpy as np, sys, os, time, json, math
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset
from collections import Counter

model_path = "/root/autodl-tmp/Qwen3-4B"

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

def make_gpu_exp_map(cl, dl):
    exp_map = np.zeros(256, dtype=np.uint8)
    for e in range(256):
        exp_map[e] = dl[cl[e]]
    return torch.from_numpy(exp_map).to(torch.int32).cuda()

def compress_tensor_gpu(t, exp_map):
    raw = t.view(torch.int16).to(torch.int32)
    sign = (raw >> 15) & 1
    exp8 = (raw >> 7) & 0xFF
    mant7 = raw & 0x7F
    exp_new = exp_map[exp8]
    return ((sign << 15) | (exp_new << 7) | mant7).to(torch.int16).view(torch.bfloat16)

# ============================================================
print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()
n_layers = model.config.num_hidden_layers
print(f"  {n_layers} layers")

# ============================================================
# Calibrate per-layer KV cache exponent distributions
# ============================================================
print("\nCalibrating per-layer KV LUTs...")
layer_kv_counters = {}  # (layer_idx, 'key'/'value') -> Counter

with torch.no_grad():
    for p in ["The quick brown fox jumps over the lazy dog in a warm summer afternoon.",
              "def quicksort(arr): return [] if not arr else quicksort([x for x in arr[1:] if x < arr[0]])",
              "In machine learning, neural networks are trained using backpropagation algorithms.",
              "The stock market experienced significant volatility as investors reacted to news.",
              "Once upon a time in a distant kingdom there lived a wise old wizard who could predict."]:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        outputs = model(**inputs, use_cache=True)
        for li, layer in enumerate(outputs.past_key_values.layers):
            for kv_name, tensor in [("key", layer.keys), ("value", layer.values)]:
                exps = ((tensor.view(torch.int16).to(torch.int32) >> 7) & 0xFF)
                key = (li, kv_name)
                if key not in layer_kv_counters:
                    layer_kv_counters[key] = Counter()
                layer_kv_counters[key].update(exps.cpu().numpy().flatten().tolist())

# Build per-layer, per-KV LUTs
# Strategy: separate windows for key and value at each layer
per_layer_emaps = {}  # (layer_idx, 'key'/'value') -> gpu_exp_map
per_layer_info = {}

for li in range(n_layers):
    for kv_name in ["key", "value"]:
        c = layer_kv_counters[(li, kv_name)]
        total = sum(c.values())

        # Find best 16-wide window by coverage
        best_base, best_cov = 115, 0
        for base in range(100, 130):
            cov = 100 * sum(c.get(e, 0) for e in range(base, base+16)) / total
            if cov > best_cov:
                best_cov, best_base = cov, base

        cl, dl = build_lut(list(range(best_base, best_base+16)), 16)
        emap = make_gpu_exp_map(cl, dl)
        per_layer_emaps[(li, kv_name)] = emap
        per_layer_info[(li, kv_name)] = (best_base, best_cov)

# Print summary
print(f"\n  {'Layer':>5} | {'Key window':>12} | {'Key cov':>8} | {'Val window':>12} | {'Val cov':>8}")
print("  " + "-" * 60)
for li in range(0, n_layers, 4):
    kb, kcov = per_layer_info[(li, "key")]
    vb, vcov = per_layer_info[(li, "value")]
    print(f"  {li:>5} | [{kb}-{kb+15}] | {kcov:>7.2f}% | [{vb}-{vb+15}] | {vcov:>7.2f}%")
# Last layer
kb, kcov = per_layer_info[(n_layers-1, "key")]
vb, vcov = per_layer_info[(n_layers-1, "value")]
print(f"  {n_layers-1:>5} | [{kb}-{kb+15}] | {kcov:>7.2f}% | [{vb}-{vb+15}] | {vcov:>7.2f}%")

# Also build global KVFloat13 for comparison
all_kv = Counter()
for c in layer_kv_counters.values():
    all_kv.update(c)
top32 = [e for e, _ in all_kv.most_common(32)]
kvf13_cov = 100 * sum(all_kv[e] for e in top32) / sum(all_kv.values())
kvf13_cl, kvf13_dl = build_lut(top32, 32)
kvf13_emap = make_gpu_exp_map(kvf13_cl, kvf13_dl)
print(f"\n  KVFloat13 global cov={kvf13_cov:.4f}%")

# Global KVFloat12 for comparison
total_all = sum(all_kv.values())
best_global, best_gcov = 115, 0
for base in range(100, 130):
    cov = 100 * sum(all_kv.get(e, 0) for e in range(base, base+16)) / total_all
    if cov > best_gcov:
        best_gcov, best_global = cov, base
kvf12g_cl, kvf12g_dl = build_lut(list(range(best_global, best_global+16)), 16)
kvf12g_emap = make_gpu_exp_map(kvf12g_cl, kvf12g_dl)
print(f"  KVFloat12 global [{best_global}-{best_global+15}] cov={best_gcov:.4f}%")

# ============================================================
# Compression functions
# ============================================================

def compress_cache_kvf13(cache):
    for layer in cache.layers:
        layer.keys = compress_tensor_gpu(layer.keys, kvf13_emap)
        layer.values = compress_tensor_gpu(layer.values, kvf13_emap)

def compress_cache_kvf12_global(cache):
    for layer in cache.layers:
        layer.keys = compress_tensor_gpu(layer.keys, kvf12g_emap)
        layer.values = compress_tensor_gpu(layer.values, kvf12g_emap)

def compress_cache_kvf12_perlayer(cache):
    for li, layer in enumerate(cache.layers):
        layer.keys = compress_tensor_gpu(layer.keys, per_layer_emaps[(li, "key")])
        layer.values = compress_tensor_gpu(layer.values, per_layer_emaps[(li, "value")])

# ============================================================
# Eval
# ============================================================

def eval_decode_ppl(model, input_ids, compress_fn=None, max_tokens=4096):
    seq_len = min(input_ids.shape[1], max_tokens)
    nlls = []
    t0 = time.time()
    cache = DynamicCache()

    model.eval()
    with torch.no_grad():
        for t in range(seq_len - 1):
            tok = input_ids[:, t:t+1]
            out = model(tok, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if compress_fn is not None:
                compress_fn(cache)
            logits = out.logits[:, -1, :]
            target = input_ids[:, t+1]
            loss = torch.nn.functional.cross_entropy(logits, target)
            nlls.append(loss.item())
            if (t+1) % 1000 == 0:
                ppl = math.exp(sum(nlls) / len(nlls))
                print(f"    [{t+1}/{seq_len}] ppl={ppl:.4f} ({time.time()-t0:.0f}s)")

    ppl = math.exp(sum(nlls) / len(nlls))
    return ppl, len(nlls), time.time() - t0

# Load data
print("\nLoading Wikitext-2...")
try:
    ds = load_dataset("/root/autodl-tmp/data/datasets/wikitext", "wikitext-2-raw-v1", split="test")
except:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"])
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

MAX_TOKENS = 4096

print(f"\n{'='*60}")
print(f"DECODE PPL — KV Cache Compression ({MAX_TOKENS} tokens)")
print(f"{'='*60}")

# 1) Baseline
print("\n1) BF16 baseline:")
ppl_base, n, e = eval_decode_ppl(model, input_ids, max_tokens=MAX_TOKENS)
print(f"  PPL={ppl_base:.4f} ({n} tok, {e:.0f}s)")
torch.cuda.empty_cache()

# 2) KVFloat13 global
print("\n2) KVFloat13 (5-bit global):")
ppl_13, _, e = eval_decode_ppl(model, input_ids, compress_cache_kvf13, MAX_TOKENS)
print(f"  PPL={ppl_13:.4f} (Δ{ppl_13-ppl_base:+.4f}, {e:.0f}s)")
torch.cuda.empty_cache()

# 3) KVFloat12 global
print(f"\n3) KVFloat12 global [{best_global}-{best_global+15}]:")
ppl_12g, _, e = eval_decode_ppl(model, input_ids, compress_cache_kvf12_global, MAX_TOKENS)
print(f"  PPL={ppl_12g:.4f} (Δ{ppl_12g-ppl_base:+.4f}, {e:.0f}s)")
torch.cuda.empty_cache()

# 4) KVFloat12 per-layer
print("\n4) KVFloat12 per-layer windows:")
ppl_12p, _, e = eval_decode_ppl(model, input_ids, compress_cache_kvf12_perlayer, MAX_TOKENS)
print(f"  PPL={ppl_12p:.4f} (Δ{ppl_12p-ppl_base:+.4f}, {e:.0f}s)")

# Summary
print(f"\n{'='*60}")
print(f"RESULTS — Qwen3-4B decode PPL ({MAX_TOKENS} tokens)")
print(f"{'='*60}")
print(f"{'Method':<40} | {'PPL':>8} | {'Δ PPL':>8} | {'KV comp':>7}")
print(f"{'-'*70}")
print(f"{'BF16 (no compression)':<40} | {ppl_base:>8.4f} | {'—':>8} | {'0%':>7}")
print(f"{'KVFloat13 (5-bit global)':<40} | {ppl_13:>8.4f} | {ppl_13-ppl_base:>+8.4f} | {'18.75%':>7}")
print(f"{'KVFloat12 global [{0}-{1}]'.format(best_global,best_global+15):<40} | {ppl_12g:>8.4f} | {ppl_12g-ppl_base:>+8.4f} | {'25.0%':>7}")
print(f"{'KVFloat12 per-layer windows':<40} | {ppl_12p:>8.4f} | {ppl_12p-ppl_base:>+8.4f} | {'25.0%':>7}")

with open("/root/kvfloat13/wikitext_kv_perlayer_results.json", "w") as f:
    json.dump({
        "model": "Qwen3-4B", "tokens": MAX_TOKENS,
        "baseline": ppl_base,
        "kvfloat13_global": {"ppl": ppl_13, "delta": ppl_13-ppl_base},
        "kvfloat12_global": {"ppl": ppl_12g, "delta": ppl_12g-ppl_base,
                             "window": [best_global, best_global+15]},
        "kvfloat12_perlayer": {"ppl": ppl_12p, "delta": ppl_12p-ppl_base},
        "per_layer_windows": {f"L{li}_{kv}": [b, b+15]
                              for (li, kv), (b, _) in per_layer_info.items()
                              if li % 4 == 0 or li == n_layers-1},
    }, f, indent=2)
print("\nSaved to wikitext_kv_perlayer_results.json")
