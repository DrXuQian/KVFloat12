"""
Qwen3-4B: per-layer KV cache exponent analysis.
Does each layer need its own LUT window?
"""
import torch, numpy as np, sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

model_path = "/root/autodl-tmp/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

prompts = [
    "The quick brown fox jumps over the lazy dog in a warm summer afternoon by the river.",
    "def quicksort(arr): return [] if not arr else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]]",
    "The integral of x^2 from 0 to 1 equals 1/3. Consider the Taylor expansion of e^x.",
    "In the beginning, there was nothing but void and darkness. Then came light, and with it, the first stars formed.",
    "xkcd 1729 αβγδ ∫∫∫ 0xDEADBEEF !!!??? ===",
]

# Per-layer, per-KV, per-input stats
layer_kv_counters = {}  # (layer, 'key'/'value') -> Counter

with torch.no_grad():
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model(**inputs, use_cache=True)
        cache = outputs.past_key_values
        for li, layer in enumerate(cache.layers):
            for kv_name, tensor in [("key", layer.keys), ("value", layer.values)]:
                exps = ((tensor.to(torch.bfloat16).view(torch.int16).to(torch.int32) >> 7) & 0xFF)
                key = (li, kv_name)
                if key not in layer_kv_counters:
                    layer_kv_counters[key] = Counter()
                layer_kv_counters[key].update(exps.cpu().numpy().flatten().tolist())

n_layers = model.config.num_hidden_layers

# ============================================================
print("=" * 80)
print(f"PER-LAYER KV CACHE EXPONENT RANGE — Qwen3-4B ({n_layers} layers)")
print("=" * 80)
print(f"{'Layer':>5} | {'KV':<6} | {'Uniq':>4} | {'Min':>4} | {'Max':>4} | {'Span':>4} | {'Center':>6} | {'16-win coverage':>15} | Best window")
print("-" * 95)

global_counter = Counter()
per_layer_best_windows = {}

for li in range(n_layers):
    for kv_name in ["key", "value"]:
        c = layer_kv_counters.get((li, kv_name), Counter())
        global_counter.update(c)

        used = sorted(c.keys())
        nonzero = [e for e in used if e != 0]
        if not nonzero:
            continue
        mn, mx = nonzero[0], nonzero[-1]
        span = mx - mn + 1
        total = sum(c.values())
        center = sum(e * cnt for e, cnt in c.items() if e != 0) / max(1, sum(cnt for e, cnt in c.items() if e != 0))

        # Find best 16-wide window for this layer+kv
        best_base, best_cov = 100, 0
        for base in range(max(90, mn-5), min(140, mx+1)):
            window = set(range(base, base + 16))
            cov = 100 * sum(c.get(e, 0) for e in window) / total
            if cov > best_cov:
                best_cov, best_base = cov, base

        per_layer_best_windows[(li, kv_name)] = (best_base, best_cov)

        if li % 4 == 0 or li == n_layers - 1:
            print(f"{li:>5} | {kv_name:<6} | {len(used):>4} | {mn:>4} | {mx:>4} | {span:>4} | {center:>6.1f} | {best_cov:>14.2f}% | [{best_base}-{best_base+15}]")

# ============================================================
print(f"\n{'='*80}")
print("GLOBAL vs PER-LAYER WINDOW COMPARISON")
print(f"{'='*80}")

# Global best window
total_global = sum(global_counter.values())
best_global_base, best_global_cov = 100, 0
for base in range(90, 135):
    window = set(range(base, base + 16))
    cov = 100 * sum(global_counter.get(e, 0) for e in window) / total_global
    if cov > best_global_cov:
        best_global_cov, best_global_base = cov, base

print(f"Global best 16-window: [{best_global_base}-{best_global_base+15}] coverage={best_global_cov:.4f}%")

# Per-layer coverage with global window
print(f"\nPer-layer coverage using GLOBAL window [{best_global_base}-{best_global_base+15}]:")
global_window = set(range(best_global_base, best_global_base + 16))
min_cov, max_cov = 100, 0
worst_layers = []

for li in range(n_layers):
    for kv_name in ["key", "value"]:
        c = layer_kv_counters.get((li, kv_name), Counter())
        total = sum(c.values())
        if total == 0:
            continue
        cov = 100 * sum(c.get(e, 0) for e in global_window) / total
        if cov < min_cov:
            min_cov = cov
        if cov > max_cov:
            max_cov = cov
        if cov < 99.0:
            worst_layers.append((li, kv_name, cov))

        # Also check: what's the coverage with this layer's OWN best window?
        own_base, own_cov = per_layer_best_windows[(li, kv_name)]

print(f"  Min coverage: {min_cov:.4f}%")
print(f"  Max coverage: {max_cov:.4f}%")
if worst_layers:
    print(f"  Layers below 99%:")
    for li, kv, cov in sorted(worst_layers):
        own_base, own_cov = per_layer_best_windows[(li, kv)]
        print(f"    L{li:>2} {kv:<5}: global={cov:.2f}% own=[{own_base}-{own_base+15}]={own_cov:.2f}%")
else:
    print(f"  All layers ≥ 99%")

# Show the variance in best windows
key_windows = [per_layer_best_windows[(li, "key")][0] for li in range(n_layers)]
val_windows = [per_layer_best_windows[(li, "value")][0] for li in range(n_layers)]
print(f"\n  Key best window bases:   min={min(key_windows)} max={max(key_windows)} spread={max(key_windows)-min(key_windows)}")
print(f"  Value best window bases: min={min(val_windows)} max={max(val_windows)} spread={max(val_windows)-min(val_windows)}")

# ============================================================
print(f"\n{'='*80}")
print("WOULD PER-LAYER WINDOWS HELP?")
print(f"{'='*80}")

# Compare: global window total coverage vs per-layer window total coverage
global_total_covered = 0
perlayer_total_covered = 0
total_all = 0

for li in range(n_layers):
    for kv_name in ["key", "value"]:
        c = layer_kv_counters.get((li, kv_name), Counter())
        total = sum(c.values())
        total_all += total

        # Global window coverage
        global_total_covered += sum(c.get(e, 0) for e in global_window)

        # Per-layer window
        own_base = per_layer_best_windows[(li, kv_name)][0]
        own_window = set(range(own_base, own_base + 16))
        perlayer_total_covered += sum(c.get(e, 0) for e in own_window)

global_pct = 100 * global_total_covered / total_all
perlayer_pct = 100 * perlayer_total_covered / total_all
print(f"  Global window coverage:    {global_pct:.6f}%")
print(f"  Per-layer window coverage: {perlayer_pct:.6f}%")
print(f"  Improvement:               {perlayer_pct - global_pct:+.6f}pp")
print(f"  {'→ Per-layer HELPS' if perlayer_pct - global_pct > 0.1 else '→ Per-layer NOT needed (marginal gain)'}")
