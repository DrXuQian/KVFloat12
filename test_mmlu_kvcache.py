"""
MMLU with KV cache compression only (weights stay BF16).
Tests KVFloat13 (5-bit) and KVFloat12 LUT (4-bit) on KV cache.

KV cache compression is applied token-by-token during inference:
after each forward pass, compress all KV cache tensors before next step.
"""
import torch, numpy as np, sys, time, json, os, gc
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import lm_eval
from lm_eval.models.huggingface import HFLM

model_path = "/root/autodl-tmp/Qwen3-4B"
target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

# ============================================================
# Compression functions
# ============================================================

def build_contiguous_lut(base, width=16):
    exps = list(range(base, base + width))
    decompress = np.array(exps, dtype=np.uint8)
    compress = np.zeros(256, dtype=np.uint8)
    for i, e in enumerate(exps):
        compress[e] = i
    for e in range(256):
        if e in set(exps):
            continue
        best_i, best_d = 0, float('inf')
        for se in exps:
            d = abs(2.0**(e-127) - 2.0**(se-127))
            if d < best_d:
                best_d, best_i = d, exps.index(se)
        compress[e] = best_i
    return compress, decompress

def build_kvf13_luts_from_kv(model, tokenizer):
    """Build KVFloat13 LUT from actual KV cache exponents."""
    counter = Counter()
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In machine learning, neural networks are trained using backpropagation.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "The stock market experienced significant volatility as investors reacted.",
        "Once upon a time in a distant kingdom there lived a wise old wizard.",
    ]
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model(**inputs, use_cache=True)
            for layer in outputs.past_key_values.layers:
                for tensor in [layer.keys, layer.values]:
                    exps = ((tensor.to(torch.bfloat16).view(torch.int16).to(torch.int32) >> 7) & 0xFF)
                    unique, counts = exps.unique(return_counts=True)
                    for e, c in zip(unique.cpu().numpy(), counts.cpu().numpy()):
                        counter[int(e)] += int(c)

    total = sum(counter.values())
    top32 = sorted([e for e, _ in counter.most_common(32)])
    cov = 100 * sum(counter[e] for e in top32) / total

    decompress = np.array(top32, dtype=np.uint8)
    compress = np.zeros(256, dtype=np.uint8)
    exp_to_idx = {e: i for i, e in enumerate(top32)}
    for e, i in exp_to_idx.items():
        compress[e] = i
    for e in range(256):
        if e in exp_to_idx:
            continue
        best_i, best_d = 0, float('inf')
        for se in top32:
            d = abs(2.0**(e-127) - 2.0**(se-127))
            if d < best_d:
                best_d, best_i = d, exp_to_idx[se]
        compress[e] = best_i

    return compress, decompress, cov, counter

def compress_tensor_lut(tensor, cl, dl):
    """Compress a tensor via LUT (works for any bit width)."""
    shape, device, dtype = tensor.shape, tensor.device, tensor.dtype
    raw = tensor.to(torch.bfloat16).contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    sign = (raw >> 15) & 1
    exp8 = (raw >> 7) & 0xFF
    mant7 = raw & 0x7F
    exp_new = dl[cl[exp8.astype(np.uint8)]].astype(np.uint16)
    decoded = ((sign << 15) | (exp_new << 7) | mant7).astype(np.uint16)
    return torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(shape).to(device)

# ============================================================
# Custom HFLM that compresses KV cache after each forward
# ============================================================

class HFLMWithKVCompression(HFLM):
    """HFLM wrapper that compresses KV cache after each model forward pass."""

    def __init__(self, compress_fn=None, **kwargs):
        super().__init__(**kwargs)
        self._compress_fn = compress_fn
        self._original_forward = None

    def _install_hook(self):
        """Install a forward hook on the model to compress KV cache."""
        if self._compress_fn is None:
            return

        model = self.model
        original_forward = model.forward

        compress_fn = self._compress_fn

        def hooked_forward(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            # Compress KV cache if present
            if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                cache = outputs.past_key_values
                if hasattr(cache, 'layers'):
                    for layer in cache.layers:
                        layer.keys = compress_fn(layer.keys)
                        layer.values = compress_fn(layer.values)
            return outputs

        model.forward = hooked_forward
        self._original_forward = original_forward

    def _remove_hook(self):
        if self._original_forward is not None:
            self.model.forward = self._original_forward
            self._original_forward = None

# ============================================================

print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()
print(f"Loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# Build LUTs
# ============================================================
print("\nBuilding KV cache LUTs...")

# KVFloat13: top-32 from KV cache
kvf13_cl, kvf13_dl, kvf13_cov, kv_counter = build_kvf13_luts_from_kv(model, tokenizer)
print(f"  KVFloat13 KV LUT: {sorted(kvf13_dl.tolist())}")
print(f"  KV cache coverage: {kvf13_cov:.4f}%")
print(f"  Unique KV exponents: {len(kv_counter)}")

# Also build weight LUT for reference
w_counter = Counter()
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    exps = ((param.data.to(torch.bfloat16).view(torch.int16).to(torch.int32) >> 7) & 0xFF)
    unique, counts = exps.unique(return_counts=True)
    for e, c in zip(unique.cpu().numpy(), counts.cpu().numpy()):
        w_counter[int(e)] += int(c)

# Combined LUT for KVFloat13
combined = w_counter + kv_counter
top32_combined = sorted([e for e, _ in combined.most_common(32)])
kv_cov_combined = 100 * sum(kv_counter.get(e, 0) for e in top32_combined) / sum(kv_counter.values())
print(f"  Combined top-32: {top32_combined}")
print(f"  KV coverage with combined LUT: {kv_cov_combined:.4f}%")

# KVFloat12: sliding window for KV cache
print("\n  Finding best 4-bit window for KV cache...")
best_base_kv, best_cov_kv = 115, 0
for base in range(110, 125):
    window = set(range(base, base + 16))
    cov = 100 * sum(kv_counter.get(e, 0) for e in window) / sum(kv_counter.values())
    if cov > best_cov_kv:
        best_cov_kv, best_base_kv = cov, base
    if base % 3 == 0:
        print(f"    [{base}-{base+15}] KV coverage={cov:.4f}%")

kvf12_cl, kvf12_dl = build_contiguous_lut(best_base_kv, 16)
print(f"  Best KVFloat12 window: [{best_base_kv}-{best_base_kv+15}] coverage={best_cov_kv:.4f}%")

# ============================================================
# MMLU tests
# ============================================================
print(f"\n{'='*60}")
print(f"MMLU (lm-eval, 5-shot) — KV cache compression only")
print(f"Weights: BF16 (unchanged)")
print(f"{'='*60}")

# 1) Baseline (no compression)
print("\n1) BF16 baseline (no KV compression):")
t0 = time.time()
lm_base = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
res_base = lm_eval.simple_evaluate(model=lm_base, tasks=["mmlu"], num_fewshot=5)
acc_base = res_base["results"]["mmlu"]["acc,none"]
stderr_base = res_base["results"]["mmlu"].get("acc_stderr,none", 0)
print(f"  BF16 MMLU acc={acc_base:.4f} ± {stderr_base:.4f} ({time.time()-t0:.0f}s)")
del lm_base
gc.collect(); torch.cuda.empty_cache()

# 2) KVFloat13 (5-bit) KV cache compression
print("\n2) KVFloat13 (5-bit) KV cache only:")
t0 = time.time()

def compress_kv_kvf13(tensor):
    return compress_tensor_lut(tensor, kvf13_cl, kvf13_dl)

lm_kvf13 = HFLMWithKVCompression(
    compress_fn=compress_kv_kvf13, pretrained=model, tokenizer=tokenizer, batch_size=4)
lm_kvf13._install_hook()
res_kvf13 = lm_eval.simple_evaluate(model=lm_kvf13, tasks=["mmlu"], num_fewshot=5)
acc_kvf13 = res_kvf13["results"]["mmlu"]["acc,none"]
stderr_kvf13 = res_kvf13["results"]["mmlu"].get("acc_stderr,none", 0)
lm_kvf13._remove_hook()
print(f"  KVFloat13 MMLU acc={acc_kvf13:.4f} ± {stderr_kvf13:.4f} ({time.time()-t0:.0f}s)")
del lm_kvf13
gc.collect(); torch.cuda.empty_cache()

# 3) KVFloat12 (4-bit) KV cache compression
print(f"\n3) KVFloat12 LUT [{best_base_kv}-{best_base_kv+15}] KV cache only:")
t0 = time.time()

def compress_kv_kvf12(tensor):
    return compress_tensor_lut(tensor, kvf12_cl, kvf12_dl)

lm_kvf12 = HFLMWithKVCompression(
    compress_fn=compress_kv_kvf12, pretrained=model, tokenizer=tokenizer, batch_size=4)
lm_kvf12._install_hook()
res_kvf12 = lm_eval.simple_evaluate(model=lm_kvf12, tasks=["mmlu"], num_fewshot=5)
acc_kvf12 = res_kvf12["results"]["mmlu"]["acc,none"]
stderr_kvf12 = res_kvf12["results"]["mmlu"].get("acc_stderr,none", 0)
lm_kvf12._remove_hook()
print(f"  KVFloat12 MMLU acc={acc_kvf12:.4f} ± {stderr_kvf12:.4f} ({time.time()-t0:.0f}s)")
del lm_kvf12
gc.collect(); torch.cuda.empty_cache()

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"KV CACHE COMPRESSION MMLU RESULTS — Qwen3-4B")
print(f"Weights: BF16 (full precision)")
print(f"{'='*60}")
print(f"{'Method':<35} | {'Acc':>7} | {'± stderr':>8} | {'Δ Acc':>7} | {'KV comp':>7}")
print(f"{'-'*72}")
print(f"{'No compression':<35} | {acc_base:>7.4f} | {stderr_base:>8.4f} | {'—':>7} | {'0%':>7}")
print(f"{'KVFloat13 (5-bit, 32-entry LUT)':<35} | {acc_kvf13:>7.4f} | {stderr_kvf13:>8.4f} | {acc_kvf13-acc_base:>+7.4f} | {'18.75%':>7}")
print(f"{'KVFloat12 [{0}-{1}] (4-bit LUT)'.format(best_base_kv, best_base_kv+15):<35} | {acc_kvf12:>7.4f} | {stderr_kvf12:>8.4f} | {acc_kvf12-acc_base:>+7.4f} | {'25.0%':>7}")

with open("/root/kvfloat13/mmlu_kvcache_results.json", "w") as f:
    json.dump({
        "model": "Qwen3-4B",
        "eval": "lm-eval mmlu 5-shot, KV cache compression only",
        "baseline": {"acc": acc_base, "stderr": stderr_base},
        "kvfloat13_kv": {"acc": acc_kvf13, "stderr": stderr_kvf13,
                         "lut": sorted(kvf13_dl.tolist()), "coverage": kvf13_cov},
        "kvfloat12_kv": {"acc": acc_kvf12, "stderr": stderr_kvf12,
                         "window": [best_base_kv, best_base_kv+15], "coverage": best_cov_kv},
    }, f, indent=2)
print("\nSaved to mmlu_kvcache_results.json")
