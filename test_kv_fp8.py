"""
FP8 KV cache compression as baseline comparison.
FP8 E4M3: sign(1) + exp(4) + mantissa(3) = 8 bits, 50% compression.
FP8 E5M2: sign(1) + exp(5) + mantissa(2) = 8 bits, 50% compression.
"""
import torch, numpy as np, sys, os, time, json, math
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset
from collections import Counter

model_path = "/root/autodl-tmp/Qwen3-4B"

# ============================================================
# KVFloat13 LUT (for comparison)
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

def make_gpu_exp_map(cl, dl):
    exp_map = np.zeros(256, dtype=np.uint8)
    for e in range(256):
        exp_map[e] = dl[cl[e]]
    return torch.from_numpy(exp_map).to(torch.int32).cuda()

def compress_tensor_lut(t, exp_map):
    raw = t.view(torch.int16).to(torch.int32)
    sign = (raw >> 15) & 1
    exp8 = (raw >> 7) & 0xFF
    mant7 = raw & 0x7F
    exp_new = exp_map[exp8]
    return ((sign << 15) | (exp_new << 7) | mant7).to(torch.int16).view(torch.bfloat16)

# ============================================================
# FP8 compression: bf16 → fp8 → bf16 round-trip
# ============================================================

def compress_cache_fp8_e4m3(cache):
    """Compress KV cache via BF16→FP8_E4M3→BF16 round-trip."""
    for layer in cache.layers:
        layer.keys = layer.keys.to(torch.float8_e4m3fn).to(torch.bfloat16)
        layer.values = layer.values.to(torch.float8_e4m3fn).to(torch.bfloat16)

def compress_cache_fp8_e5m2(cache):
    """Compress KV cache via BF16→FP8_E5M2→BF16 round-trip."""
    for layer in cache.layers:
        layer.keys = layer.keys.to(torch.float8_e5m2).to(torch.bfloat16)
        layer.values = layer.values.to(torch.float8_e5m2).to(torch.bfloat16)

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

# ============================================================
print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

# Calibrate KVFloat13
print("\nCalibrating KVFloat13...")
kv_counter = Counter()
with torch.no_grad():
    for p in ["The quick brown fox jumps over the lazy dog.",
              "def fibonacci(n): return n if n <= 1 else fibonacci(n-1)",
              "In machine learning, neural networks are trained."]:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        out = model(**inputs, use_cache=True)
        for layer in out.past_key_values.layers:
            for t in [layer.keys, layer.values]:
                exps = ((t.view(torch.int16).to(torch.int32) >> 7) & 0xFF)
                kv_counter.update(exps.cpu().numpy().flatten().tolist())

top32 = [e for e, _ in kv_counter.most_common(32)]
kvf13_cl, kvf13_dl = build_lut(top32, 32)
kvf13_emap = make_gpu_exp_map(kvf13_cl, kvf13_dl)
print(f"  KVFloat13 cov={100*sum(kv_counter[e] for e in top32)/sum(kv_counter.values()):.4f}%")

# Load data
print("\nLoading Wikitext-2...")
try:
    ds = load_dataset("/root/autodl-tmp/data/datasets/wikitext", "wikitext-2-raw-v1", split="test")
except:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"])
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

MAX_TOKENS = 4096

print(f"\n{'='*65}")
print(f"DECODE PPL — KV Cache Compression Comparison ({MAX_TOKENS} tokens)")
print(f"{'='*65}")

# 1) Baseline
print("\n1) BF16 baseline:")
ppl_base, n, e = eval_decode_ppl(model, input_ids, max_tokens=MAX_TOKENS)
print(f"  PPL={ppl_base:.4f} ({e:.0f}s)")
torch.cuda.empty_cache()

# 2) KVFloat13
print("\n2) KVFloat13 (5-bit, 18.75% compression):")
ppl_13, _, e = eval_decode_ppl(model, input_ids,
    lambda c: [setattr(l, 'keys', compress_tensor_lut(l.keys, kvf13_emap)) or
               setattr(l, 'values', compress_tensor_lut(l.values, kvf13_emap))
               for l in c.layers], MAX_TOKENS)
print(f"  PPL={ppl_13:.4f} (Δ{ppl_13-ppl_base:+.4f}, {e:.0f}s)")
torch.cuda.empty_cache()

# 3) FP8 E4M3
print("\n3) FP8 E4M3 (50% compression):")
ppl_fp8_e4m3, _, e = eval_decode_ppl(model, input_ids, compress_cache_fp8_e4m3, MAX_TOKENS)
print(f"  PPL={ppl_fp8_e4m3:.4f} (Δ{ppl_fp8_e4m3-ppl_base:+.4f}, {e:.0f}s)")
torch.cuda.empty_cache()

# 4) FP8 E5M2
print("\n4) FP8 E5M2 (50% compression):")
ppl_fp8_e5m2, _, e = eval_decode_ppl(model, input_ids, compress_cache_fp8_e5m2, MAX_TOKENS)
print(f"  PPL={ppl_fp8_e5m2:.4f} (Δ{ppl_fp8_e5m2-ppl_base:+.4f}, {e:.0f}s)")

# Summary
print(f"\n{'='*65}")
print(f"RESULTS — Qwen3-4B KV Cache Compression ({MAX_TOKENS} tokens)")
print(f"{'='*65}")
print(f"{'Method':<35} | {'PPL':>8} | {'Δ PPL':>8} | {'Bits':>4} | {'Compress':>8}")
print(f"{'-'*72}")
print(f"{'BF16 (no compression)':<35} | {ppl_base:>8.4f} | {'—':>8} | {'16':>4} | {'0%':>8}")
print(f"{'KVFloat13 (5-bit exp)':<35} | {ppl_13:>8.4f} | {ppl_13-ppl_base:>+8.4f} | {'13':>4} | {'18.75%':>8}")
print(f"{'FP8 E4M3':<35} | {ppl_fp8_e4m3:>8.4f} | {ppl_fp8_e4m3-ppl_base:>+8.4f} | {'8':>4} | {'50%':>8}")
print(f"{'FP8 E5M2':<35} | {ppl_fp8_e5m2:>8.4f} | {ppl_fp8_e5m2-ppl_base:>+8.4f} | {'8':>4} | {'50%':>8}")

with open("/root/kvfloat13/kv_comparison_results.json", "w") as f:
    json.dump({
        "model": "Qwen3-4B", "tokens": MAX_TOKENS,
        "baseline": ppl_base,
        "kvfloat13": {"ppl": ppl_13, "bits": 13, "compress": "18.75%"},
        "fp8_e4m3": {"ppl": ppl_fp8_e4m3, "bits": 8, "compress": "50%"},
        "fp8_e5m2": {"ppl": ppl_fp8_e5m2, "bits": 8, "compress": "50%"},
    }, f, indent=2)
print("\nSaved to kv_comparison_results.json")
