"""
Long-context decode PPL: test where FP8 noise accumulates vs KVFloat13 zero-error.
Sweep sequence lengths: 4K, 8K, 16K, 32K.
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

def compress_tensor_lut(t, exp_map):
    raw = t.view(torch.int16).to(torch.int32)
    sign = (raw >> 15) & 1
    exp8 = (raw >> 7) & 0xFF
    mant7 = raw & 0x7F
    exp_new = exp_map[exp8]
    return ((sign << 15) | (exp_new << 7) | mant7).to(torch.int16).view(torch.bfloat16)

def compress_cache_kvf13(cache, emap):
    for layer in cache.layers:
        layer.keys = compress_tensor_lut(layer.keys, emap)
        layer.values = compress_tensor_lut(layer.values, emap)

def compress_cache_fp8_e4m3(cache):
    for layer in cache.layers:
        layer.keys = layer.keys.to(torch.float8_e4m3fn).to(torch.bfloat16)
        layer.values = layer.values.to(torch.float8_e4m3fn).to(torch.bfloat16)

def eval_decode_ppl(model, input_ids, compress_fn, max_tokens):
    """Only compute PPL on last 1024 tokens (after cache is warm)."""
    seq_len = min(input_ids.shape[1], max_tokens)
    nlls = []
    t0 = time.time()
    cache = DynamicCache()
    eval_start = max(0, seq_len - 1024 - 1)  # eval last 1024 tokens

    model.eval()
    with torch.no_grad():
        for t in range(seq_len - 1):
            tok = input_ids[:, t:t+1]
            out = model(tok, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if compress_fn is not None:
                compress_fn(cache)

            if t >= eval_start:
                logits = out.logits[:, -1, :]
                target = input_ids[:, t+1]
                loss = torch.nn.functional.cross_entropy(logits, target)
                nlls.append(loss.item())

            if (t+1) % 4000 == 0:
                elapsed = time.time() - t0
                tps = (t+1) / elapsed
                if nlls:
                    ppl = math.exp(sum(nlls) / len(nlls))
                    print(f"    [{t+1}/{seq_len}] ppl={ppl:.4f} ({elapsed:.0f}s, {tps:.0f} tok/s)")
                else:
                    print(f"    [{t+1}/{seq_len}] warming up ({elapsed:.0f}s, {tps:.0f} tok/s)")

    ppl = math.exp(sum(nlls) / len(nlls))
    elapsed = time.time() - t0
    return ppl, len(nlls), elapsed

# ============================================================
print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

# Calibrate KVFloat13
print("\nCalibrating...")
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
print(f"  KVFloat13 ready")

# Load data
print("\nLoading Wikitext-2...")
try:
    ds = load_dataset("/root/autodl-tmp/data/datasets/wikitext", "wikitext-2-raw-v1", split="test")
except:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"])
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
print(f"  Available tokens: {input_ids.shape[1]}")

# ============================================================
print(f"\n{'='*70}")
print("LONG-CONTEXT DECODE PPL — KV Cache Compression Scaling")
print("PPL measured on LAST 1024 tokens (after N-1024 tokens of context)")
print(f"{'='*70}")

results = {}
for context_len in [4096, 8192, 16384, 32768]:
    if context_len > input_ids.shape[1]:
        print(f"\n  Skipping {context_len} (only {input_ids.shape[1]} tokens available)")
        continue

    print(f"\n{'='*50}")
    print(f"  Context: {context_len} tokens (eval last 1024)")
    print(f"{'='*50}")

    # Baseline
    print(f"  BF16:")
    ppl_base, n, e = eval_decode_ppl(model, input_ids, None, context_len)
    print(f"    PPL={ppl_base:.4f} ({e:.0f}s)")
    torch.cuda.empty_cache()

    # KVFloat13
    print(f"  KVFloat13:")
    ppl_13, _, e = eval_decode_ppl(model, input_ids,
        lambda c: compress_cache_kvf13(c, kvf13_emap), context_len)
    print(f"    PPL={ppl_13:.4f} (Δ{ppl_13-ppl_base:+.4f}, {e:.0f}s)")
    torch.cuda.empty_cache()

    # FP8 E4M3
    print(f"  FP8 E4M3:")
    ppl_fp8, _, e = eval_decode_ppl(model, input_ids,
        compress_cache_fp8_e4m3, context_len)
    print(f"    PPL={ppl_fp8:.4f} (Δ{ppl_fp8-ppl_base:+.4f}, {e:.0f}s)")
    torch.cuda.empty_cache()

    results[context_len] = {
        "baseline": ppl_base,
        "kvfloat13": {"ppl": ppl_13, "delta": ppl_13 - ppl_base},
        "fp8_e4m3": {"ppl": ppl_fp8, "delta": ppl_fp8 - ppl_base},
    }

# Summary
print(f"\n{'='*70}")
print("SUMMARY: PPL vs Context Length")
print(f"{'='*70}")
print(f"{'Context':>8} | {'BF16':>8} | {'KVF13':>8} | {'Δ KVF13':>8} | {'FP8':>8} | {'Δ FP8':>8}")
print("-" * 60)
for ctx in sorted(results.keys()):
    r = results[ctx]
    print(f"{ctx:>8} | {r['baseline']:>8.4f} | {r['kvfloat13']['ppl']:>8.4f} | "
          f"{r['kvfloat13']['delta']:>+8.4f} | {r['fp8_e4m3']['ppl']:>8.4f} | "
          f"{r['fp8_e4m3']['delta']:>+8.4f}")

with open("/root/kvfloat13/long_context_results.json", "w") as f:
    json.dump({"model": "Qwen3-4B", "results": {str(k): v for k, v in results.items()}}, f, indent=2)
print("\nSaved to long_context_results.json")
