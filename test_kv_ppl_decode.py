"""
KV cache compression: token-by-token decode PPL on Wikitext-2.
Simulates real inference: each token's KV is compressed before next token reads it.
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

def compress_cache_gpu(cache, exp_map):
    for layer in cache.layers:
        for attr in ['keys', 'values']:
            t = getattr(layer, attr)
            raw = t.view(torch.int16).to(torch.int32)
            sign = (raw >> 15) & 1
            exp8 = (raw >> 7) & 0xFF
            mant7 = raw & 0x7F
            exp_new = exp_map[exp8]
            result = (sign << 15) | (exp_new << 7) | mant7
            setattr(layer, attr, result.to(torch.int16).view(torch.bfloat16))

def eval_decode_ppl(model, input_ids, compress_fn=None, max_tokens=8192):
    """Token-by-token PPL with KV cache compression."""
    seq_len = min(input_ids.shape[1], max_tokens)
    nlls = []
    t0 = time.time()

    model.eval()
    cache = DynamicCache()

    with torch.no_grad():
        for t in range(seq_len - 1):
            tok = input_ids[:, t:t+1]
            out = model(tok, past_key_values=cache, use_cache=True)
            cache = out.past_key_values

            if compress_fn is not None:
                compress_fn(cache)

            # Loss: predict next token
            logits = out.logits[:, -1, :]
            target = input_ids[:, t+1]
            loss = torch.nn.functional.cross_entropy(logits, target)
            nlls.append(loss.item())

            if (t+1) % 1000 == 0:
                ppl = math.exp(sum(nlls) / len(nlls))
                elapsed = time.time() - t0
                tps = (t+1) / elapsed
                print(f"    [{t+1}/{seq_len}] ppl={ppl:.4f} ({elapsed:.0f}s, {tps:.0f} tok/s)")

    ppl = math.exp(sum(nlls) / len(nlls))
    elapsed = time.time() - t0
    return ppl, len(nlls), elapsed

# ============================================================
print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

# Calibrate
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
kvf13_cov = 100 * sum(kv_counter[e] for e in top32) / sum(kv_counter.values())
kvf13_cl, kvf13_dl = build_lut(top32, 32)
kvf13_emap = make_gpu_exp_map(kvf13_cl, kvf13_dl)
print(f"  KVFloat13 cov={kvf13_cov:.4f}%")

total_kv = sum(kv_counter.values())
best_base, best_cov = 115, 0
for base in range(110, 125):
    cov = 100 * sum(kv_counter.get(e, 0) for e in range(base, base+16)) / total_kv
    if cov > best_cov:
        best_cov, best_base = cov, base
kvf12_cl, kvf12_dl = build_lut(list(range(best_base, best_base+16)), 16)
kvf12_emap = make_gpu_exp_map(kvf12_cl, kvf12_dl)
print(f"  KVFloat12 [{best_base}-{best_base+15}] cov={best_cov:.4f}%")

# Load data
print("\nLoading Wikitext-2...")
try:
    ds = load_dataset("/root/autodl-tmp/data/datasets/wikitext", "wikitext-2-raw-v1", split="test")
except:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"])
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
print(f"  Tokens: {input_ids.shape[1]}")

MAX_TOKENS = 4096  # 4K tokens for decode PPL (token-by-token is slow)

print(f"\n{'='*60}")
print(f"DECODE PPL — KV Cache Compression (weights=BF16, {MAX_TOKENS} tokens)")
print(f"{'='*60}")

# 1) Baseline
print("\n1) BF16 baseline (no KV compression):")
ppl_base, n, elapsed = eval_decode_ppl(model, input_ids, compress_fn=None, max_tokens=MAX_TOKENS)
print(f"  Baseline PPL={ppl_base:.4f} ({n} tokens, {elapsed:.0f}s)")

torch.cuda.empty_cache()

# 2) KVFloat13
print("\n2) KVFloat13 (5-bit):")
ppl_13, _, elapsed = eval_decode_ppl(
    model, input_ids,
    compress_fn=lambda c: compress_cache_gpu(c, kvf13_emap),
    max_tokens=MAX_TOKENS)
print(f"  KVFloat13 PPL={ppl_13:.4f} (Δ{ppl_13-ppl_base:+.4f}, {elapsed:.0f}s)")

torch.cuda.empty_cache()

# 3) KVFloat12
print(f"\n3) KVFloat12 [{best_base}-{best_base+15}]:")
ppl_12, _, elapsed = eval_decode_ppl(
    model, input_ids,
    compress_fn=lambda c: compress_cache_gpu(c, kvf12_emap),
    max_tokens=MAX_TOKENS)
print(f"  KVFloat12 PPL={ppl_12:.4f} (Δ{ppl_12-ppl_base:+.4f}, {elapsed:.0f}s)")

# Summary
print(f"\n{'='*60}")
print(f"RESULTS — Qwen3-4B decode PPL ({MAX_TOKENS} tokens)")
print(f"{'='*60}")
print(f"{'Method':<35} | {'PPL':>8} | {'Δ PPL':>8} | {'KV comp':>7}")
print(f"{'-'*65}")
print(f"{'BF16 (no compression)':<35} | {ppl_base:>8.4f} | {'—':>8} | {'0%':>7}")
print(f"{'KVFloat13 (5-bit LUT)':<35} | {ppl_13:>8.4f} | {ppl_13-ppl_base:>+8.4f} | {'18.75%':>7}")
print(f"{'KVFloat12 [{0}-{1}]'.format(best_base,best_base+15):<35} | {ppl_12:>8.4f} | {ppl_12-ppl_base:>+8.4f} | {'25.0%':>7}")

with open("/root/kvfloat13/wikitext_kv_results.json", "w") as f:
    json.dump({
        "model": "Qwen3-4B", "dataset": "wikitext-2", "mode": "decode",
        "max_tokens": MAX_TOKENS,
        "baseline_ppl": ppl_base,
        "kvfloat13": {"ppl": ppl_13, "delta": ppl_13-ppl_base, "coverage": kvf13_cov},
        "kvfloat12": {"ppl": ppl_12, "delta": ppl_12-ppl_base,
                      "window": [best_base, best_base+15], "coverage": best_cov},
    }, f, indent=2)
print("\nSaved to wikitext_kv_results.json")
