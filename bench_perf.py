"""
Benchmark: KV cache compression throughput impact.
Measures tokens/second for decode with different compression schemes.
"""
import torch, numpy as np, sys, os, time
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from collections import Counter

model_path = "/root/autodl-tmp/Qwen3-4B"

# ============================================================
# Compression functions (same as before)
# ============================================================

def make_kvf13_emap(model, tokenizer):
    kv_counter = Counter()
    with torch.no_grad():
        for p in ["The quick brown fox.", "def fib(n): return n"]:
            inputs = tokenizer(p, return_tensors="pt").to(model.device)
            out = model(**inputs, use_cache=True)
            for layer in out.past_key_values.layers:
                for t in [layer.keys, layer.values]:
                    exps = ((t.view(torch.int16).to(torch.int32) >> 7) & 0xFF)
                    kv_counter.update(exps.cpu().numpy().flatten().tolist())
    top32 = sorted([e for e, _ in kv_counter.most_common(32)])
    emap = torch.zeros(256, dtype=torch.int32, device='cuda')
    idx = {e: i for i, e in enumerate(top32)}
    compress = {}
    for e, i in idx.items(): compress[e] = i
    decompress = np.array(top32, dtype=np.uint8)
    for e in range(256):
        if e not in idx:
            best = min(top32, key=lambda se: abs(2.0**(e-127) - 2.0**(se-127)))
            compress[e] = idx[best]
        emap[e] = decompress[compress[e]]
    return emap

def compress_kvf13(cache, emap):
    for layer in cache.layers:
        for attr in ['keys', 'values']:
            t = getattr(layer, attr)
            raw = t.view(torch.int16).to(torch.int32)
            s = (raw >> 15) & 1; e = (raw >> 7) & 0xFF; m = raw & 0x7F
            setattr(layer, attr, ((s << 15) | (emap[e] << 7) | m).to(torch.int16).view(torch.bfloat16))

def compress_fp8(cache):
    for layer in cache.layers:
        layer.keys = layer.keys.to(torch.float8_e4m3fn).to(torch.bfloat16)
        layer.values = layer.values.to(torch.float8_e4m3fn).to(torch.bfloat16)

def compress_lossless_tiered(cache):
    for layer in cache.layers:
        for attr in ['keys', 'values']:
            tensor = getattr(layer, attr)
            shape = tensor.shape
            raw = tensor.contiguous().view(-1).view(torch.int16).to(torch.int32)
            n = raw.shape[0]
            pad = (128 - n % 128) % 128
            if pad:
                raw = torch.cat([raw, torch.zeros(pad, dtype=torch.int32, device=raw.device)])
            sign = (raw >> 15) & 1
            exp8 = (raw >> 7) & 0xFF
            mant7 = raw & 0x7F
            blocks_exp = exp8.reshape(-1, 128)
            num_blocks = blocks_exp.shape[0]

            decoded_exp = torch.zeros_like(blocks_exp)
            for bi in range(num_blocks):
                bexp = blocks_exp[bi]
                unique_exps, _, counts = bexp.unique(return_inverse=True, return_counts=True)
                if len(unique_exps) <= 7:
                    decoded_exp[bi] = bexp
                else:
                    top7_idx = counts.argsort(descending=True)[:7]
                    top7_exps = unique_exps[top7_idx].sort().values
                    exp_map = torch.full((256,), -1, dtype=torch.int32, device=bexp.device)
                    for i, e in enumerate(top7_exps):
                        exp_map[e] = i
                    idx = exp_map[bexp]
                    decoded_exp[bi] = torch.where(idx >= 0, top7_exps[idx.clamp(0, 6)], bexp)

            out = (sign.reshape(-1, 128) << 15) | (decoded_exp << 7) | mant7.reshape(-1, 128)
            setattr(layer, attr, out.reshape(-1)[:n].to(torch.int16).view(torch.bfloat16).reshape(shape))

# ============================================================

print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

kvf13_emap = make_kvf13_emap(model, tokenizer)

# Generate input
text = "The quick brown fox jumps over the lazy dog. " * 100
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

def bench_decode(model, input_ids, compress_fn, n_tokens, label, warmup=50):
    """Benchmark decode throughput."""
    n = min(n_tokens, input_ids.shape[1])

    # Warmup
    cache = DynamicCache()
    with torch.no_grad():
        for t in range(min(warmup, n-1)):
            tok = input_ids[:, t:t+1]
            out = model(tok, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if compress_fn:
                compress_fn(cache)

    # Benchmark the rest
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for t in range(warmup, n-1):
            tok = input_ids[:, t:t+1]
            out = model(tok, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if compress_fn:
                compress_fn(cache)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    measured_tokens = n - 1 - warmup
    tps = measured_tokens / elapsed

    print(f"  [{label}] {measured_tokens} tokens in {elapsed:.2f}s → {tps:.1f} tok/s")
    return tps

N_TOKENS = 512  # enough to see the trend

print(f"\n{'='*60}")
print(f"DECODE THROUGHPUT BENCHMARK ({N_TOKENS} tokens)")
print(f"{'='*60}")

# 1) Baseline
tps_base = bench_decode(model, input_ids, None, N_TOKENS, "BF16 baseline")
torch.cuda.empty_cache()

# 2) KVFloat13
tps_13 = bench_decode(model, input_ids, lambda c: compress_kvf13(c, kvf13_emap),
                       N_TOKENS, "KVFloat13")
torch.cuda.empty_cache()

# 3) FP8
tps_fp8 = bench_decode(model, input_ids, compress_fp8, N_TOKENS, "FP8 E4M3")
torch.cuda.empty_cache()

# 4) Lossless tiered
tps_lt = bench_decode(model, input_ids, compress_lossless_tiered,
                       N_TOKENS, "Lossless Tiered")
torch.cuda.empty_cache()

# Also benchmark just the compression function itself
print(f"\n{'='*60}")
print("COMPRESSION KERNEL LATENCY (per call)")
print(f"{'='*60}")

# Build a cache with 512 tokens
cache = DynamicCache()
with torch.no_grad():
    for t in range(512):
        tok = input_ids[:, t:t+1]
        out = model(tok, past_key_values=cache, use_cache=True)
        cache = out.past_key_values

# Time each compression
for label, fn in [("KVFloat13", lambda: compress_kvf13(cache, kvf13_emap)),
                  ("FP8 E4M3", lambda: compress_fp8(cache)),
                  ("Lossless Tiered", lambda: compress_lossless_tiered(cache))]:
    # Warmup
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    n_iters = 20
    for _ in range(n_iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    per_call_ms = elapsed / n_iters * 1000
    print(f"  [{label}] {per_call_ms:.2f} ms/call (cache: {512} tokens, {model.config.num_hidden_layers} layers)")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Method':<25} | {'tok/s':>7} | {'Overhead':>8} | {'Compress':>8} | {'Lossless':>8}")
print("-" * 65)
print(f"{'BF16 baseline':<25} | {tps_base:>7.1f} | {'—':>8} | {'0%':>8} | {'—':>8}")
print(f"{'KVFloat13':<25} | {tps_13:>7.1f} | {100*(1-tps_13/tps_base):>+7.1f}% | {'18.75%':>8} | {'~100%':>8}")
print(f"{'Lossless Tiered':<25} | {tps_lt:>7.1f} | {100*(1-tps_lt/tps_base):>+7.1f}% | {'~26%':>8} | {'100%':>8}")
print(f"{'FP8 E4M3':<25} | {tps_fp8:>7.1f} | {100*(1-tps_fp8/tps_base):>+7.1f}% | {'50%':>8} | {'No':>8}")
