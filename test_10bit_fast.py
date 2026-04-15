"""
Fast 10-bit KV cache compression: per-block top-8 exponent, vectorized.
No Python loops over blocks — uses torch vectorized ops.
"""
import torch, numpy as np, sys, os, time, json, math
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset
from collections import Counter

model_path = "/root/autodl-tmp/Qwen3-4B"

def compress_10bit_fast(tensor):
    """
    Per-block top-8 exponent compression.
    For each 128-value block: pick top-8 exponents, map others to nearest.
    Simulates 10-bit format (3-bit exp_idx + 7-bit mant, sign separate).
    """
    shape = tensor.shape
    raw = tensor.contiguous().view(-1).view(torch.int16).to(torch.int32)
    n = raw.shape[0]
    pad = (128 - n % 128) % 128
    if pad:
        raw = torch.cat([raw, torch.zeros(pad, dtype=torch.int32, device=raw.device)])

    sign = (raw >> 15) & 1
    exp8 = (raw >> 7) & 0xFF
    mant7 = raw & 0x7F

    blocks_exp = exp8.reshape(-1, 128)  # [num_blocks, 128]
    num_blocks = blocks_exp.shape[0]

    # For each block, find top-8 exponents
    # Strategy: use the block's max and span to pick a contiguous range of 8
    # This avoids expensive per-block unique/sort in Python
    block_max = blocks_exp.max(dim=1).values  # [num_blocks]
    block_min = blocks_exp.min(dim=1).values

    # Per-block: center the 8-wide window on the distribution
    # Use: base = max(block_min, block_max - 7) to cover the top
    base = torch.clamp(block_max - 7, min=0)  # [num_blocks]

    # Map: offset = exp - base, clamp to [0, 7]
    offsets = blocks_exp - base.unsqueeze(1)  # [num_blocks, 128]
    offsets = offsets.clamp(0, 7)

    # Reconstruct exp
    new_exp = base.unsqueeze(1) + offsets  # [num_blocks, 128]

    # Reassemble
    result = (sign.reshape(-1, 128) << 15) | (new_exp << 7) | mant7.reshape(-1, 128)
    return result.reshape(-1)[:n].to(torch.int16).view(torch.bfloat16).reshape(shape)

def compress_cache_10bit(cache):
    for layer in cache.layers:
        layer.keys = compress_10bit_fast(layer.keys)
        layer.values = compress_10bit_fast(layer.values)

# KVFloat13 for comparison
def make_kvf13_emap(model, tokenizer):
    kv_counter = Counter()
    with torch.no_grad():
        for p in ["The quick brown fox.", "def fib(n): return n", "Neural networks."]:
            inputs = tokenizer(p, return_tensors="pt").to(model.device)
            out = model(**inputs, use_cache=True)
            for layer in out.past_key_values.layers:
                for t in [layer.keys, layer.values]:
                    exps = ((t.view(torch.int16).to(torch.int32) >> 7) & 0xFF)
                    kv_counter.update(exps.cpu().numpy().flatten().tolist())

    top32 = sorted([e for e, _ in kv_counter.most_common(32)])
    emap = torch.zeros(256, dtype=torch.int32, device='cuda')
    compress = np.zeros(256, dtype=np.uint8)
    decompress = np.array(top32, dtype=np.uint8)
    idx = {e: i for i, e in enumerate(top32)}
    for e, i in idx.items():
        compress[e] = i
    for e in range(256):
        if e not in idx:
            best = min(top32, key=lambda se: abs(2.0**(e-127) - 2.0**(se-127)))
            compress[e] = idx[best]
    for e in range(256):
        emap[e] = decompress[compress[e]]
    return emap

def compress_kvf13_gpu(cache, emap):
    for layer in cache.layers:
        for attr in ['keys', 'values']:
            t = getattr(layer, attr)
            raw = t.view(torch.int16).to(torch.int32)
            s = (raw >> 15) & 1
            e = (raw >> 7) & 0xFF
            m = raw & 0x7F
            setattr(layer, attr, ((s << 15) | (emap[e] << 7) | m).to(torch.int16).view(torch.bfloat16))

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
            if compress_fn:
                compress_fn(cache)
            logits = out.logits[:, -1, :]
            target = input_ids[:, t+1]
            loss = torch.nn.functional.cross_entropy(logits, target)
            nlls.append(loss.item())
            if (t+1) % 1000 == 0:
                ppl = math.exp(sum(nlls) / len(nlls))
                tps = (t+1) / (time.time() - t0)
                print(f"    [{t+1}/{seq_len}] ppl={ppl:.4f} ({time.time()-t0:.0f}s, {tps:.0f} tok/s)")
    return math.exp(sum(nlls) / len(nlls)), time.time() - t0

# ============================================================
print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

kvf13_emap = make_kvf13_emap(model, tokenizer)

print("\nLoading Wikitext-2...")
try:
    ds = load_dataset("/root/autodl-tmp/data/datasets/wikitext", "wikitext-2-raw-v1", split="test")
except:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
input_ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids.to(model.device)

MAX = 4096
print(f"\n{'='*65}")
print(f"DECODE PPL — 10-bit vs KVFloat13 vs FP8 ({MAX} tokens)")
print(f"{'='*65}")

# 1) Baseline
print("\n1) BF16:")
ppl_base, t = eval_decode_ppl(model, input_ids, max_tokens=MAX)
print(f"  PPL={ppl_base:.4f} ({t:.0f}s)")
torch.cuda.empty_cache()

# 2) KVFloat13
print("\n2) KVFloat13 (13-bit, 18.75%):")
ppl_13, t = eval_decode_ppl(model, input_ids, lambda c: compress_kvf13_gpu(c, kvf13_emap), MAX)
print(f"  PPL={ppl_13:.4f} (Δ{ppl_13-ppl_base:+.4f}, {t:.0f}s)")
torch.cuda.empty_cache()

# 3) 10-bit per-block (fast)
print("\n3) 10-bit per-block top-8 (~28%):")
ppl_10, t = eval_decode_ppl(model, input_ids, compress_cache_10bit, MAX)
print(f"  PPL={ppl_10:.4f} (Δ{ppl_10-ppl_base:+.4f}, {t:.0f}s)")
torch.cuda.empty_cache()

# 4) FP8
print("\n4) FP8 E4M3 (8-bit, 50%):")
def compress_fp8(c):
    for l in c.layers:
        l.keys = l.keys.to(torch.float8_e4m3fn).to(torch.bfloat16)
        l.values = l.values.to(torch.float8_e4m3fn).to(torch.bfloat16)
ppl_fp8, t = eval_decode_ppl(model, input_ids, compress_fp8, MAX)
print(f"  PPL={ppl_fp8:.4f} (Δ{ppl_fp8-ppl_base:+.4f}, {t:.0f}s)")

# Summary
print(f"\n{'='*65}")
print(f"RESULTS")
print(f"{'='*65}")
print(f"{'Method':<35} | {'bits':>4} | {'Compress':>8} | {'PPL':>8} | {'Δ PPL':>8}")
print("-" * 70)
print(f"{'BF16':<35} | {'16':>4} | {'0%':>8} | {ppl_base:>8.4f} | {'—':>8}")
print(f"{'KVFloat13 (5-bit LUT)':<35} | {'13':>4} | {'18.75%':>8} | {ppl_13:>8.4f} | {ppl_13-ppl_base:>+8.4f}")
print(f"{'10-bit (3-bit/block base+off)':<35} | {'10':>4} | {'~28%':>8} | {ppl_10:>8.4f} | {ppl_10-ppl_base:>+8.4f}")
print(f"{'FP8 E4M3':<35} | {'8':>4} | {'50%':>8} | {ppl_fp8:>8.4f} | {ppl_fp8-ppl_base:>+8.4f}")

with open("/root/kvfloat13/zipserv_results.json", "w") as f:
    json.dump({"model": "Qwen3-4B", "tokens": MAX, "baseline": ppl_base,
               "kvfloat13": ppl_13, "packed_10bit": ppl_10, "fp8": ppl_fp8}, f, indent=2)
print("\nSaved")
