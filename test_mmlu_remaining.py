"""
Run remaining MMLU tests: base+offset and KVFloat13 (5-bit).
Previous results: BF16=0.7012, LUT[117-132]=0.6985.
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

def encode_decode_bo_fast(bf16_uint16):
    n = len(bf16_uint16)
    pad = (128 - n % 128) % 128
    if pad:
        bf16_uint16 = np.concatenate([bf16_uint16, np.zeros(pad, dtype=np.uint16)])
    sign = (bf16_uint16 >> 15) & 1
    exp8 = ((bf16_uint16 >> 7) & 0xFF).astype(np.int16)
    mant7 = bf16_uint16 & 0x7F
    blocks_exp = exp8.reshape(-1, 128)
    blocks_sign = sign.reshape(-1, 128)
    blocks_mant = mant7.reshape(-1, 128)
    maxes = blocks_exp.max(axis=1)
    bases = np.maximum(0, maxes - 15).astype(np.int16)
    offsets = np.clip(blocks_exp - bases[:, None], 0, 15)
    exp8_new = (bases[:, None] + offsets).astype(np.uint16)
    result = (blocks_sign.astype(np.uint16) << 15) | (exp8_new << 7) | blocks_mant.astype(np.uint16)
    return result.reshape(-1)[:n]

def compress_inplace(model, compress_fn):
    """Compress weights IN-PLACE (no saved copy to save memory)."""
    total_match = 0
    total_vals = 0
    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        raw = param.data.to(torch.bfloat16).contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
        n = len(raw)
        decoded = compress_fn(raw)
        total_match += int(np.sum(raw == decoded))
        total_vals += n
        param.data = torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(param.shape).to(param.device)
    print(f"  Bit-exact: {total_match:,}/{total_vals:,} ({100*total_match/total_vals:.6f}%)")

# ============================================================
print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

# ============================================================
# Test 1: base+offset (compress in-place, no backup to save VRAM)
# ============================================================
print(f"\n{'='*60}")
print("base+offset (in-place compression)")
print(f"{'='*60}")

t0 = time.time()
compress_inplace(model, encode_decode_bo_fast)
print(f"  Compression: {time.time()-t0:.1f}s")

gc.collect()
torch.cuda.empty_cache()

t0 = time.time()
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
results_bo = lm_eval.simple_evaluate(model=lm, tasks=["mmlu"], num_fewshot=5)
elapsed = time.time() - t0
acc_bo = results_bo["results"]["mmlu"]["acc,none"]
stderr_bo = results_bo["results"]["mmlu"].get("acc_stderr,none", 0)
print(f"  base+offset MMLU acc={acc_bo:.4f} ± {stderr_bo:.4f} ({elapsed:.0f}s)")
del lm
gc.collect()
torch.cuda.empty_cache()

# Reload model for KVFloat13 test
del model
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# Test 2: KVFloat13 (5-bit, should be 100% lossless)
# ============================================================
print(f"\n{'='*60}")
print("KVFloat13 (5-bit, 32-entry LUT)")
print(f"{'='*60}")

print("Reloading model...")
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

# Build LUT from model weights
counter = Counter()
for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue
    exps = ((param.data.to(torch.bfloat16).view(torch.int16).to(torch.int32) >> 7) & 0xFF)
    unique, counts = exps.unique(return_counts=True)
    for e, c in zip(unique.cpu().numpy(), counts.cpu().numpy()):
        counter[int(e)] += int(c)

total = sum(counter.values())
top32 = sorted([e for e, _ in counter.most_common(32)])
cov = 100 * sum(counter[e] for e in top32) / total
print(f"  Top-32 exponents: {top32}")
print(f"  Coverage: {cov:.6f}%")

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

def kvf13_compress(raw):
    sign = (raw >> 15) & 1
    exp8 = (raw >> 7) & 0xFF
    mant7 = raw & 0x7F
    exp5 = compress[exp8.astype(np.uint8)]
    exp8_new = decompress[exp5].astype(np.uint16)
    return ((sign << 15) | (exp8_new << 7) | mant7).astype(np.uint16)

t0 = time.time()
compress_inplace(model, kvf13_compress)
print(f"  Compression: {time.time()-t0:.1f}s")

gc.collect()
torch.cuda.empty_cache()

t0 = time.time()
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
results_13 = lm_eval.simple_evaluate(model=lm, tasks=["mmlu"], num_fewshot=5)
elapsed = time.time() - t0
acc_13 = results_13["results"]["mmlu"]["acc,none"]
stderr_13 = results_13["results"]["mmlu"].get("acc_stderr,none", 0)
print(f"  KVFloat13 MMLU acc={acc_13:.4f} ± {stderr_13:.4f} ({elapsed:.0f}s)")

# ============================================================
print(f"\n{'='*60}")
print(f"COMPLETE MMLU RESULTS — Qwen3-4B (lm-eval, 5-shot)")
print(f"{'='*60}")
print(f"{'Method':<25} | {'Acc':>7} | {'± stderr':>8} | {'Δ Acc':>7} | {'Compress':>8}")
print(f"{'-'*62}")
print(f"{'BF16 baseline':<25} | {'0.7012':>7} | {'0.0037':>8} | {'—':>7} | {'0%':>8}")
print(f"{'KVFloat13 (5-bit)':<25} | {acc_13:>7.4f} | {stderr_13:>8.4f} | {acc_13-0.7012:>+7.4f} | {'18.75%':>8}")
print(f"{'LUT [117-132] (4-bit)':<25} | {'0.6985':>7} | {'0.0037':>8} | {'-0.0027':>7} | {'25.0%':>8}")
print(f"{'base+offset (4-bit)':<25} | {acc_bo:>7.4f} | {stderr_bo:>8.4f} | {acc_bo-0.7012:>+7.4f} | {'18.4%':>8}")

with open("/root/kvfloat13/mmlu_lmeval_results.json", "w") as f:
    json.dump({
        "model": "Qwen3-4B", "eval": "lm-eval mmlu 5-shot",
        "baseline": {"acc": 0.7012, "stderr": 0.0037},
        "kvfloat13_5bit": {"acc": acc_13, "stderr": stderr_13, "coverage": cov},
        "lut_117_132": {"acc": 0.6985, "stderr": 0.0037},
        "base_offset": {"acc": acc_bo, "stderr": stderr_bo},
    }, f, indent=2)
print("\nSaved to mmlu_lmeval_results.json")
