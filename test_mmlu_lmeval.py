"""
MMLU via lm-eval harness on Qwen3-4B.
Tests: BF16 baseline, LUT[117-132], base+offset.
"""
import torch, numpy as np, sys, time, json, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

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

def encode_decode_lut_fast(bf16_uint16, compress_lut, decompress_lut):
    sign = (bf16_uint16 >> 15) & 1
    exp8 = (bf16_uint16 >> 7) & 0xFF
    mant7 = bf16_uint16 & 0x7F
    exp4 = compress_lut[exp8.astype(np.uint8)]
    exp8_new = decompress_lut[exp4].astype(np.uint16)
    return ((sign << 15) | (exp8_new << 7) | mant7).astype(np.uint16)

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
import lm_eval
from lm_eval.models.huggingface import HFLM

model_path = "/root/autodl-tmp/Qwen3-4B"
target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

def compress_weights_lut(model, cl, dl):
    saved = {}
    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        saved[name] = param.data.clone()
        raw = param.data.to(torch.bfloat16).contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
        decoded = encode_decode_lut_fast(raw, cl, dl)
        param.data = torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(param.shape).to(param.device)
    return saved

def compress_weights_bo(model):
    saved = {}
    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        saved[name] = param.data.clone()
        raw = param.data.to(torch.bfloat16).contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
        decoded = encode_decode_bo_fast(raw)
        param.data = torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(param.shape).to(param.device)
    return saved

def restore_weights(model, saved):
    for name, orig in saved.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        getattr(obj, parts[-1]).data = orig

def run_mmlu(model, tokenizer, label):
    print(f"\n  [{label}] Starting lm-eval MMLU...")
    t0 = time.time()
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["mmlu"],
        num_fewshot=5,
    )
    elapsed = time.time() - t0
    acc = results["results"]["mmlu"]["acc,none"]
    stderr = results["results"]["mmlu"].get("acc_stderr,none", 0)
    print(f"  [{label}] MMLU acc={acc:.4f} ± {stderr:.4f} ({elapsed:.0f}s)")
    return acc, stderr, results

# ============================================================

print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

print(f"\n{'='*60}")
print(f"MMLU (lm-eval) — Qwen3-4B")
print(f"{'='*60}")

# 1) Baseline
print("\n1) BF16 baseline:")
acc_base, stderr_base, res_base = run_mmlu(model, tokenizer, "BF16")

# 2) LUT [117-132]
print("\n2) LUT [117-132]:")
cl, dl = build_contiguous_lut(117, 16)
t0 = time.time()
saved = compress_weights_lut(model, cl, dl)
print(f"  Compression: {time.time()-t0:.1f}s")
acc_lut, stderr_lut, res_lut = run_mmlu(model, tokenizer, "LUT[117-132]")
restore_weights(model, saved)

# 3) base+offset
print("\n3) Per-block base+offset:")
t0 = time.time()
saved = compress_weights_bo(model)
print(f"  Compression: {time.time()-t0:.1f}s")
acc_bo, stderr_bo, res_bo = run_mmlu(model, tokenizer, "base+offset")
restore_weights(model, saved)

# Summary
print(f"\n{'='*60}")
print(f"MMLU RESULTS — Qwen3-4B (lm-eval, 5-shot)")
print(f"{'='*60}")
print(f"{'Method':<25} | {'Acc':>7} | {'± stderr':>8} | {'Δ Acc':>7} | {'Compress':>8}")
print(f"{'-'*62}")
print(f"{'BF16 baseline':<25} | {acc_base:>7.4f} | {stderr_base:>8.4f} | {'—':>7} | {'0%':>8}")
print(f"{'LUT [117-132]':<25} | {acc_lut:>7.4f} | {stderr_lut:>8.4f} | {acc_lut-acc_base:>+7.4f} | {'25.0%':>8}")
print(f"{'Per-block base+offset':<25} | {acc_bo:>7.4f} | {stderr_bo:>8.4f} | {acc_bo-acc_base:>+7.4f} | {'18.4%':>8}")

# Save
results_all = {
    "model": "Qwen3-4B",
    "eval": "lm-eval mmlu 5-shot",
    "baseline": {"acc": acc_base, "stderr": stderr_base},
    "lut_117_132": {"acc": acc_lut, "stderr": stderr_lut},
    "base_offset": {"acc": acc_bo, "stderr": stderr_bo},
}
with open("/root/kvfloat13/mmlu_lmeval_results.json", "w") as f:
    json.dump(results_all, f, indent=2)
print("\nSaved to mmlu_lmeval_results.json")
