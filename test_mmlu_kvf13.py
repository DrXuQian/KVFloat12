"""
MMLU via lm-eval: KVFloat13 (5-bit, 32-entry global LUT) on Qwen3-4B.
Should be FULLY LOSSLESS if top-32 coverage = 100%.
"""
import torch, numpy as np, sys, time, json, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import lm_eval
from lm_eval.models.huggingface import HFLM

model_path = "/root/autodl-tmp/Qwen3-4B"
target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

def build_kvf13_luts(model):
    """Build 5-bit LUT from actual model weight exponents."""
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

    decompress = np.array(top32, dtype=np.uint8)
    compress = np.zeros(256, dtype=np.uint8)
    exp_to_idx = {e: i for i, e in enumerate(top32)}
    for e, i in exp_to_idx.items():
        compress[e] = i
    supported = set(top32)
    for e in range(256):
        if e in supported:
            continue
        best_i, best_d = 0, float('inf')
        for se in top32:
            d = abs(2.0**(e-127) - 2.0**(se-127))
            if d < best_d:
                best_d, best_i = d, exp_to_idx[se]
        compress[e] = best_i

    cov = 100 * sum(counter[e] for e in top32) / total
    print(f"  KVFloat13 LUT: {top32}")
    print(f"  Coverage: {cov:.6f}% ({len(counter)} unique exponents)")
    return compress, decompress, cov

def compress_weights_kvf13(model, cl, dl):
    saved = {}
    total_match = 0
    total_vals = 0
    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        saved[name] = param.data.clone()
        raw = param.data.to(torch.bfloat16).contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
        n = len(raw)

        sign = (raw >> 15) & 1
        exp8 = (raw >> 7) & 0xFF
        mant7 = raw & 0x7F
        exp5 = cl[exp8.astype(np.uint8)]
        exp8_new = dl[exp5].astype(np.uint16)
        decoded = ((sign << 15) | (exp8_new << 7) | mant7).astype(np.uint16)

        total_match += int(np.sum(raw == decoded))
        total_vals += n

        param.data = torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).reshape(param.shape).to(param.device)

    print(f"  Bit-exact: {total_match:,}/{total_vals:,} ({100*total_match/total_vals:.6f}%)")
    return saved

def restore_weights(model, saved):
    for name, orig in saved.items():
        parts = name.split('.')
        obj = model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        getattr(obj, parts[-1]).data = orig

# ============================================================

print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

print("\nBuilding KVFloat13 LUT...")
cl, dl, cov = build_kvf13_luts(model)

print("\nCompressing weights...")
t0 = time.time()
saved = compress_weights_kvf13(model, cl, dl)
print(f"  Compression time: {time.time()-t0:.1f}s")

print("\nRunning lm-eval MMLU (5-shot, batch=4)...")
t0 = time.time()
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
results = lm_eval.simple_evaluate(model=lm, tasks=["mmlu"], num_fewshot=5)
elapsed = time.time() - t0

acc = results["results"]["mmlu"]["acc,none"]
stderr = results["results"]["mmlu"].get("acc_stderr,none", 0)
print(f"\n  KVFloat13 MMLU acc={acc:.4f} ± {stderr:.4f} ({elapsed:.0f}s)")

restore_weights(model, saved)

# Save
with open("/root/kvfloat13/mmlu_kvf13_results.json", "w") as f:
    json.dump({"model": "Qwen3-4B", "method": "KVFloat13 5-bit",
               "coverage": cov, "acc": acc, "stderr": stderr}, f, indent=2)
print("Saved to mmlu_kvf13_results.json")
