"""
Fast MMLU eval: direct 4-choice accuracy without lm-eval harness.
Tests BF16 baseline, LUT[117-132], base+offset on Qwen3-4B.
"""
import torch, numpy as np, sys, time, json, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
sys.path.insert(0, '/root/kvfloat13')
from test_qwen3_fast import (encode_decode_lut_fast, encode_decode_bo_fast,
                              build_contiguous_lut)

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

def eval_mmlu(model, tokenizer, dataset, limit=None):
    """
    5-shot MMLU evaluation using log-likelihood of A/B/C/D.
    """
    choices = ["A", "B", "C", "D"]
    choice_tokens = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]

    correct = 0
    total = 0
    subjects = {}

    items = list(dataset)
    if limit:
        items = items[:limit]

    model.eval()
    batch_size = 4

    # Process in batches
    all_prompts = []
    all_answers = []
    all_subjects_list = []

    for item in items:
        question = item["question"]
        choice_list = item["choices"]
        answer_idx = item["answer"]  # 0-3
        subject = item.get("subject", "unknown")

        prompt = f"Question: {question}\n"
        for i, c in enumerate(choice_list):
            prompt += f"{choices[i]}. {c}\n"
        prompt += "Answer:"

        all_prompts.append(prompt)
        all_answers.append(answer_idx)
        all_subjects_list.append(subject)

    # Batch inference
    for batch_start in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[batch_start:batch_start + batch_size]
        batch_answers = all_answers[batch_start:batch_start + batch_size]
        batch_subjects = all_subjects_list[batch_start:batch_start + batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                          max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Get logits at the last non-padding position for each item
            logits = outputs.logits

        for i in range(len(batch_prompts)):
            # Find last non-padding token position
            attn_mask = inputs["attention_mask"][i]
            last_pos = attn_mask.sum() - 1

            token_logits = logits[i, last_pos]
            choice_logits = token_logits[choice_tokens]
            pred = choice_logits.argmax().item()

            is_correct = (pred == batch_answers[i])
            correct += int(is_correct)
            total += 1

            subj = batch_subjects[i]
            if subj not in subjects:
                subjects[subj] = [0, 0]
            subjects[subj][0] += int(is_correct)
            subjects[subj][1] += 1

        if total % 500 == 0 and total > 0:
            print(f"    Progress: {total}/{len(all_prompts)} acc={100*correct/total:.2f}%")

    acc = correct / total if total > 0 else 0
    return acc, correct, total, subjects

# ============================================================

print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()

print("Loading MMLU dataset...")
ds = load_dataset("cais/mmlu", "all", split="test")
print(f"MMLU test set: {len(ds)} questions")

LIMIT = None  # Full eval

print(f"\n{'='*60}")
print(f"MMLU EVALUATION — Qwen3-4B {'(limit='+str(LIMIT)+')' if LIMIT else '(full)'}")
print(f"{'='*60}")

results = {}

# 1) Baseline
print("\n1) BF16 baseline:")
t0 = time.time()
acc_base, c, t, subj_base = eval_mmlu(model, tokenizer, ds, limit=LIMIT)
print(f"  BF16: acc={acc_base:.4f} ({c}/{t}) [{time.time()-t0:.0f}s]")
results["baseline"] = {"acc": acc_base, "correct": c, "total": t}

# 2) LUT [117-132]
print("\n2) LUT [117-132]:")
cl, dl = build_contiguous_lut(117, 16)
t0 = time.time()
saved = compress_weights_lut(model, cl, dl)
print(f"  Compression: {time.time()-t0:.1f}s")
t0 = time.time()
acc_lut, c, t, subj_lut = eval_mmlu(model, tokenizer, ds, limit=LIMIT)
restore_weights(model, saved)
print(f"  LUT[117-132]: acc={acc_lut:.4f} ({c}/{t}) [{time.time()-t0:.0f}s]")
results["lut_117_132"] = {"acc": acc_lut, "correct": c, "total": t}

# 3) base+offset
print("\n3) Per-block base+offset:")
t0 = time.time()
saved = compress_weights_bo(model)
print(f"  Compression: {time.time()-t0:.1f}s")
t0 = time.time()
acc_bo, c, t, subj_bo = eval_mmlu(model, tokenizer, ds, limit=LIMIT)
restore_weights(model, saved)
print(f"  base+offset: acc={acc_bo:.4f} ({c}/{t}) [{time.time()-t0:.0f}s]")
results["base_offset"] = {"acc": acc_bo, "correct": c, "total": t}

# Summary
print(f"\n{'='*60}")
print(f"MMLU RESULTS — Qwen3-4B")
print(f"{'='*60}")
print(f"{'Method':<25} | {'Acc':>7} | {'Δ Acc':>7} | {'Compress':>8}")
print(f"{'-'*55}")
print(f"{'BF16 baseline':<25} | {acc_base:>7.4f} | {'—':>7} | {'0%':>8}")
print(f"{'LUT [117-132]':<25} | {acc_lut:>7.4f} | {acc_lut-acc_base:>+7.4f} | {'25.0%':>8}")
print(f"{'Per-block base+offset':<25} | {acc_bo:>7.4f} | {acc_bo-acc_base:>+7.4f} | {'18.4%':>8}")

# Per-subject comparison (show top diffs)
print(f"\nPer-subject differences (LUT vs baseline, sorted by |diff|):")
diffs = []
for subj in subj_base:
    base_acc = subj_base[subj][0] / subj_base[subj][1] if subj_base[subj][1] > 0 else 0
    lut_acc = subj_lut.get(subj, [0,0])
    lut_acc = lut_acc[0] / lut_acc[1] if lut_acc[1] > 0 else 0
    diffs.append((subj, base_acc, lut_acc, lut_acc - base_acc, subj_base[subj][1]))
diffs.sort(key=lambda x: -abs(x[3]))
for subj, ba, la, d, n in diffs[:10]:
    print(f"  {subj:<40} base={ba:.3f} lut={la:.3f} Δ={d:+.3f} (n={n})")

# Save
results["per_subject_base"] = {k: {"correct": v[0], "total": v[1]} for k, v in subj_base.items()}
with open("/root/kvfloat13/mmlu_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to mmlu_results.json")
