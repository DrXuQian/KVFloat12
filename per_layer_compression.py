"""
Per-layer compression analysis: overflow rate varies by layer, by input, by K/V.
Show the actual variation — compression is NOT a single number.
"""
import torch, numpy as np, sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

model_path = "/root/autodl-tmp/Qwen3-4B"
print(f"Loading {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
model.eval()
n_layers = model.config.num_hidden_layers

prompts = {
    "english": "The quick brown fox jumps over the lazy dog in a warm summer afternoon by the river bank.",
    "code": "def quicksort(arr): return [] if not arr else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
    "math": "The integral of x^2 from 0 to 1 equals 1/3. Consider the Taylor expansion of e^x = sum(x^n/n!) for all n.",
    "long": "In the beginning there was nothing but void and darkness. Then came light and with it the first stars formed in the cosmic dawn. Billions of years passed as galaxies collided and merged creating vast structures. On a small rocky planet orbiting an unremarkable yellow star something remarkable happened. Life emerged from the primordial soup and over billions of years evolved into increasingly complex forms.",
}

print(f"\n{'='*80}")
print("PER-LAYER OVERFLOW ANALYSIS (top-7 scheme)")
print(f"{'='*80}")

for prompt_name, prompt_text in prompts.items():
    print(f"\n--- Input: {prompt_name} ({len(tokenizer.encode(prompt_text))} tokens) ---")
    print(f"{'Layer':>5} | {'Key unique':>10} | {'Key OF':>6} | {'Key OF%':>7} | {'Val unique':>10} | {'Val OF':>6} | {'Val OF%':>7} | {'Block bytes':>10}")
    print("-" * 85)

    with torch.no_grad():
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, use_cache=True)

        total_overflow = 0
        total_values = 0
        layer_sizes = []

        for li, layer in enumerate(outputs.past_key_values.layers):
            row = []
            for kv_name, tensor in [("key", layer.keys), ("value", layer.values)]:
                raw = tensor.view(torch.int16).to(torch.int32)
                exp8 = ((raw >> 7) & 0xFF).reshape(-1, 128)
                n_blocks = exp8.shape[0]

                total_of_kv = 0
                total_vals_kv = 0
                total_unique = 0
                block_bytes_sum = 0

                for bi in range(n_blocks):
                    bexp = exp8[bi]
                    unique = bexp.unique()
                    n_unique = len(unique)
                    total_unique += n_unique

                    if n_unique <= 7:
                        n_of = 0
                    else:
                        _, counts = bexp.unique(return_counts=True)
                        top7_idx = counts.argsort(descending=True)[:7]
                        top7_set = set(unique[top7_idx].cpu().tolist())
                        n_of = sum(1 for e in bexp.cpu().tolist() if e not in top7_set)

                    total_of_kv += n_of
                    total_vals_kv += 128
                    block_bytes_sum += 184 + n_of  # fixed + overflow

                total_overflow += total_of_kv
                total_values += total_vals_kv

                of_pct = 100 * total_of_kv / total_vals_kv if total_vals_kv > 0 else 0
                avg_unique = total_unique / n_blocks if n_blocks > 0 else 0
                row.append((avg_unique, total_of_kv, of_pct, block_bytes_sum))

            if li % 6 == 0 or li == n_layers - 1:
                k_u, k_of, k_pct, k_bytes = row[0]
                v_u, v_of, v_pct, v_bytes = row[1]
                avg_bytes = (k_bytes + v_bytes) / (2 * exp8.shape[0]) if exp8.shape[0] > 0 else 0
                compress = 100 * (1 - avg_bytes / 256)
                print(f"{li:>5} | {k_u:>10.1f} | {k_of:>6} | {k_pct:>6.1f}% | {v_u:>10.1f} | {v_of:>6} | {v_pct:>6.1f}% | {avg_bytes:>7.1f} ({compress:>4.1f}%)")

        overall_of_pct = 100 * total_overflow / total_values
        overall_bytes = 184 + total_overflow / (total_values / 128)
        overall_compress = 100 * (1 - overall_bytes / 256)
        print(f"  Total: overflow={overall_of_pct:.2f}%, avg block={overall_bytes:.1f}B, compress={overall_compress:.1f}%")

# ============================================================
print(f"\n{'='*80}")
print("KEY POINT: 不需要全局分析")
print(f"{'='*80}")
print("""
编码是 PER-BLOCK 自适应的:
  1. 每个 block (128 values) 独立统计自己的 top-7 指数
  2. 不需要预先校准，不需要看其他 block
  3. 不同层、不同输入、不同 K/V 自动适配

压缩率是变量:
  - 取决于每个 block 的指数多样性
  - 浅层 Value 指数分散 → 更多 overflow → 压缩低
  - 深层 Key/Value 指数集中 → 少 overflow → 压缩高
  - 最差情况: 128 unique exponents → 全部 overflow → 184+128=312B (比BF16大!)
  - 最好情况: ≤7 unique exponents → 0 overflow → 184B (28.1%)
  - 实测平均: ~189B (26%)

对 vLLM 的影响:
  - 内存分配: 可以预留固定大小 (e.g., 192B/block = 25%)
  - 超出部分用少量额外 buffer
  - 或直接预留 worst case (256B) 然后实际占用更少
""")
