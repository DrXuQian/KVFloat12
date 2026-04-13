"""
Debug: check if KVFloat12 decode produces actual inf/nan values
"""
import torch, numpy as np, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
sys.path.insert(0, '/root/kvfloat13')
from split_lut_kvfloat12 import encode_kvf12, decode_kvf12, build_lut_from_counter, extract_exp8

model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")

target_suffixes = ['q_proj.weight','k_proj.weight','v_proj.weight',
                   'o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight']

w_counter = Counter()
for name, param in model.named_parameters():
    if any(name.endswith(s) for s in target_suffixes):
        w_counter.update(extract_exp8(param.data.to(torch.bfloat16)).tolist())

w_compress, w_decompress = build_lut_from_counter(w_counter, 16)
print(f"Weight LUT: {sorted(w_decompress.tolist())}")

# Check every decoded tensor for inf/nan
print("\n=== Checking decoded tensors for inf/nan ===")
total_inf = 0
total_nan = 0
total_vals = 0

for name, param in model.named_parameters():
    if not any(name.endswith(s) for s in target_suffixes):
        continue

    bf16 = param.data.to(torch.bfloat16)
    shape = bf16.shape
    raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
    n = len(raw)
    pad = (128 - n % 128) % 128
    raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw

    s1, s2 = encode_kvf12(raw_p, w_compress)
    decoded_u16 = decode_kvf12(s1, s2, w_decompress)[:n]

    # Convert to float to check
    decoded_bf16 = torch.from_numpy(decoded_u16.astype(np.int16)).view(torch.bfloat16)
    decoded_f32 = decoded_bf16.float()

    n_inf = torch.isinf(decoded_f32).sum().item()
    n_nan = torch.isnan(decoded_f32).sum().item()
    total_inf += n_inf
    total_nan += n_nan
    total_vals += n

    if n_inf > 0 or n_nan > 0:
        print(f"  {name}: {n_inf} inf, {n_nan} nan out of {n}")

        # Show the original values that became inf/nan
        inf_mask = torch.isinf(decoded_f32)
        nan_mask = torch.isnan(decoded_f32)

        if n_inf > 0:
            orig_at_inf = torch.from_numpy(raw).view(torch.bfloat16)[inf_mask.numpy()].float()
            dec_at_inf_u16 = decoded_u16[inf_mask.numpy()]
            print(f"    Original values at inf positions: {orig_at_inf[:10].tolist()}")
            print(f"    Decoded uint16 at inf positions:  {[f'0x{v:04x}' for v in dec_at_inf_u16[:10]]}")
            # Decode the uint16 manually
            for v in dec_at_inf_u16[:5]:
                s = (v >> 15) & 1
                e = (v >> 7) & 0xFF
                m = v & 0x7F
                print(f"    0x{v:04x}: sign={s} exp={e} mant={m}")

        if n_nan > 0:
            orig_at_nan = torch.from_numpy(raw).view(torch.bfloat16)[nan_mask.numpy()].float()
            dec_at_nan_u16 = decoded_u16[nan_mask.numpy()]
            print(f"    Original values at nan positions: {orig_at_nan[:10].tolist()}")
            print(f"    Decoded uint16 at nan positions:  {[f'0x{v:04x}' for v in dec_at_nan_u16[:10]]}")
            for v in dec_at_nan_u16[:5]:
                s = (v >> 15) & 1
                e = (v >> 7) & 0xFF
                m = v & 0x7F
                print(f"    0x{v:04x}: sign={s} exp={e} mant={m}")

print(f"\nTotal: {total_inf} inf, {total_nan} nan out of {total_vals}")

if total_inf == 0 and total_nan == 0:
    print("No inf/nan found — PPL explosion is from magnitude errors only")

    # Show the worst absolute errors
    print("\n=== Top absolute errors (first 3 layers) ===")
    for name, param in model.named_parameters():
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        layer = name.split('.')[2] if 'layers' in name else '?'
        if int(layer) > 2:
            continue

        bf16 = param.data.to(torch.bfloat16)
        raw = bf16.contiguous().view(-1).view(torch.int16).cpu().numpy().astype(np.uint16)
        n = len(raw)
        pad = (128 - n % 128) % 128
        raw_p = np.concatenate([raw, np.zeros(pad, dtype=np.uint16)]) if pad else raw

        s1, s2 = encode_kvf12(raw_p, w_compress)
        decoded = decode_kvf12(s1, s2, w_decompress)[:n]

        orig_f = torch.from_numpy(raw.astype(np.int16)).view(torch.bfloat16).float().numpy()
        dec_f = torch.from_numpy(decoded.astype(np.int16)).view(torch.bfloat16).float().numpy()

        abs_err = np.abs(orig_f - dec_f)
        worst_idx = np.argsort(abs_err)[-5:][::-1]

        tname = name.rsplit('.', 1)[0].rsplit('.', 1)[-1]
        print(f"\n  L{layer}.{tname} top-5 errors:")
        for idx in worst_idx:
            oe = (raw[idx] >> 7) & 0xFF
            de = (decoded[idx] >> 7) & 0xFF
            print(f"    [{idx}] orig={orig_f[idx]:>12.6f} (exp={oe}) → dec={dec_f[idx]:>12.6f} (exp={de}) err={abs_err[idx]:.6f}")
