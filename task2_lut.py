"""
Task 2: LUT Construction for KVFloat13
Builds compress and decompress lookup tables from exponent frequency data.
"""

import numpy as np
import json

def build_kvf13_luts(exponent_frequencies: dict) -> tuple:
    """
    Build compress_lut (256→5-bit) and decompress_lut (5-bit→8-bit).

    For top-32 exponents: direct bijective mapping.
    For others: map to nearest supported exponent (minimize |2^orig - 2^nearest|).
    """
    # Sort by frequency, take top 32
    sorted_exps = sorted(exponent_frequencies.items(), key=lambda x: -x[1])
    top32_exp8 = [int(e) for e, _ in sorted_exps[:32]]

    # Sort the top-32 by exponent value for monotonic mapping
    top32_sorted = sorted(top32_exp8)

    # Build decompress LUT: exp5 → exp8
    decompress_lut = np.array(top32_sorted, dtype=np.uint8)

    # Build compress LUT: exp8 → exp5
    compress_lut = np.zeros(256, dtype=np.uint8)

    # Direct mapping for supported exponents
    exp8_to_exp5 = {exp8: exp5 for exp5, exp8 in enumerate(top32_sorted)}
    for exp8, exp5 in exp8_to_exp5.items():
        compress_lut[exp8] = exp5

    # For unsupported exponents: find nearest supported
    supported_set = set(top32_sorted)
    for exp8 in range(256):
        if exp8 in supported_set:
            continue
        # Find nearest supported exponent minimizing |2^(exp8-127) - 2^(nearest-127)|
        # = minimizing |2^exp8 - 2^nearest| which for integers means closest exp value
        # But properly: for exp values, the "nearest in value" can differ from "nearest in index"
        # Since 2^x is monotonic, the nearest in absolute float value is simply
        # the closest exponent value (higher exponent = larger value, exponentially)
        # Actually we should check both neighbors:
        best_exp5 = 0
        best_dist = float('inf')
        for supported_exp8 in top32_sorted:
            # Compare actual float magnitudes: 2^(exp-127)
            dist = abs(2.0 ** (exp8 - 127) - 2.0 ** (supported_exp8 - 127))
            if dist < best_dist:
                best_dist = dist
                best_exp5 = exp8_to_exp5[supported_exp8]
        compress_lut[exp8] = best_exp5

    return compress_lut, decompress_lut

def main():
    with open('/root/kvfloat13/exponent_frequencies.json') as f:
        freq_data = json.load(f)

    # Convert string keys to int
    exponent_frequencies = {int(k): v for k, v in freq_data.items()}

    compress_lut, decompress_lut = build_kvf13_luts(exponent_frequencies)

    print("=" * 60)
    print("KVFloat13 LUT Construction")
    print("=" * 60)

    print(f"\nDecompress LUT (exp5 → exp8), {len(decompress_lut)} entries:")
    for i, exp8 in enumerate(decompress_lut):
        print(f"  exp5={i:>2d} → exp8={exp8:>3d} (2^{exp8-127:>+4d})")

    print(f"\nCompress LUT sample (unsupported exponent mappings):")
    supported = set(decompress_lut.tolist())
    for exp8 in range(256):
        if exp8 not in supported:
            target_exp8 = decompress_lut[compress_lut[exp8]]
            print(f"  exp8={exp8:>3d} → exp5={compress_lut[exp8]:>2d} → exp8={target_exp8:>3d}")

    # Verify round-trip for supported exponents
    errors = 0
    for exp5 in range(32):
        exp8 = decompress_lut[exp5]
        if compress_lut[exp8] != exp5:
            print(f"  ERROR: exp5={exp5} → exp8={exp8} → exp5={compress_lut[exp8]}")
            errors += 1
    print(f"\nRound-trip verification: {'PASS' if errors == 0 else f'FAIL ({errors} errors)'}")

    # Save LUTs
    np.save('/root/kvfloat13/compress_lut.npy', compress_lut)
    np.save('/root/kvfloat13/decompress_lut.npy', decompress_lut)
    print("Saved: compress_lut.npy, decompress_lut.npy")

if __name__ == "__main__":
    main()
