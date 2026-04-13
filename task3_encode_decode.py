"""
Task 3: Encode / Decode Implementation for KVFloat13
"""

import numpy as np

def encode_kvf13(bf16_uint16: np.ndarray, compress_lut: np.ndarray) -> tuple:
    """
    Encode BF16 (as uint16 array, length must be multiple of 128)
    into 3 packed streams per block.

    Returns (signs_packed, exp_hi_packed, em_bytes) as flat arrays.
    """
    assert bf16_uint16.dtype == np.uint16
    n = len(bf16_uint16)
    assert n % 128 == 0, f"Length {n} not multiple of 128"
    num_blocks = n // 128

    # Extract fields
    sign = ((bf16_uint16 >> 15) & 1).astype(np.uint8)
    exp8 = ((bf16_uint16 >> 7) & 0xFF).astype(np.uint8)
    mant7 = (bf16_uint16 & 0x7F).astype(np.uint8)

    # Map exp8 → exp5 via LUT
    exp5 = compress_lut[exp8]

    # Split exp5 into high 4 bits and low 1 bit
    exp_hi4 = (exp5 >> 1).astype(np.uint8)
    exp_lo1 = (exp5 & 1).astype(np.uint8)

    # em_byte = (exp_lo1 << 7) | mant7
    em_bytes = ((exp_lo1 << 7) | mant7).astype(np.uint8)

    # Pack signs: 8 bits per byte, 16 bytes per block
    sign_reshaped = sign.reshape(num_blocks, 128)
    signs_packed = np.packbits(sign_reshaped, axis=1, bitorder='little')
    # packbits with little endian: bit 0 of each byte = first value
    # This gives 16 bytes per block

    # Pack exp_hi4: 2 nibbles per byte, 64 bytes per block
    exp_hi_reshaped = exp_hi4.reshape(num_blocks, 128)
    # Pack pairs: even index = low nibble, odd index = high nibble
    even = exp_hi_reshaped[:, 0::2]  # (num_blocks, 64)
    odd = exp_hi_reshaped[:, 1::2]   # (num_blocks, 64)
    exp_hi_packed = (even | (odd << 4)).astype(np.uint8)

    # em_bytes: already 1 byte per value, 128 bytes per block
    em_reshaped = em_bytes.reshape(num_blocks, 128)

    return signs_packed, exp_hi_packed, em_reshaped


def decode_kvf13(signs_packed: np.ndarray, exp_hi_packed: np.ndarray,
                  em: np.ndarray, decompress_lut: np.ndarray) -> np.ndarray:
    """
    Decode 3 streams back to BF16 uint16 array.

    signs_packed: (num_blocks, 16) uint8
    exp_hi_packed: (num_blocks, 64) uint8
    em: (num_blocks, 128) uint8
    decompress_lut: (32,) uint8

    Returns uint16 array (BF16 representation).
    """
    num_blocks = signs_packed.shape[0]
    n = num_blocks * 128

    # Unpack signs: 16 bytes → 128 bits per block
    sign = np.unpackbits(signs_packed, axis=1, bitorder='little')[:, :128]
    sign = sign.astype(np.uint16)

    # Unpack exp_hi4: 64 bytes → 128 nibbles per block
    even = (exp_hi_packed & 0x0F).astype(np.uint16)   # low nibble
    odd = ((exp_hi_packed >> 4) & 0x0F).astype(np.uint16)  # high nibble
    # Interleave: even[i] goes to position 2i, odd[i] to position 2i+1
    exp_hi4 = np.empty((num_blocks, 128), dtype=np.uint16)
    exp_hi4[:, 0::2] = even
    exp_hi4[:, 1::2] = odd

    # Extract from em bytes
    em_uint16 = em.astype(np.uint16)
    exp_lo1 = (em_uint16 >> 7) & 1
    mant7 = em_uint16 & 0x7F

    # Reconstruct exp5
    exp5 = ((exp_hi4 << 1) | exp_lo1).astype(np.uint8)

    # LUT lookup: exp5 → exp8
    exp8 = decompress_lut[exp5].astype(np.uint16)

    # Reassemble BF16
    bf16 = (sign << 15) | (exp8 << 7) | mant7

    return bf16.reshape(n).astype(np.uint16)


def main():
    """Quick self-test."""
    # Build simple test LUTs (identity-ish for exponents 96-127)
    compress_lut = np.zeros(256, dtype=np.uint8)
    decompress_lut = np.arange(96, 128, dtype=np.uint8)
    for i in range(32):
        compress_lut[96 + i] = i

    # Test with known values
    n = 128 * 4  # 4 blocks
    rng = np.random.RandomState(42)
    signs = rng.randint(0, 2, n).astype(np.uint16)
    exps = rng.randint(96, 128, n).astype(np.uint16)
    mants = rng.randint(0, 128, n).astype(np.uint16)
    bf16_orig = (signs << 15) | (exps << 7) | mants

    # Encode
    s_packed, e_packed, em = encode_kvf13(bf16_orig.astype(np.uint16), compress_lut)
    print(f"Encoded {n} values into {s_packed.nbytes + e_packed.nbytes + em.nbytes} bytes")
    print(f"  signs:  {s_packed.shape} = {s_packed.nbytes} bytes")
    print(f"  exp_hi: {e_packed.shape} = {e_packed.nbytes} bytes")
    print(f"  em:     {em.shape} = {em.nbytes} bytes")
    print(f"  BF16 original: {n * 2} bytes")
    print(f"  Compression ratio: {100 * (1 - (s_packed.nbytes + e_packed.nbytes + em.nbytes) / (n * 2)):.2f}%")

    # Decode
    decoded = decode_kvf13(s_packed, e_packed, em, decompress_lut)

    # Verify
    match = np.sum(bf16_orig == decoded)
    print(f"\nBit-exact match: {match}/{n} ({100*match/n:.2f}%)")
    if match == n:
        print("PASS: Perfect round-trip!")
    else:
        mismatches = np.where(bf16_orig != decoded)[0]
        print(f"FAIL: {len(mismatches)} mismatches")
        for idx in mismatches[:10]:
            print(f"  [{idx}] orig=0x{bf16_orig[idx]:04x} decoded=0x{decoded[idx]:04x}")

if __name__ == "__main__":
    main()
