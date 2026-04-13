/**
 * Task 6: CPU NEON / AVX2 Decode Kernel for KVFloat13
 *
 * ARM NEON: Primary target for edge deployment (Apple Silicon, Snapdragon)
 * x86 AVX2: Secondary target for desktop/server CPU inference
 */

#include <stdint.h>
#include <string.h>

// Block layout (same as CUDA)
typedef struct {
    uint8_t  signs[16];
    uint8_t  exp_hi[64];
    uint8_t  exp_lo_mant[128];
} block_kvf13;

// ============================================================
// ARM NEON implementation
// ============================================================
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

void kvf13_decode_128_neon(
    const block_kvf13* block,
    const uint8_t* exp_lut,    // 32-entry decompress LUT
    uint16_t* bf16_out         // 128 BF16 values as uint16
) {
    // Load decompress LUT into NEON registers (32 entries = 2 × 16)
    const uint8x16_t lut_lo = vld1q_u8(exp_lut);        // entries 0-15
    const uint8x16_t lut_hi = vld1q_u8(exp_lut + 16);   // entries 16-31

    // Process 16 values per iteration (8 iterations for 128 values)
    for (int i = 0; i < 128; i += 16) {
        // --- Load 16 em bytes ---
        uint8x16_t em_vec = vld1q_u8(block->exp_lo_mant + i);

        // Extract exp_lo (top bit of each em byte)
        uint8x16_t exp_lo = vshrq_n_u8(em_vec, 7);

        // Extract mantissa (bottom 7 bits)
        uint8x16_t mant = vandq_u8(em_vec, vdupq_n_u8(0x7F));

        // --- Load and unpack 8 exp_hi nibble bytes → 16 values ---
        uint8x8_t exp_hi_packed = vld1_u8(block->exp_hi + i / 2);

        // Unpack low nibbles (even indices)
        uint8x8_t lo_nibs = vand_u8(exp_hi_packed, vdup_n_u8(0x0F));
        // Unpack high nibbles (odd indices)
        uint8x8_t hi_nibs = vshr_n_u8(exp_hi_packed, 4);

        // Interleave: [lo0, hi0, lo1, hi1, ...] → 16 values
        uint8x8x2_t zipped = vzip_u8(lo_nibs, hi_nibs);
        uint8x16_t exp_h4 = vcombine_u8(zipped.val[0], zipped.val[1]);

        // --- Reconstruct exp5 = (exp_h4 << 1) | exp_lo ---
        uint8x16_t exp5 = vorrq_u8(vshlq_n_u8(exp_h4, 1), exp_lo);

        // --- LUT lookup: exp5 → exp8 ---
        // vtbl handles 16-entry tables; for 32 entries we use the range:
        // if exp5 < 16: use lut_lo, else: use lut_hi with (exp5 - 16)
        // vqtbl1q handles out-of-range by returning 0, so we combine:
        uint8x16_t exp8_lo = vqtbl1q_u8(lut_lo, exp5);
        uint8x16_t exp5_minus16 = vsubq_u8(exp5, vdupq_n_u8(16));
        uint8x16_t exp8_hi = vqtbl1q_u8(lut_hi, exp5_minus16);
        // Merge: if exp5 >= 16, exp8_lo is 0 and exp8_hi has the value
        uint8x16_t exp8 = vorrq_u8(exp8_lo, exp8_hi);

        // --- Load sign bits ---
        // Signs for positions i..i+15 are at sign bytes i/8 and i/8+1
        // Bit k of sign_byte j covers position j*8+k
        uint8_t sign_bytes[2];
        sign_bytes[0] = block->signs[i / 8];
        sign_bytes[1] = block->signs[i / 8 + 1];
        uint16_t sign_bits = sign_bytes[0] | ((uint16_t)sign_bytes[1] << 8);

        // Expand sign bits to per-element bytes (0 or 1)
        uint8_t sign_array[16];
        for (int j = 0; j < 16; j++) {
            sign_array[j] = (sign_bits >> j) & 1;
        }
        uint8x16_t sign_vec = vld1q_u8(sign_array);

        // --- Assemble BF16: (sign << 15) | (exp8 << 7) | mant ---
        // Work in uint16 pairs
        // Lower 8 values
        uint16x8_t sign_lo = vmovl_u8(vget_low_u8(sign_vec));
        uint16x8_t exp8_lo_w = vmovl_u8(vget_low_u8(exp8));
        uint16x8_t mant_lo = vmovl_u8(vget_low_u8(mant));

        uint16x8_t bf16_lo = vorrq_u16(
            vorrq_u16(vshlq_n_u16(sign_lo, 15), vshlq_n_u16(exp8_lo_w, 7)),
            mant_lo
        );

        // Upper 8 values
        uint16x8_t sign_hi_w = vmovl_u8(vget_high_u8(sign_vec));
        uint16x8_t exp8_hi_w = vmovl_u8(vget_high_u8(exp8));
        uint16x8_t mant_hi = vmovl_u8(vget_high_u8(mant));

        uint16x8_t bf16_hi = vorrq_u16(
            vorrq_u16(vshlq_n_u16(sign_hi_w, 15), vshlq_n_u16(exp8_hi_w, 7)),
            mant_hi
        );

        vst1q_u16(bf16_out + i, bf16_lo);
        vst1q_u16(bf16_out + i + 8, bf16_hi);
    }
}

// Fused decode + dot product (NEON)
float kvf13_dot_128_neon(
    const block_kvf13* block,
    const uint16_t* x_bf16,    // 128 BF16 input values
    const uint8_t* exp_lut
) {
    uint16_t decoded[128];
    kvf13_decode_128_neon(block, exp_lut, decoded);

    // BF16 dot product using NEON float conversion
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (int i = 0; i < 128; i += 16) {
        // Convert BF16 to FP32 by left-shifting 16 bits
        // Load 4 BF16 values, expand to FP32
        for (int j = 0; j < 4; j++) {
            uint16x4_t d_bf16 = vld1_u16(decoded + i + j * 4);
            uint16x4_t x_bf16_v = vld1_u16(x_bf16 + i + j * 4);

            // BF16→FP32: reinterpret (bf16 << 16) as float32
            uint32x4_t d_u32 = vshll_n_u16(d_bf16, 16);
            uint32x4_t x_u32 = vshll_n_u16(x_bf16_v, 16);

            float32x4_t d_f32 = vreinterpretq_f32_u32(d_u32);
            float32x4_t x_f32 = vreinterpretq_f32_u32(x_u32);

            switch (j) {
                case 0: sum0 = vfmaq_f32(sum0, d_f32, x_f32); break;
                case 1: sum1 = vfmaq_f32(sum1, d_f32, x_f32); break;
                case 2: sum2 = vfmaq_f32(sum2, d_f32, x_f32); break;
                case 3: sum3 = vfmaq_f32(sum3, d_f32, x_f32); break;
            }
        }
    }

    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    return vaddvq_f32(sum0);
}
#endif  // ARM NEON

// ============================================================
// x86 AVX2 implementation
// ============================================================
#if defined(__AVX2__)
#include <immintrin.h>

void kvf13_decode_128_avx2(
    const block_kvf13* block,
    const uint8_t* exp_lut,    // 32-entry decompress LUT
    uint16_t* bf16_out         // 128 BF16 values
) {
    // Load LUT into YMM register for vpshufb (only 32 entries, fits in one YMM)
    // vpshufb operates on 128-bit lanes independently, so we need the LUT in both lanes
    __m128i lut_lo = _mm_loadu_si128((const __m128i*)exp_lut);       // entries 0-15
    __m128i lut_hi = _mm_loadu_si128((const __m128i*)(exp_lut + 16));// entries 16-31

    // Process 32 values per iteration (4 iterations for 128 values)
    for (int i = 0; i < 128; i += 32) {
        // --- Load 32 em bytes ---
        __m256i em_vec = _mm256_loadu_si256((const __m256i*)(block->exp_lo_mant + i));

        // exp_lo = em >> 7
        __m256i exp_lo = _mm256_srli_epi16(em_vec, 7);
        exp_lo = _mm256_and_si256(exp_lo, _mm256_set1_epi8(0x01));

        // mant = em & 0x7F
        __m256i mant = _mm256_and_si256(em_vec, _mm256_set1_epi8(0x7F));

        // --- Load and unpack 16 exp_hi nibble bytes → 32 values ---
        __m128i exp_hi_packed = _mm_loadu_si128((const __m128i*)(block->exp_hi + i / 2));

        // Low nibbles (even indices)
        __m128i lo_nibs = _mm_and_si128(exp_hi_packed, _mm_set1_epi8(0x0F));
        // High nibbles (odd indices)
        __m128i hi_nibs = _mm_srli_epi16(exp_hi_packed, 4);
        hi_nibs = _mm_and_si128(hi_nibs, _mm_set1_epi8(0x0F));

        // Interleave
        __m128i interleaved_lo = _mm_unpacklo_epi8(lo_nibs, hi_nibs);
        __m128i interleaved_hi = _mm_unpackhi_epi8(lo_nibs, hi_nibs);
        __m256i exp_h4 = _mm256_set_m128i(interleaved_hi, interleaved_lo);

        // exp5 = (exp_h4 << 1) | exp_lo
        __m256i exp5 = _mm256_or_si256(
            _mm256_and_si256(_mm256_slli_epi16(exp_h4, 1), _mm256_set1_epi8(0x3E)),
            exp_lo
        );

        // --- LUT lookup via pshufb ---
        // Split into <16 and >=16 ranges
        __m128i exp5_lo128 = _mm256_castsi256_si128(exp5);
        __m128i exp5_hi128 = _mm256_extracti128_si256(exp5, 1);

        // For entries 0-15: pshufb with lut_lo (out-of-range gives 0 via masking)
        __m128i result_lo_a = _mm_shuffle_epi8(lut_lo, exp5_lo128);
        __m128i result_lo_b = _mm_shuffle_epi8(lut_lo, exp5_hi128);

        // For entries 16-31: subtract 16, pshufb with lut_hi
        __m128i exp5_m16_lo = _mm_sub_epi8(exp5_lo128, _mm_set1_epi8(16));
        __m128i exp5_m16_hi = _mm_sub_epi8(exp5_hi128, _mm_set1_epi8(16));
        __m128i result_hi_a = _mm_shuffle_epi8(lut_hi, exp5_m16_lo);
        __m128i result_hi_b = _mm_shuffle_epi8(lut_hi, exp5_m16_hi);

        // Mask: if exp5 >= 16, use hi result; else use lo result
        __m128i mask_lo = _mm_cmpgt_epi8(exp5_lo128, _mm_set1_epi8(15));
        __m128i mask_hi = _mm_cmpgt_epi8(exp5_hi128, _mm_set1_epi8(15));

        __m128i exp8_lo128 = _mm_blendv_epi8(result_lo_a, result_hi_a, mask_lo);
        __m128i exp8_hi128 = _mm_blendv_epi8(result_lo_b, result_hi_b, mask_hi);

        __m256i exp8 = _mm256_set_m128i(exp8_hi128, exp8_lo128);

        // --- Load sign bits ---
        // 32 sign bits = 4 bytes
        uint32_t sign_word;
        memcpy(&sign_word, &block->signs[i / 8], 4);

        // Expand 32 bits to 32 bytes (0x00 or 0x01)
        __m256i sign_expanded = _mm256_set1_epi32((int)sign_word);
        __m256i bit_indices = _mm256_setr_epi8(
            0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15,
            16,17,18,19,20,21,22,23, 24,25,26,27,28,29,30,31
        );
        // Create bitmask for each position
        __m256i bit_select = _mm256_sllv_epi32(
            _mm256_set1_epi32(1),
            _mm256_and_si256(bit_indices, _mm256_set1_epi8(0x1F))
        );
        // Actually, simpler approach: broadcast and shift
        // For each byte position j, we want (sign_word >> j) & 1
        uint8_t sign_array[32];
        for (int j = 0; j < 32; j++) {
            sign_array[j] = (sign_word >> j) & 1;
        }
        __m256i sign_vec = _mm256_loadu_si256((const __m256i*)sign_array);

        // --- Assemble BF16 in 16-bit: (sign << 15) | (exp8 << 7) | mant ---
        // Process in two halves of 16 values each (low 128-bit and high 128-bit)
        for (int half = 0; half < 2; half++) {
            __m128i s8, e8, m8;
            if (half == 0) {
                s8 = _mm256_castsi256_si128(sign_vec);
                e8 = _mm256_castsi256_si128(exp8);
                m8 = _mm256_castsi256_si128(mant);
            } else {
                s8 = _mm256_extracti128_si256(sign_vec, 1);
                e8 = _mm256_extracti128_si256(exp8, 1);
                m8 = _mm256_extracti128_si256(mant, 1);
            }

            // Widen to 16-bit (lower 8 bytes → 8 uint16)
            __m256i s16 = _mm256_cvtepu8_epi16(s8);
            __m256i e16 = _mm256_cvtepu8_epi16(e8);
            __m256i m16 = _mm256_cvtepu8_epi16(m8);

            __m256i bf16 = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_slli_epi16(s16, 15),
                    _mm256_slli_epi16(e16, 7)
                ),
                m16
            );

            _mm256_storeu_si256((__m256i*)(bf16_out + i + half * 16), bf16);
        }
    }
}
#endif  // AVX2

// ============================================================
// Scalar fallback (portable C)
// ============================================================
void kvf13_decode_128_scalar(
    const block_kvf13* block,
    const uint8_t* exp_lut,
    uint16_t* bf16_out
) {
    for (int i = 0; i < 128; i++) {
        // Sign
        uint8_t sign = (block->signs[i / 8] >> (i % 8)) & 1;

        // Exp high nibble
        uint8_t exp_hi_byte = block->exp_hi[i / 2];
        uint8_t exp_h4 = (i & 1) ? (exp_hi_byte >> 4) : (exp_hi_byte & 0x0F);

        // Em byte
        uint8_t em = block->exp_lo_mant[i];
        uint8_t exp_lo1 = em >> 7;
        uint8_t mant7 = em & 0x7F;

        // Reconstruct
        uint8_t exp5 = (exp_h4 << 1) | exp_lo1;
        uint8_t exp8 = exp_lut[exp5];

        bf16_out[i] = ((uint16_t)sign << 15) | ((uint16_t)exp8 << 7) | mant7;
    }
}
