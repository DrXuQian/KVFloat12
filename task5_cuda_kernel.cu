/**
 * Task 5: CUDA Decode Kernel for KVFloat13
 *
 * Fused KVFloat13 decode + dot product
 * Used for both attention (Q × K_compressed) and GEMM (input × W_compressed)
 *
 * Block structure per 128 values:
 *   signs[16]       — 128 × 1-bit sign, packed 8/byte = 16 bytes
 *   exp_hi[64]      — 128 × 4-bit exp_high, nibble packed = 64 bytes
 *   exp_lo_mant[128]— 128 × [exp_low(1)|mantissa(7)] = 128 bytes
 *   Total: 208 bytes per 128 values
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================
// Decompress LUT in constant memory (set once at model load)
// Maps 5-bit compressed exponent → 8-bit BF16 exponent
// ============================================================
__constant__ uint8_t d_exp5_to_exp8[32];

// Block layout (packed, 208 bytes per 128 values)
struct block_kvf13 {
    uint8_t  signs[16];         // 128 bits packed
    uint8_t  exp_hi[64];        // 128 nibbles packed
    uint8_t  exp_lo_mant[128];  // 128 bytes direct
};

static_assert(sizeof(block_kvf13) == 208, "block_kvf13 must be 208 bytes");

// ============================================================
// Device function: Decode 4 KVFloat13 values from a block
// Each thread in a warp handles 4 consecutive values
// tid = lane index [0..31], base = tid * 4
// ============================================================
__device__ __forceinline__ void kvf13_decode_4(
    const block_kvf13* __restrict__ block,
    const int tid,
    __nv_bfloat16 out[4]
) {
    const int base = tid * 4;

    // Load sign bits: which byte contains our 4 values?
    // base/8 gives the byte index, base%8 gives bit offset
    const int sign_byte_idx = base >> 3;
    const int sign_bit_off  = base & 7;
    const uint8_t sign_byte = block->signs[sign_byte_idx];

    // Load exp_hi: 2 bytes contain 4 nibbles (base/2 .. base/2+1)
    // Each byte stores 2 nibbles: low nibble = even index, high nibble = odd index
    const int exp_hi_byte_idx = base >> 1;  // base/2
    const uint8_t exp_hi_b0 = block->exp_hi[exp_hi_byte_idx];
    const uint8_t exp_hi_b1 = block->exp_hi[exp_hi_byte_idx + 1];

    // Load em: 4 consecutive bytes (perfectly coalesced)
    const uint32_t em4 = *reinterpret_cast<const uint32_t*>(&block->exp_lo_mant[base]);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // Extract sign bit
        uint32_t sign = (sign_byte >> (sign_bit_off + i)) & 1u;

        // Extract exp_hi4 (nibble)
        // i=0,1 from exp_hi_b0; i=2,3 from exp_hi_b1
        uint32_t exp_h4;
        if (i < 2) {
            exp_h4 = (exp_hi_b0 >> (i * 4)) & 0xFu;
        } else {
            exp_h4 = (exp_hi_b1 >> ((i - 2) * 4)) & 0xFu;
        }

        // Extract em byte
        uint32_t em_b = (em4 >> (i * 8)) & 0xFFu;

        // Reconstruct 5-bit exponent
        uint32_t exp5 = (exp_h4 << 1) | (em_b >> 7);

        // LUT decompress → 8-bit exponent
        uint32_t exp8 = d_exp5_to_exp8[exp5];

        // Mantissa (untouched)
        uint32_t mant7 = em_b & 0x7Fu;

        // Reassemble BF16
        uint16_t bf16_bits = (uint16_t)((sign << 15) | (exp8 << 7) | mant7);
        out[i] = *reinterpret_cast<__nv_bfloat16*>(&bf16_bits);
    }
}

// ============================================================
// Fused decode + dot product: one warp processes one block (128 values)
// Returns the dot product in lane 0
// ============================================================
__device__ float kvf13_fused_dot_128(
    const block_kvf13* __restrict__ block,
    const __nv_bfloat16* __restrict__ x,    // 128-dim input vector
    const int tid                            // lane within warp [0..31]
) {
    const int base = tid * 4;

    // Decode 4 values
    __nv_bfloat16 vals[4];
    kvf13_decode_4(block, tid, vals);

    // Dot product with input
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sum += __bfloat162float(vals[i]) * __bfloat162float(x[base + i]);
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    return sum;  // Only lane 0 has the final result
}

// ============================================================
// Kernel: Matrix-vector product  y = W_compressed × x
// W is stored as KVFloat13 blocks, x is BF16 input, y is FP32 output
//
// Grid: (num_rows / ROWS_PER_BLOCK, 1, 1)
// Block: (32 * ROWS_PER_BLOCK, 1, 1)  — one warp per row
//
// Each row has (cols/128) KVFloat13 blocks
// ============================================================
#define ROWS_PER_TB 4  // Rows per thread block (warps per TB)

__global__ void kvf13_matvec_kernel(
    const block_kvf13* __restrict__ W,  // [num_rows × blocks_per_row]
    const __nv_bfloat16* __restrict__ x,  // [cols]
    float* __restrict__ y,               // [num_rows]
    const int num_rows,
    const int blocks_per_row              // cols / 128
) {
    const int warp_id = threadIdx.x >> 5;          // Which warp in this TB
    const int lane    = threadIdx.x & 31;          // Lane within warp
    const int row     = blockIdx.x * ROWS_PER_TB + warp_id;

    if (row >= num_rows) return;

    float row_sum = 0.0f;

    // Process all blocks in this row
    const block_kvf13* row_blocks = W + (size_t)row * blocks_per_row;
    for (int b = 0; b < blocks_per_row; b++) {
        float dot = kvf13_fused_dot_128(&row_blocks[b], &x[b * 128], lane);
        if (lane == 0) {
            row_sum += dot;
        }
    }

    // Write result
    if (lane == 0) {
        y[row] = row_sum;
    }
}

// ============================================================
// Kernel: Batch decode — decompress KVFloat13 blocks to BF16
// Used for KV cache decompression before standard attention
// ============================================================
__global__ void kvf13_decode_kernel(
    const block_kvf13* __restrict__ src,   // [num_blocks]
    __nv_bfloat16* __restrict__ dst,       // [num_blocks * 128]
    const int num_blocks
) {
    const int block_idx = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;

    if (block_idx >= num_blocks) return;

    __nv_bfloat16 vals[4];
    kvf13_decode_4(&src[block_idx], lane, vals);

    const int base = block_idx * 128 + lane * 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        dst[base + i] = vals[i];
    }
}

// ============================================================
// Kernel: Batch encode — compress BF16 to KVFloat13
// Used for online KV cache compression during inference
// One warp per block of 128 values
// ============================================================
__constant__ uint8_t d_exp8_to_exp5[256];  // compress LUT

__global__ void kvf13_encode_kernel(
    const __nv_bfloat16* __restrict__ src,  // [num_blocks * 128]
    block_kvf13* __restrict__ dst,          // [num_blocks]
    const int num_blocks
) {
    const int block_idx = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;

    if (block_idx >= num_blocks) return;

    const int base = block_idx * 128 + lane * 4;

    // Each thread processes 4 values
    uint8_t signs_4 = 0;
    uint8_t exp_hi_nib[4];
    uint8_t em[4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint16_t bf16_bits = *reinterpret_cast<const uint16_t*>(&src[base + i]);
        uint32_t sign = (bf16_bits >> 15) & 1;
        uint32_t exp8 = (bf16_bits >> 7) & 0xFF;
        uint32_t mant7 = bf16_bits & 0x7F;

        uint32_t exp5 = d_exp8_to_exp5[exp8];
        uint32_t exp_hi4 = exp5 >> 1;
        uint32_t exp_lo1 = exp5 & 1;

        signs_4 |= (sign << i);
        exp_hi_nib[i] = (uint8_t)exp_hi4;
        em[i] = (uint8_t)((exp_lo1 << 7) | mant7);
    }

    // Write em bytes (4 consecutive, coalesced)
    *reinterpret_cast<uint32_t*>(&dst[block_idx].exp_lo_mant[lane * 4]) =
        em[0] | (em[1] << 8) | (em[2] << 16) | (em[3] << 24);

    // Write exp_hi nibbles (2 bytes for 4 nibbles)
    dst[block_idx].exp_hi[lane * 2]     = exp_hi_nib[0] | (exp_hi_nib[1] << 4);
    dst[block_idx].exp_hi[lane * 2 + 1] = exp_hi_nib[2] | (exp_hi_nib[3] << 4);

    // Write sign bits: need atomic OR since multiple threads write to same byte
    // lane*4 / 8 = lane/2 is the byte index
    // lane*4 % 8 = (lane&1)*4 is the bit offset within that byte
    const int sign_byte_idx = lane >> 1;
    const int sign_bit_off  = (lane & 1) * 4;
    atomicOr(reinterpret_cast<unsigned int*>(&dst[block_idx].signs[sign_byte_idx & ~3]),
             (unsigned int)signs_4 << (sign_bit_off + (sign_byte_idx & 3) * 8));
}

// ============================================================
// Host API
// ============================================================

extern "C" {

// Initialize LUTs in constant memory
void kvf13_init_luts(const uint8_t* compress_lut_h, const uint8_t* decompress_lut_h) {
    cudaMemcpyToSymbol(d_exp5_to_exp8, decompress_lut_h, 32);
    cudaMemcpyToSymbol(d_exp8_to_exp5, compress_lut_h, 256);
}

// Matrix-vector product: y = W × x
void kvf13_matvec(
    const block_kvf13* W_d,
    const __nv_bfloat16* x_d,
    float* y_d,
    int num_rows,
    int cols
) {
    int blocks_per_row = cols / 128;
    dim3 grid((num_rows + ROWS_PER_TB - 1) / ROWS_PER_TB);
    dim3 block(32 * ROWS_PER_TB);
    kvf13_matvec_kernel<<<grid, block>>>(W_d, x_d, y_d, num_rows, blocks_per_row);
}

// Batch decode: decompress KVFloat13 → BF16
void kvf13_decode(
    const block_kvf13* src_d,
    __nv_bfloat16* dst_d,
    int num_blocks
) {
    const int warps_per_tb = 8;
    dim3 block(32 * warps_per_tb);
    dim3 grid((num_blocks + warps_per_tb - 1) / warps_per_tb);
    kvf13_decode_kernel<<<grid, block>>>(src_d, dst_d, num_blocks);
}

// Batch encode: compress BF16 → KVFloat13
void kvf13_encode(
    const __nv_bfloat16* src_d,
    block_kvf13* dst_d,
    int num_blocks
) {
    const int warps_per_tb = 8;
    dim3 block(32 * warps_per_tb);
    dim3 grid((num_blocks + warps_per_tb - 1) / warps_per_tb);

    // Zero out sign bytes first (since encode uses atomicOr)
    cudaMemset(dst_d, 0, (size_t)num_blocks * sizeof(block_kvf13));

    kvf13_encode_kernel<<<grid, block>>>(src_d, dst_d, num_blocks);
}

}  // extern "C"
