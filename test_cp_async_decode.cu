/**
 * Minimal test: cp_async load 208B KVFloat13 chunk to smem, decode to BF16, write back.
 *
 * Grid: 1 block
 * Block: 16 threads (bdx=16, matching head_dim=128 / vec_size=8)
 *
 * Flow per thread:
 *   1. cp_async: load 16B from global → staging smem (thread tx loads bytes [tx*16, tx*16+15])
 *      13 threads active (208/16=13), threads 13-15 predicated off
 *   2. cp_async::commit_group + wait_group<0> + __syncthreads
 *   3. Decode: read from staging smem, decode 8 BF16 values per thread
 *   4. Write decoded BF16 to global output
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// ============================================================
// KVFloat13 constants and LUT
// ============================================================
#define KVF13_SIGN_OFF   0
#define KVF13_EXP_HI_OFF 16
#define KVF13_EM_OFF     80
#define KVF13_CHUNK_BYTES 208

__constant__ uint8_t c_lut[32] = {
    0, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115,
    116, 117, 118, 119, 120, 121, 122, 123,
    124, 125, 126, 127, 128, 129, 130, 131
};

// ============================================================
// Kernel: cp_async load → decode → write
// ============================================================
__global__ void test_cp_async_decode_kernel(
    const uint8_t* __restrict__ packed_input,   // [208] packed KVFloat13
    __nv_bfloat16* __restrict__ bf16_output,    // [128] decoded BF16
    int n_chunks                                // number of chunks to process
) {
    extern __shared__ uint8_t smem[];
    uint8_t* staging = smem;              // 208B staging for packed data
    // (In real kernel, k_smem/v_smem would follow)

    const uint32_t tx = threadIdx.x;      // 0..15
    constexpr uint32_t VEC_SIZE = 8;      // values per thread
    constexpr uint32_t NUM_LOADS = 13;    // 208 / 16

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        const uint8_t* src = packed_input + chunk * KVF13_CHUNK_BYTES;
        __nv_bfloat16* dst = bf16_output + chunk * 128;

        // ---- Phase 1: cp_async load 208B → staging smem ----
        // Thread tx loads 16 bytes at offset tx*16
        // Only threads 0-12 have valid data (13 × 16 = 208)
        {
            bool active = (tx < NUM_LOADS);
            // cp_async: global → smem, 128-bit (16 bytes), async
            if (active) {
                // Use inline PTX for cp.async.cg.shared.global
                uint32_t smem_addr = __cvta_generic_to_shared(staging + tx * 16);
                const void* gmem_ptr = src + tx * 16;
                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;\n"
                    :: "r"(smem_addr), "l"(gmem_ptr)
                );
            }
            // Commit and wait
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 0;\n");
            __syncthreads();
        }

        // ---- Phase 2: decode from staging smem → registers → global ----
        {
            uint32_t pos = tx * VEC_SIZE;  // 0, 8, 16, ..., 120

            #pragma unroll
            for (uint32_t i = 0; i < VEC_SIZE; i++) {
                uint32_t p = pos + i;
                uint32_t sign = (staging[KVF13_SIGN_OFF + p / 8] >> (p & 7)) & 1u;
                uint32_t eh = staging[KVF13_EXP_HI_OFF + p / 2];
                uint32_t exp_h4 = (p & 1) ? ((eh >> 4) & 0xFu) : (eh & 0xFu);
                uint32_t em = staging[KVF13_EM_OFF + p];
                uint32_t exp5 = (exp_h4 << 1) | (em >> 7);
                uint32_t exp8 = c_lut[exp5];
                uint32_t mant7 = em & 0x7Fu;
                uint16_t bf16_bits = (uint16_t)((sign << 15) | (exp8 << 7) | mant7);
                dst[p] = *reinterpret_cast<__nv_bfloat16*>(&bf16_bits);
            }
        }
        __syncthreads();
    }
}

// ============================================================
// Host test
// ============================================================
void encode_kvfloat13_host(const uint16_t* bf16_in, uint8_t* packed_out, int n) {
    // Simple host-side encode for testing
    // compress_lut: exp8 → exp5
    uint8_t compress[256];
    uint8_t decompress[32] = {
        0, 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,
        116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131
    };
    for (int e = 0; e < 256; e++) {
        int best = 0;
        float best_d = 1e30f;
        for (int j = 0; j < 32; j++) {
            float d = fabsf(powf(2.0f, e-127) - powf(2.0f, decompress[j]-127));
            if (d < best_d) { best_d = d; best = j; }
        }
        compress[e] = best;
    }

    memset(packed_out, 0, KVF13_CHUNK_BYTES);
    for (int i = 0; i < n; i++) {
        uint16_t val = bf16_in[i];
        uint8_t sign = (val >> 15) & 1;
        uint8_t exp8 = (val >> 7) & 0xFF;
        uint8_t mant7 = val & 0x7F;
        uint8_t exp5 = compress[exp8];
        uint8_t exp_hi4 = exp5 >> 1;
        uint8_t exp_lo1 = exp5 & 1;

        // Pack sign
        if (sign) packed_out[KVF13_SIGN_OFF + i/8] |= (1 << (i % 8));
        // Pack exp_hi nibble
        if (i & 1)
            packed_out[KVF13_EXP_HI_OFF + i/2] |= (exp_hi4 << 4);
        else
            packed_out[KVF13_EXP_HI_OFF + i/2] |= exp_hi4;
        // Pack exp_lo + mant
        packed_out[KVF13_EM_OFF + i] = (exp_lo1 << 7) | mant7;
    }
}

int main() {
    // Generate test data: 128 BF16 values
    uint16_t h_bf16_orig[128];
    srand(42);
    for (int i = 0; i < 128; i++) {
        // Random BF16 in reasonable range
        float val = (float)(rand() % 2000 - 1000) / 500.0f;
        __nv_bfloat16 bval = __float2bfloat16(val);
        h_bf16_orig[i] = *reinterpret_cast<uint16_t*>(&bval);
    }

    // Encode to KVFloat13
    uint8_t h_packed[KVF13_CHUNK_BYTES];
    encode_kvfloat13_host(h_bf16_orig, h_packed, 128);

    // Expected decode: encode → decode should be ~lossless
    uint8_t decompress[32] = {
        0, 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,
        116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131
    };
    uint16_t h_bf16_expected[128];
    for (int i = 0; i < 128; i++) {
        uint8_t sign = (h_packed[KVF13_SIGN_OFF + i/8] >> (i % 8)) & 1;
        uint8_t eh = h_packed[KVF13_EXP_HI_OFF + i/2];
        uint8_t exp_h4 = (i & 1) ? ((eh >> 4) & 0xF) : (eh & 0xF);
        uint8_t em = h_packed[KVF13_EM_OFF + i];
        uint8_t exp5 = (exp_h4 << 1) | (em >> 7);
        uint8_t exp8 = decompress[exp5];
        uint8_t mant7 = em & 0x7F;
        h_bf16_expected[i] = ((uint16_t)sign << 15) | ((uint16_t)exp8 << 7) | mant7;
    }

    // Allocate device memory
    uint8_t* d_packed;
    __nv_bfloat16* d_output;
    cudaMalloc(&d_packed, KVF13_CHUNK_BYTES);
    cudaMalloc(&d_output, 128 * sizeof(__nv_bfloat16));
    cudaMemcpy(d_packed, h_packed, KVF13_CHUNK_BYTES, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, 128 * sizeof(__nv_bfloat16));

    // Launch kernel
    int smem_size = 256;  // 208B staging + padding
    test_cp_async_decode_kernel<<<1, 16, smem_size>>>(d_packed, d_output, 1);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Read back
    uint16_t h_bf16_result[128];
    cudaMemcpy(h_bf16_result, d_output, 128 * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // Compare
    int match = 0, mismatch = 0;
    for (int i = 0; i < 128; i++) {
        if (h_bf16_result[i] == h_bf16_expected[i]) {
            match++;
        } else {
            mismatch++;
            if (mismatch <= 5) {
                printf("  MISMATCH [%d]: expected=0x%04x got=0x%04x\n",
                       i, h_bf16_expected[i], h_bf16_result[i]);
            }
        }
    }

    printf("Results: %d/128 match, %d mismatch\n", match, mismatch);
    printf("%s\n", mismatch == 0 ? "PASS" : "FAIL");

    // Benchmark
    int N = 10000;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < 100; i++)
        test_cp_async_decode_kernel<<<1, 16, smem_size>>>(d_packed, d_output, 1);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < N; i++)
        test_cp_async_decode_kernel<<<1, 16, smem_size>>>(d_packed, d_output, 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Throughput: %.1f ns/chunk (%.1f GB/s effective)\n",
           ms * 1e6 / N,
           (float)KVF13_CHUNK_BYTES * N / (ms * 1e-3) / 1e9);

    cudaFree(d_packed);
    cudaFree(d_output);
    return mismatch > 0 ? 1 : 0;
}
