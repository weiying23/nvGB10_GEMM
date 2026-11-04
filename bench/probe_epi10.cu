// gemm_v10e.cu —— v10：TMA + SWIZZLE_128B + BK=64 + 2-stage（nvjet 正统路线）
//
// v9 的 padded 与 v8 的零同步互斥（mbarrier full 依赖 TMA，TMA 写不了 padded 布局），
// 本版用 TMA 的硬件 swizzle 合流：SWIZZLE_128B 让 TMA 写入时自动做 XOR 槽位交错，
// 同时拿到「零 bank conflict + TMA + 零 __syncthreads」。
//
// 布局（swizzle 公式已经探针逐位验证，见 bench/probe_tmaswiz.cu）：
//   TMA box 内元素 (行 r, 内维元素 c)，行距 128B（64 个 bf16）：
//   物理字节偏移 = r*128 + ((c/8) ^ (r%8))*16 + (c%8)*2
//   A：box {64,128}（内维 K=64 元素 = 128B，外维 M=128 行），每 stage 1 次 TMA
//   B：box {64,64}，BN=256 拆 4 个 chunk（warp_n 天然对应 chunk），每 stage 4 次 TMA
//   smem/stage = 16KB(A) + 32KB(B) = 48KB；NUM_STAGES=2 → 96KB（≤99KB）
//
// mma 循环：BK=64 = 4 个 k16 slice，分 2 个 pair 处理（每 pair 16 ldsm + 64 mma），
// 片段寄存器与 v9 同为 64 个（pair 内复用），寄存器不膨胀。
//
// 前提：M/N/K 为 16 的倍数（TMA 16B 对齐；越界自动填 0，写回带边界判断）。

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cublas_v2.h>

using bf16 = __nv_bfloat16;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define CK_DRV(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { \
    const char* s; cuGetErrorString(e, &s); fprintf(stderr, "CU error %s @%d\n", s, __LINE__); exit(1); } } while(0)

#define BM 128
#define BN 256
#define BK 64
#define WM 64
#define WN 64
#define NUM_STAGES 2
#define NUM_M 4
#define NUM_N 8
#define A_STAGE_BYTES  (BM * 128)          // 128 行 × 128B
#define B_CHUNK_BYTES  (64 * 128)          // 64 行 × 128B
#define B_STAGE_BYTES  (4 * B_CHUNK_BYTES)
#define TMA_BYTES_PER_STAGE (A_STAGE_BYTES + B_STAGE_BYTES)

// ---- mbarrier ----
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t c) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(c));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* b, uint32_t bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(a), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(a));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
    uint32_t done = 0;
    while (!done) {
        asm volatile("{\n\t.reg .pred p;\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
                     "selp.b32 %0, 1, 0, p;\n\t}\n" : "=r"(done) : "r"(a), "r"(ph));
    }
}
__device__ __forceinline__ void fence_mbar_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n");
}

// ---- TMA 2D load ----
__device__ __forceinline__ void tma_load_2d(const void* tmap, void* smem_dst,
                                            uint64_t* mbar, int x, int y) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_dst);
    uint32_t m = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(s), "l"(tmap), "r"(x), "r"(y), "r"(m));
}

__global__ void __launch_bounds__(256, 1) gemm_v10e(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    float* __restrict__ C,
    int M, int N, int K) {

    // ---- block swizzle（同 v7/v9）----
    constexpr uint32_t GROUP_M = 8;
    const uint32_t num_m_blocks = (M + BM - 1) / BM;
    const uint32_t num_n_blocks = (N + BN - 1) / BN;
    const uint32_t bid = blockIdx.x;
    const uint32_t group_size = GROUP_M * num_n_blocks;
    const uint32_t group_id = bid / group_size;
    const uint32_t in_group = bid % group_size;
    const uint32_t m_first = min(group_id * GROUP_M, num_m_blocks);
    const uint32_t m_g = min(GROUP_M, num_m_blocks - m_first);
    const uint32_t m_block_idx = m_first + in_group % m_g;
    const uint32_t n_block_idx = in_group / m_g;
    if (n_block_idx >= num_n_blocks) return;
    const int block_m = m_block_idx * BM;
    const int block_n = n_block_idx * BN;

    // ---- smem 布局：每 stage 连续 [A 16KB | B 32KB]，再接 mbarrier ----
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(smem_raw + NUM_STAGES * TMA_BYTES_PER_STAGE);
    uint64_t* empty_bar = full_bar + NUM_STAGES;

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / 4;
    const int warp_n = warp % 4;         // 0..3：天然对应 B 的 4 个 64 列 chunk

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < NUM_STAGES; ++i) {
            mbar_init(&full_bar[i], 1);
            mbar_init(&empty_bar[i], 8);
        }
    }
    fence_mbar_init();
    __syncthreads();

    const int num_k_blocks = (K + BK - 1) / BK;

    // producer：装一个 stage（A 1 条 + B 4 条 TMA）
    auto issue_stage = [&](int s, int kb) {
        uint8_t* a_dst = smem_raw + s * TMA_BYTES_PER_STAGE;
        uint8_t* b_dst = a_dst + A_STAGE_BYTES;
        mbar_arrive_expect_tx(&full_bar[s], TMA_BYTES_PER_STAGE);
        tma_load_2d(&tma_a, a_dst, &full_bar[s], kb * BK, block_m);
        #pragma unroll
        for (int g = 0; g < 4; ++g)
            tma_load_2d(&tma_b, b_dst + g * B_CHUNK_BYTES, &full_bar[s],
                        block_n + g * 64, kb * BK);
    };

    // 累加器（同 v7/v9）
    float c[NUM_M][NUM_N][4];
    #pragma unroll
    for (int i = 0; i < NUM_M; ++i)
        #pragma unroll
        for (int j = 0; j < NUM_N; ++j) c[i][j][0] = c[i][j][1] = c[i][j][2] = c[i][j][3] = 0.0f;

    // prologue：预取 stage 0
    if (tid == 0 && num_k_blocks > 0) issue_stage(0, 0);

    uint32_t phase[NUM_STAGES] = {0};
    uint32_t prod_empty_phase[NUM_STAGES] = {0};

    // ---- 主循环（零 __syncthreads，BK=64 分 2 个 pair 处理）----
    for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int s = kb % NUM_STAGES;

        // ① producer（thread 0）：等 empty → 装下一 stage
        if (tid == 0 && kb + 1 < num_k_blocks) {
            int ns = (kb + 1) % NUM_STAGES;
            if (kb >= 1) {
                mbar_wait(&empty_bar[ns], prod_empty_phase[ns]);
                prod_empty_phase[ns] ^= 1;
            }
            issue_stage(ns, kb + 1);
        }

        // ② 等当前 stage 数据就绪
        mbar_wait(&full_bar[s], phase[s]);
        phase[s] ^= 1;

        const uint8_t* a_base = smem_raw + s * TMA_BYTES_PER_STAGE;
        const uint8_t* b_base = a_base + A_STAGE_BYTES;
        const int atile = lane / 8, ar8 = lane % 8;

        // ③ 2 个 pair × (16 ldsm + 64 mma)，片段寄存器 pair 内复用
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            uint32_t a[2][NUM_M][4];
            #pragma unroll
            for (int kii = 0; kii < 2; ++kii) {
                const int ks = p * 2 + kii;          // K 块内 k16 slice 0..3
                #pragma unroll
                for (int mi = 0; mi < NUM_M; ++mi) {
                    const int r = warp_m * WM + mi * 16 + ar8 + (atile % 2) * 8;
                    const int sub = ks * 2 + atile / 2;
                    uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                        a_base + r * 128 + (sub ^ (r & 7)) * 16);
                    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(a[kii][mi][0]), "=r"(a[kii][mi][1]), "=r"(a[kii][mi][2]), "=r"(a[kii][mi][3]) : "r"(addr));
                }
            }
            uint32_t b[2][4][4];
            #pragma unroll
            for (int kii = 0; kii < 2; ++kii) {
                const int ks = p * 2 + kii;
                #pragma unroll
                for (int nj = 0; nj < 4; ++nj) {
                    const int k = ks * 16 + lane % 8 + (lane / 8 % 2) * 8;
                    const int nsub = nj * 2 + (lane / 8 / 2);
                    uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                        b_base + warp_n * B_CHUNK_BYTES + k * 128 + (nsub ^ (k & 7)) * 16);
                    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(b[kii][nj][0]), "=r"(b[kii][nj][1]), "=r"(b[kii][nj][2]), "=r"(b[kii][nj][3]) : "r"(addr));
                }
            }
            #pragma unroll
            for (int kii = 0; kii < 2; ++kii)
                #pragma unroll
                for (int mi = 0; mi < NUM_M; ++mi)
                    #pragma unroll
                    for (int nj = 0; nj < 4; ++nj)
                        #pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            int ni = nj * 2 + half;
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                         : "+f"(c[mi][ni][0]), "+f"(c[mi][ni][1]), "+f"(c[mi][ni][2]), "+f"(c[mi][ni][3])
                                         : "r"(a[kii][mi][0]), "r"(a[kii][mi][1]), "r"(a[kii][mi][2]), "r"(a[kii][mi][3]),
                                           "r"(b[kii][nj][half*2]), "r"(b[kii][nj][half*2+1]));
                        }
        }

        // ④ 释放 stage（fence：SM121 上 arrive 必须在 LDSM 读完成之后）
        asm volatile("fence.acq_rel.cta;\n");
        if (lane == 0) mbar_arrive(&empty_bar[s]);
    }

    // ---- epilogue 消融：128 累加器归约（保活），写回成本≈0 ----
    {
        float sum = 0;
        #pragma unroll
        for (int mi = 0; mi < NUM_M; ++mi)
            #pragma unroll
            for (int ni = 0; ni < NUM_N; ++ni)
                sum += c[mi][ni][0] + c[mi][ni][1] + c[mi][ni][2] + c[mi][ni][3];
        if (sum == -12345.678f)
            C[(block_m + warp_m * WM + lane / 4) * N + block_n + warp_n * WN] = sum;
    }
}

// ---- host：tensor map（SWIZZLE_128B）----
static void create_tensor_map_A(CUtensorMap* map, const bf16* dA, int M, int K) {
    cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)M};
    cuuint64_t gstr[1] = {(cuuint64_t)K * sizeof(bf16)};
    cuuint32_t box[2] = {BK, BM};                 // 内维 K=64（128B），外维 M=128
    cuuint32_t es[2] = {1, 1};
    CK_DRV(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        (void*)dA, gdim, gstr, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
static void create_tensor_map_B(CUtensorMap* map, const bf16* dB, int K, int N) {
    cuuint64_t gdim[2] = {(cuuint64_t)N, (cuuint64_t)K};
    cuuint64_t gstr[1] = {(cuuint64_t)N * sizeof(bf16)};
    cuuint32_t box[2] = {64, 64};                  // 内维 N=64（128B），外维 K=64
    cuuint32_t es[2] = {1, 1};
    CK_DRV(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        (void*)dB, gdim, gstr, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 4096;
    int N = argc > 2 ? atoi(argv[2]) : 4096;
    int K = argc > 3 ? atoi(argv[3]) : 4096;
    bool do_check = (argc <= 4);

    if (M % 16 != 0 || N % 16 != 0 || K % 16 != 0) {
        fprintf(stderr, "Error: M/N/K must be multiples of 16 (TMA alignment)\n");
        return 1;
    }

    std::vector<bf16> hA(M * K), hB(K * N);
    std::vector<float> hC(M * N);
    srand(42);
    auto randf = []() { return (float)(rand() % 2000 - 1000) / 1000.0f; };
    for (int i = 0; i < M * K; ++i) hA[i] = __float2bfloat16_rn(randf());
    for (int i = 0; i < K * N; ++i) hB[i] = __float2bfloat16_rn(randf());

    bf16 *dA, *dB; float *dC;
    CK(cudaMalloc(&dA, sizeof(bf16) * M * K));
    CK(cudaMalloc(&dB, sizeof(bf16) * K * N));
    CK(cudaMalloc(&dC, sizeof(float) * M * N));
    CK(cudaMemcpy(dA, hA.data(), sizeof(bf16) * M * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), sizeof(bf16) * K * N, cudaMemcpyHostToDevice));

    CUtensorMap tmap_a, tmap_b;
    create_tensor_map_A(&tmap_a, dA, M, K);
    create_tensor_map_B(&tmap_b, dB, K, N);

    constexpr int SMEM_BYTES = NUM_STAGES * TMA_BYTES_PER_STAGE
                             + 2 * NUM_STAGES * sizeof(uint64_t);   // 96KB + 32B
    CK(cudaFuncSetAttribute(gemm_v10e, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));

    const uint32_t num_m_blocks = (M + BM - 1) / BM;
    const uint32_t num_n_blocks = (N + BN - 1) / BN;
    dim3 block(256);
    dim3 grid(num_m_blocks * num_n_blocks);

    const int iters = 10;
    for (int i = 0; i < 3; ++i) gemm_v10e<<<grid, block, SMEM_BYTES>>>(tmap_a, tmap_b, dC, M, N, K);
    CK(cudaDeviceSynchronize());

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) gemm_v10e<<<grid, block, SMEM_BYTES>>>(tmap_a, tmap_b, dC, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;

    double flop = 2.0 * M * N * K;
    double tflops = flop / (ms / 1e3) / 1e12;
    printf("[M=%d N=%d K=%d] v10(TMA+SW128+BK64): %.3f ms, %.4f TFLOPS (峰值 118.37 的 %.2f%%)\n",
           M, N, K, ms, tflops, tflops / 118.37 * 100);

    if (do_check) {
        CK(cudaMemcpy(hC.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        float* dC_cublas;
        CK(cudaMalloc(&dC_cublas, sizeof(float) * M * N));
        cublasHandle_t h; cublasCreate(&h);
        float ca = 1.0f, cb = 0.0f;
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                     &ca, dB, CUDA_R_16BF, N, dA, CUDA_R_16BF, K,
                     &cb, dC_cublas, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        std::vector<float> hC_cublas(M * N);
        CK(cudaMemcpy(hC_cublas.data(), dC_cublas, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        cublasDestroy(h); cudaFree(dC_cublas);
        double max_abs = 0, max_rel = 0, l2 = 0, l2_ref = 0;
        for (int i = 0; i < M * N; ++i) {
            double d = (double)hC[i] - hC_cublas[i];
            max_abs = std::max(max_abs, std::fabs(d));
            max_rel = std::max(max_rel, std::fabs(d) / std::max(1.0, std::fabs((double)hC_cublas[i])));
            l2 += d * d; l2_ref += (double)hC_cublas[i] * hC_cublas[i];
        }
        printf("正确性 vs cuBLAS: max_abs_err=%.3e  max_rel_err=%.3e  rel_L2=%.3e  %s\n",
               max_abs, max_rel, std::sqrt(l2) / std::sqrt(l2_ref), max_rel < 1e-3 ? "[PASS]" : "[FAIL]");
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
