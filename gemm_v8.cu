// gemm_v8.cu —— v8：TMA + mbarrier 流水（替换 cp.async，解放 LSU）
//
// Roofline 依据（v7 分析）：v7 完全不是 DRAM-bound（t_v7/t_DRAM_min = 5~2300×），
// 46% 算力在等 LSU 争抢（cp.async 6×LDGSTS/线程）+ 同步阻塞（2×__syncthreads/K块）。
// 本版：TMA 一条指令搬整 tile（LSU 完全让给 ldsm），mbarrier 替代第一次 __syncthreads。
// ldsm/mma 结构与 v7 完全一致（零改动）。
//
// 前提：M/N/K 为 16 的倍数（TMA 16B 对齐要求）。TMA 越界读自动填 0。

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
#define BK 32
#define WM 64
#define WN 64
#define NUM_STAGES 3
#define NUM_M 4
#define NUM_N 8
#define TMA_BYTES_PER_STAGE (sizeof(bf16) * (BM * BK + BK * BN))

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

__global__ void __launch_bounds__(256, 1) gemm_v8(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    float* __restrict__ C,
    int M, int N, int K) {

    // ---- block swizzle（同 v7）----
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

    // ---- smem 布局（同 v7 + mbarrier）----
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    bf16 (*As)[BM][BK] = reinterpret_cast<bf16 (*)[BM][BK]>(smem_raw);
    bf16 (*Bs)[BK][BN] = reinterpret_cast<bf16 (*)[BK][BN]>(
        smem_raw + NUM_STAGES * sizeof(bf16) * BM * BK);
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(
        smem_raw + NUM_STAGES * sizeof(bf16) * (BM * BK + BK * BN));
    uint64_t* empty_bar = reinterpret_cast<uint64_t*>(
        smem_raw + NUM_STAGES * sizeof(bf16) * (BM * BK + BK * BN)
        + NUM_STAGES * sizeof(uint64_t));

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / 4;
    const int warp_n = warp % 4;

    // ---- 初始化 mbarrier ----
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < NUM_STAGES; ++i) {
            mbar_init(&full_bar[i], 1);      // producer arrive（1 次 = thread 0 的 expect_tx）
            mbar_init(&empty_bar[i], 8);     // consumer arrive（8 次 = 每个 warp 的 leader）
        }
    }
    fence_mbar_init();
    __syncthreads();

    // ---- TMA 加载（不再用 lambda，直接内联在主循环中）----

    // ---- 累加器（同 v7）----
    float c[NUM_M][NUM_N][4];
    #pragma unroll
    for (int i = 0; i < NUM_M; ++i)
        #pragma unroll
        for (int j = 0; j < NUM_N; ++j) c[i][j][0] = c[i][j][1] = c[i][j][2] = c[i][j][3] = 0.0f;

    const int num_k_blocks = (K + BK - 1) / BK;

    // ---- prologue：预取 stage 0, 1（首次使用，不需要等 empty）----
    if (tid == 0) {
        if (num_k_blocks > 0) {
            mbar_arrive_expect_tx(&full_bar[0], TMA_BYTES_PER_STAGE);
            tma_load_2d(&tma_a, As[0], &full_bar[0], 0, block_m);
            tma_load_2d(&tma_b, Bs[0], &full_bar[0], block_n, 0);
        }
        if (num_k_blocks > 1) {
            mbar_arrive_expect_tx(&full_bar[1], TMA_BYTES_PER_STAGE);
            tma_load_2d(&tma_a, As[1], &full_bar[1], BK, block_m);
            tma_load_2d(&tma_b, Bs[1], &full_bar[1], block_n, BK);
        }
    }

    uint32_t phase[NUM_STAGES] = {0, 0, 0};
    uint32_t prod_empty_phase[NUM_STAGES] = {0, 0, 0};  // producer 侧 empty phase（只 tid==0 使用）

    // ---- 主循环（零 __syncthreads）----
    for (int k = 0; k < num_k_blocks; ++k) {
        const int s = k % NUM_STAGES;

        // ① 预取 stage (k+2)%3（thread 0：等 empty → TMA → arrive full）
        if (tid == 0 && k + 2 < num_k_blocks) {
            int next_s = (k + 2) % NUM_STAGES;
            if (k >= 1) {  // stage 复用时才需要等 empty（首次使用不需要）
                mbar_wait(&empty_bar[next_s], prod_empty_phase[next_s]);
                prod_empty_phase[next_s] ^= 1;
            }
            mbar_arrive_expect_tx(&full_bar[next_s], TMA_BYTES_PER_STAGE);
            tma_load_2d(&tma_a, As[next_s], &full_bar[next_s], (k + 2) * BK, block_m);
            tma_load_2d(&tma_b, Bs[next_s], &full_bar[next_s], block_n, (k + 2) * BK);
        }

        // ② 等当前 stage 数据就绪（mbarrier full）
        mbar_wait(&full_bar[s], phase[s]);
        phase[s] ^= 1;

        // ③ 计算（与 v7 完全一致——零改动）
        const int atile = lane / 8, ar8 = lane % 8;
        uint32_t a[2][NUM_M][4];
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki)
            #pragma unroll
            for (int mi = 0; mi < NUM_M; ++mi) {
                int arow = warp_m * WM + mi * 16;
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                    &As[s][arow + ar8 + (atile % 2) * 8][ki * 16 + (atile / 2) * 8]);
                asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[ki][mi][0]), "=r"(a[ki][mi][1]), "=r"(a[ki][mi][2]), "=r"(a[ki][mi][3]) : "r"(addr));
            }

        uint32_t b[2][4][4];
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki)
            #pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
                int brow = warp_n * WN + nj * 16;
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                    &Bs[s][lane % 8 + (lane / 8 % 2) * 8 + ki * 16][(lane / 8 / 2) * 8 + brow]);
                asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b[ki][nj][0]), "=r"(b[ki][nj][1]), "=r"(b[ki][nj][2]), "=r"(b[ki][nj][3]) : "r"(addr));
            }

        #pragma unroll
        for (int ki = 0; ki < 2; ++ki)
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
                                     : "r"(a[ki][mi][0]), "r"(a[ki][mi][1]), "r"(a[ki][mi][2]), "r"(a[ki][mi][3]),
                                       "r"(b[ki][nj][half*2]), "r"(b[ki][nj][half*2+1]));
                    }

        // ④ consumer 释放当前 stage（warp leader arrive on empty barrier）
        if (lane == 0) mbar_arrive(&empty_bar[s]);
    }

    // ---- 写回（同 v7）----
    const int crow = lane / 4, ccol = (lane % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < NUM_M; ++mi)
        #pragma unroll
        for (int ni = 0; ni < NUM_N; ++ni) {
            int gr = block_m + warp_m * WM + mi * 16 + crow;
            int gc = block_n + warp_n * WN + ni * 8 + ccol;
            if (gr < M && gc < N) {
                C[gr * N + gc]         = c[mi][ni][0];
                C[gr * N + gc + 1]     = c[mi][ni][1];
                if (gr + 8 < M) {
                    C[(gr + 8) * N + gc]     = c[mi][ni][2];
                    C[(gr + 8) * N + gc + 1] = c[mi][ni][3];
                }
            }
        }
}

// ---- host：创建 tensor map ----
static void create_tensor_map_A(CUtensorMap* map, const bf16* dA, int M, int K) {
    cuuint64_t globalDim[2] = {(cuuint64_t)K, (cuuint64_t)M};       // 内维=K（连续），外维=M
    cuuint64_t globalStrides[1] = {(cuuint64_t)K * sizeof(bf16)};   // 外维 stride（字节）
    cuuint32_t boxDim[2] = {BK, BM};                                 // 内维=BK，外维=BM
    cuuint32_t elementStrides[2] = {1, 1};
    CK_DRV(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        (void*)dA, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
static void create_tensor_map_B(CUtensorMap* map, const bf16* dB, int K, int N) {
    cuuint64_t globalDim[2] = {(cuuint64_t)N, (cuuint64_t)K};       // 内维=N（连续），外维=K
    cuuint64_t globalStrides[1] = {(cuuint64_t)N * sizeof(bf16)};   // 外维 stride（字节）
    cuuint32_t boxDim[2] = {BN, BK};                                 // 内维=BN，外维=BK
    cuuint32_t elementStrides[2] = {1, 1};
    CK_DRV(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        (void*)dB, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
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

    // 创建 tensor map
    CUtensorMap tmap_a, tmap_b;
    create_tensor_map_A(&tmap_a, dA, M, K);
    create_tensor_map_B(&tmap_b, dB, K, N);

    constexpr int SMEM_BYTES = NUM_STAGES * sizeof(bf16) * (BM * BK + BK * BN)
                             + 2 * NUM_STAGES * sizeof(uint64_t);  // full + empty mbarriers
    CK(cudaFuncSetAttribute(gemm_v8, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));

    const uint32_t num_m_blocks = (M + BM - 1) / BM;
    const uint32_t num_n_blocks = (N + BN - 1) / BN;
    dim3 block(256);
    dim3 grid(num_m_blocks * num_n_blocks);

    const int iters = 10;
    for (int i = 0; i < 3; ++i) gemm_v8<<<grid, block, SMEM_BYTES>>>(tmap_a, tmap_b, dC, M, N, K);
    CK(cudaDeviceSynchronize());

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) gemm_v8<<<grid, block, SMEM_BYTES>>>(tmap_a, tmap_b, dC, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;

    double flop = 2.0 * M * N * K;
    double tflops = flop / (ms / 1e3) / 1e12;
    printf("[M=%d N=%d K=%d] v8(TMA): %.3f ms, %.4f TFLOPS (峰值 118.37 的 %.2f%%)\n",
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