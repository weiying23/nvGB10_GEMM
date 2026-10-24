// gemm_v7.cu —— v7：照抄 cuBLAS/CUTLASS 配置（128×256×32，每 warp 64×64，3 stage）
//
// 依据 v6 教训 + nsys 抓到的 cuBLAS kernel（cutlass_80_tensorop_bf16_s16816gemm_128x256_32x3）：
// 相同的 mma.sync 指令，差异在 warp tile（64×64 vs 16×16）。本版每 warp 32 个 mma 输出
// （64×64 / 16×8），128 个累加器/线程，~230 寄存器，低 occupancy 换极高 ILP。
//
// 配置：BM=128, BN=256, BK=32, 8 warps(256线程, 2×4 排布), 3 stage, smem 72KB(dynamic)

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

#define BM 128
#define BN 256
#define BK 32
#define WM 64           // 每 warp M
#define WN 64           // 每 warp N
#define NUM_STAGES 3
#define NUM_M 4         // WM/16
#define NUM_N 8         // WN/8

__device__ __forceinline__ void cp_async16(bf16* smem_dst, const bf16* gmem_src) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__global__ void __launch_bounds__(256, 1) gemm_mma_v7(const bf16* __restrict__ A,
                            const bf16* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    bf16 (*As)[BM][BK] = reinterpret_cast<bf16 (*)[BM][BK]>(smem_raw);
    bf16 (*Bs)[BK][BN] = reinterpret_cast<bf16 (*)[BK][BN]>(smem_raw + NUM_STAGES * sizeof(bf16) * BM * BK);

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / 4;   // 0..1（M 方向 2 段）
    const int warp_n = warp % 4;   // 0..3（N 方向 4 段）
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    constexpr int AS_BLOCKS = BM * BK / 8;      // 512
    constexpr int BS_BLOCKS = BK * BN / 8;      // 1024
    constexpr int TOTAL_BLOCKS = AS_BLOCKS + BS_BLOCKS;  // 1536，256 线程 × 6 块

    auto load_async = [&](int stage, int k_start) {
        #pragma unroll
        for (int i = 0; i < TOTAL_BLOCKS / 256; ++i) {
            int idx = i * 256 + tid;
            if (idx < AS_BLOCKS) {
                int row = idx / (BK / 8), col = (idx % (BK / 8)) * 8;
                cp_async16(&As[stage][row][col], &A[(block_m + row) * K + (k_start + col)]);
            } else {
                int j = idx - AS_BLOCKS;
                int row = j / (BN / 8), col = (j % (BN / 8)) * 8;
                cp_async16(&Bs[stage][row][col], &B[(k_start + row) * N + (block_n + col)]);
            }
        }
    };

    // 累加器：4 M 段 × 8 N 段 × 4 float = 128 个
    float c[NUM_M][NUM_N][4];
    #pragma unroll
    for (int i = 0; i < NUM_M; ++i)
        #pragma unroll
        for (int j = 0; j < NUM_N; ++j) c[i][j][0] = c[i][j][1] = c[i][j][2] = c[i][j][3] = 0.0f;

    // prologue：预取 stage 0, 1
    load_async(0, 0);      cp_async_commit();
    load_async(1, BK);     cp_async_commit();

    for (int k0 = 0, s = 0; k0 < K; k0 += BK, s = (s + 1) % NUM_STAGES) {
        if (k0 + 2 * BK < K) {
            load_async((s + 2) % NUM_STAGES, k0 + 2 * BK);
            cp_async_commit();
        }
        if      (k0 + 2 * BK < K) cp_async_wait<2>();
        else if (k0 +     BK < K) cp_async_wait<1>();
        else                      cp_async_wait<0>();
        __syncthreads();

        // ---- A ldsm：8 条 x4（4 M 段 × 2 K 段）----
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

        // ---- B ldsm：8 条 x4.trans（2 K 段 × 4 N 段）----
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

        // ---- mma：K 方向 2 段 × M 4 段 × N 8 段 = 64 条 ----
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

        __syncthreads();
    }

    // ---- 写回：每线程负责 4 M 段 × 8 N 段 × (2×2) ----
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

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 4096;
    int N = argc > 2 ? atoi(argv[2]) : 4096;
    int K = argc > 3 ? atoi(argv[3]) : 4096;
    bool do_check = (argc <= 4);

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

    constexpr int SMEM_BYTES = NUM_STAGES * sizeof(bf16) * (BM * BK + BK * BN);  // 72KB
    CK(cudaFuncSetAttribute(gemm_mma_v7, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));

    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    const int iters = 10;
    for (int i = 0; i < 3; ++i) gemm_mma_v7<<<grid, block, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
    CK(cudaDeviceSynchronize());

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) gemm_mma_v7<<<grid, block, SMEM_BYTES>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;

    double flop = 2.0 * M * N * K;
    double tflops = flop / (ms / 1e3) / 1e12;
    printf("[M=%d N=%d K=%d] v7(cutlass cfg): %.3f ms, %.4f TFLOPS (峰值 118.37 的 %.2f%%)\n",
           M, N, K, ms, tflops, tflops / 118.37 * 100);

    if (do_check) {
        CK(cudaMemcpy(hC.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

        float* dC_cublas;
        CK(cudaMalloc(&dC_cublas, sizeof(float) * M * N));
        cublasHandle_t h;
        cublasCreate(&h);
        float ca = 1.0f, cb = 0.0f;
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                     &ca, dB, CUDA_R_16BF, N, dA, CUDA_R_16BF, K,
                     &cb, dC_cublas, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        std::vector<float> hC_cublas(M * N);
        CK(cudaMemcpy(hC_cublas.data(), dC_cublas, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        cublasDestroy(h);
        cudaFree(dC_cublas);

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