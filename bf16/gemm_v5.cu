// gemm_v5.cu —— v5：4-stage 多缓冲流水（先验证多 stage 是否有效，用 __syncthreads）
//
// 依据 v4/v5 结论：2-stage 重叠深度不够。本版把 buffer 从 2 → 4，
// prologue 预取 3 个 + 主循环超前预取 1 个，wait_group 参数放大到 3。
// mbarrier 的 expect_tx 异步同步留到确认多 stage 有效后（v6）。

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

#define BM 64
#define BN 32
#define BK 16
#define WM 16
#define WN 16
#define NUM_STAGES 4

__device__ __forceinline__ void cp_async16(bf16* smem_dst, const bf16* gmem_src) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__global__ void gemm_mma_v5(const bf16* __restrict__ A,
                            const bf16* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    __shared__ bf16 As[NUM_STAGES][BM][BK];
    __shared__ bf16 Bs[NUM_STAGES][BK][BN];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / 2;
    const int warp_n = warp % 2;
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    constexpr int AS_BLOCKS = BM * BK / 8;
    constexpr int BS_BLOCKS = BK * BN / 8;
    constexpr int TOTAL_BLOCKS = AS_BLOCKS + BS_BLOCKS;

    auto load_async = [&](int stage, int k_start) {
        for (int i = tid; i < TOTAL_BLOCKS; i += 256) {
            if (i < AS_BLOCKS) {
                int row = i / (BK / 8), col = (i % (BK / 8)) * 8;
                cp_async16(&As[stage][row][col], &A[(block_m + row) * K + (k_start + col)]);
            } else {
                int j = i - AS_BLOCKS;
                int row = j / (BN / 8), col = (j % (BN / 8)) * 8;
                cp_async16(&Bs[stage][row][col], &B[(k_start + row) * N + (block_n + col)]);
            }
        }
    };

    float c[2][4] = {{0,0,0,0},{0,0,0,0}};

    // prologue：预取 stage 0,1,2
    #pragma unroll
    for (int i = 0; i < NUM_STAGES - 1; ++i) {
        load_async(i, i * BK);
        cp_async_commit();
    }

    for (int k0 = 0, s = 0; k0 < K; k0 += BK, s = (s + 1) % NUM_STAGES) {
        // 超前预取 stage s+3
        if (k0 + (NUM_STAGES - 1) * BK < K) {
            load_async((s + NUM_STAGES - 1) % NUM_STAGES, k0 + (NUM_STAGES - 1) * BK);
            cp_async_commit();
        }

        // 等 stage s：在飞组数随剩余 K 块递减
        if      (k0 + 3 * BK < K) cp_async_wait<3>();
        else if (k0 + 2 * BK < K) cp_async_wait<2>();
        else if (k0 +     BK < K) cp_async_wait<1>();
        else                      cp_async_wait<0>();
        __syncthreads();

        // mma stage s
        const int arow = warp_m * WM;
        const int atile = lane / 8, ar8 = lane % 8;
        uint32_t a[4];
        uint32_t a_addr = (uint32_t)__cvta_generic_to_shared(
            &As[s][arow + ar8 + (atile % 2) * 8][(atile / 2) * 8]);
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(a_addr));

        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const int brow = warp_n * WN + ni * 8;
            uint32_t b[2];
            uint32_t b_addr = (uint32_t)__cvta_generic_to_shared(
                &Bs[s][lane % 8 + (lane / 8 % 2) * 8][brow]);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(b[0]), "=r"(b[1]) : "r"(b_addr));

            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(c[ni][0]), "+f"(c[ni][1]), "+f"(c[ni][2]), "+f"(c[ni][3])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        }
        __syncthreads();
    }

    const int crow = lane / 4, ccol = (lane % 4) * 2;
    #pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
        int gr = block_m + warp_m * WM + crow;
        int gc = block_n + warp_n * WN + ni * 8 + ccol;
        if (gr < M && gc < N) {
            C[gr * N + gc]         = c[ni][0];
            C[gr * N + gc + 1]     = c[ni][1];
            if (gr + 8 < M) {
                C[(gr + 8) * N + gc]     = c[ni][2];
                C[(gr + 8) * N + gc + 1] = c[ni][3];
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

    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    const int iters = 10;
    for (int i = 0; i < 3; ++i) gemm_mma_v5<<<grid, block>>>(dA, dB, dC, M, N, K);
    CK(cudaDeviceSynchronize());

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) gemm_mma_v5<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;

    double flop = 2.0 * M * N * K;
    double tflops = flop / (ms / 1e3) / 1e12;
    printf("[M=%d N=%d K=%d] v5(4-stage): %.3f ms, %.4f TFLOPS (峰值 118.37 的 %.2f%%)\n",
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