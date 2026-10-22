// gemm_v1.cu —— v1：分块 tile + shared memory（提高算术强度/数据复用）
//
// 依据 v0 结论（reports/v0.md）：naive 是 DRAM-bound（AI≈4.9），根因是数据被重复读 434 倍。
// 本版把 A/B 的 tile 协作搬进 shared memory，block 内所有线程复用，AI 从 4.9 提升到 T/2。
//
// C = A @ B，A:[M,K] BF16，B:[K,N] BF16，C:[M,N] FP32（float 累加）

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

// tile 尺寸（第一版：32×32×32，简单清晰）
#define TM 32
#define TN 32
#define BK 32

// ---------------------------------------------------------------------------
// v1：tiled GEMM，block 16×16=256 线程，每线程累加 2×2=4 个 C 元素
// ---------------------------------------------------------------------------
__global__ void gemm_tiled(const bf16* __restrict__ A,
                           const bf16* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    __shared__ bf16 As[TM][BK];
    __shared__ bf16 Bs[BK][TN];

    const int block_row = blockIdx.y * TM;
    const int block_col = blockIdx.x * TN;
    const int tx = threadIdx.x, ty = threadIdx.y;   // 0..15

    float acc[2][2] = {{0.f, 0.f}, {0.f, 0.f}};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 协作加载 As[TM][BK] = 32×32 = 1024 元素，256 线程 × 4
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = i * 256 + ty * 16 + tx;
            int r = idx / BK, c = idx % BK;
            int gr = block_row + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : bf16(0.0f);
        }
        // 协作加载 Bs[BK][TN] = 32×32 = 1024 元素
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = i * 256 + ty * 16 + tx;
            int r = idx / TN, c = idx % TN;
            int gr = k0 + r, gc = block_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : bf16(0.0f);
        }
        __syncthreads();

        // 计算：每线程累加自己的 2×2 子块
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a0 = __bfloat162float(As[ty      ][k]);
            float a1 = __bfloat162float(As[ty + 16 ][k]);
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                float b = __bfloat162float(Bs[k][tx + j * 16]);
                acc[0][j] += a0 * b;
                acc[1][j] += a1 * b;
            }
        }
        __syncthreads();
    }

    // 写回
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int gr = block_row + ty + i * 16;
            int gc = block_col + tx + j * 16;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 1024;
    int N = argc > 2 ? atoi(argv[2]) : 1024;
    int K = argc > 3 ? atoi(argv[3]) : 1024;
    bool do_check = (argc <= 4);

    std::vector<float> hA_f(M * K), hB_f(K * N);
    std::vector<bf16> hA(M * K), hB(K * N);
    std::vector<float> hC(M * N);
    srand(42);
    auto randf = []() { return (float)(rand() % 2000 - 1000) / 1000.0f; };
    for (int i = 0; i < M * K; ++i) { hA_f[i] = randf(); hA[i] = __float2bfloat16_rn(hA_f[i]); }
    for (int i = 0; i < K * N; ++i) { hB_f[i] = randf(); hB[i] = __float2bfloat16_rn(hB_f[i]); }

    bf16 *dA, *dB; float *dC;
    CK(cudaMalloc(&dA, sizeof(bf16) * M * K));
    CK(cudaMalloc(&dB, sizeof(bf16) * K * N));
    CK(cudaMalloc(&dC, sizeof(float) * M * N));
    CK(cudaMemcpy(dA, hA.data(), sizeof(bf16) * M * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), sizeof(bf16) * K * N, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM);

    const int iters = 10;
    for (int i = 0; i < 3; ++i) gemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    CK(cudaDeviceSynchronize());

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) gemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;

    double flop = 2.0 * M * N * K;
    double tflops = flop / (ms / 1e3) / 1e12;
    printf("[M=%d N=%d K=%d] tiled: %.3f ms, %.4f TFLOPS (峰值 118.37 的 %.2f%%)\n",
           M, N, K, ms, tflops, tflops / 118.37 * 100);

    if (do_check) {
        CK(cudaMemcpy(hC.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

        // cuBLAS 同精度参考（正确性判据）
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
        printf("正确性 vs cuBLAS(BF16同精度): max_abs_err=%.3e  max_rel_err=%.3e  rel_L2=%.3e  %s\n",
               max_abs, max_rel, std::sqrt(l2) / std::sqrt(l2_ref), max_rel < 1e-3 ? "[PASS]" : "[FAIL]");
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}