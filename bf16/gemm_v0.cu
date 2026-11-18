// gemm_v0.cu —— 消融表第 0 步：朴素 GEMM（global 直算，无任何优化）
//
// 本机基线（criterion.md §0.3）：
//   - BF16 峰值 118.37 TFLOPS，内存带宽 255.6 GB/s（单向读）
//   - cuBLAS BF16：4096^3 = 87.10 TFLOPS（峰值 73.6%），8192^3 = 96.15（81.2%）
//
// 本 kernel 每个线程只算 C 的一个元素，直接反复读 global，作为性能天坑底线。
//
// C = A @ B，A:[M,K] BF16，B:[K,N] BF16，C:[M,N] FP32（累加用 float）

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

// ---------------------------------------------------------------------------
// 第 0 步：朴素 kernel，每线程一个输出元素
// ---------------------------------------------------------------------------
__global__ void gemm_naive(const bf16* __restrict__ A,
                           const bf16* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += __bfloat162float(A[row * K + k]) *
                   __bfloat162float(B[k * N + col]);
        }
        C[row * N + col] = acc;
    }
}

// CPU 参考（真值）：用原始 FP32 输入做 double 累加，用于观察 BF16 量化损失（非正确性判据）
static void cpu_reference_fp32(const std::vector<float>& A,
                               const std::vector<float>& B,
                               std::vector<double>& C,
                               int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += (double)A[i * K + k] * (double)B[k * N + j];
            C[i * N + j] = acc;
        }
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 1024;
    int N = argc > 2 ? atoi(argv[2]) : 1024;
    int K = argc > 3 ? atoi(argv[3]) : 1024;
    // 第 4 个参数任意值 = 跳过 CPU 正确性验证（大尺寸性能测试用）
    bool do_check = (argc <= 4);

    // ---- host 数据：随机 float，量化到 BF16（同时保留原始 FP32 用于真值参考）----
    std::vector<float> hA_f(M * K), hB_f(K * N);   // 原始 FP32（真值）
    std::vector<bf16> hA(M * K), hB(K * N);        // BF16 量化输入
    std::vector<float> hC(M * N);
    srand(42);
    auto randf = []() { return (float)(rand() % 2000 - 1000) / 1000.0f; };  // [-1, 1)
    for (int i = 0; i < M * K; ++i) { hA_f[i] = randf(); hA[i] = __float2bfloat16_rn(hA_f[i]); }
    for (int i = 0; i < K * N; ++i) { hB_f[i] = randf(); hB[i] = __float2bfloat16_rn(hB_f[i]); }

    // ---- device 分配 + 拷贝 ----
    bf16 *dA, *dB; float *dC;
    CK(cudaMalloc(&dA, sizeof(bf16) * M * K));
    CK(cudaMalloc(&dB, sizeof(bf16) * K * N));
    CK(cudaMalloc(&dC, sizeof(float) * M * N));
    CK(cudaMemcpy(dA, hA.data(), sizeof(bf16) * M * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), sizeof(bf16) * K * N, cudaMemcpyHostToDevice));

    // ---- 启动配置 ----
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // ---- 计时（warmup + 多次取平均）----
    const int iters = 10;
    for (int i = 0; i < 3; ++i) gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);  // warmup
    CK(cudaDeviceSynchronize());

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;

    double flop = 2.0 * M * N * K;
    double tflops = flop / (ms / 1e3) / 1e12;
    printf("[M=%d N=%d K=%d] naive: %.3f ms, %.4f TFLOPS (峰值 118.37 的 %.2f%%)\n",
           M, N, K, ms, tflops, tflops / 118.37 * 100);

    // ---- 正确性验证 ----
    if (do_check) {
        CK(cudaMemcpy(hC.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

        // 验证 0：cuBLAS 同精度参考（BF16 输入，FP32 累加输出，作为正确性/回归基准）
        float* dC_cublas;
        CK(cudaMalloc(&dC_cublas, sizeof(float) * M * N));
        cublasHandle_t cublas_h;
        cublasCreate(&cublas_h);
        float calpha = 1.0f, cbeta = 0.0f;
        // 行主序 C[M,N] = A[M,K] @ B[K,N]；cuBLAS 列主序，等价于 C^T = B^T @ A^T
        cublasGemmEx(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                     &calpha, dB, CUDA_R_16BF, N, dA, CUDA_R_16BF, K,
                     &cbeta, dC_cublas, CUDA_R_32F, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        std::vector<float> hC_cublas(M * N);
        CK(cudaMemcpy(hC_cublas.data(), dC_cublas, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        cublasDestroy(cublas_h);
        cudaFree(dC_cublas);

        double cublas_max_abs = 0, cublas_max_rel = 0, cublas_l2 = 0, cublas_l2_ref = 0;
        for (int i = 0; i < M * N; ++i) {
            double d = (double)hC[i] - hC_cublas[i];
            cublas_max_abs = std::max(cublas_max_abs, std::fabs(d));
            cublas_max_rel = std::max(cublas_max_rel, std::fabs(d) / std::max(1.0, std::fabs((double)hC_cublas[i])));
            cublas_l2 += d * d;
            cublas_l2_ref += (double)hC_cublas[i] * hC_cublas[i];
        }
        double cublas_rel_l2 = std::sqrt(cublas_l2) / std::sqrt(cublas_l2_ref);

        printf("正确性 vs cuBLAS(BF16同精度): max_abs_err=%.3e  max_rel_err=%.3e  rel_L2=%.3e  %s\n",
               cublas_max_abs, cublas_max_rel, cublas_rel_l2,
               cublas_max_rel < 1e-3 ? "[PASS]" : "[FAIL]");

        // 参考：BF16 相对 FP32 真值的量化损失（非正确性判据，仅供了解精度水平）
        std::vector<double> hC_true(M * N);
        cpu_reference_fp32(hA_f, hB_f, hC_true, M, N, K);
        double max_rel = 0;
        for (int i = 0; i < M * N; ++i)
            max_rel = std::max(max_rel,
                std::fabs((double)hC[i] - hC_true[i]) / std::max(1.0, std::fabs(hC_true[i])));
        printf("参考(BF16 相对 FP32 真值的量化损失): max_rel_err=%.3e\n", max_rel);
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}