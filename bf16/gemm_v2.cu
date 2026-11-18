// gemm_v2.cu —— v2：Tensor Core（mma.sync m16n8k16 BF16）+ ldmatrix + shared memory
//
// ldmatrix 地址布局（经 dump 验证）：
//   A: x4 非 trans，2×2 象限地址  &As[arow + r8 + (tile%2)*8][(tile/2)*8]
//   B: x2 trans，[K][N] 存储，地址  &Bs[lane%8 + (lane/8%2)*8][brow]

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

__global__ void gemm_mma(const bf16* __restrict__ A,
                         const bf16* __restrict__ B,
                         float* __restrict__ C,
                         int M, int N, int K) {
    __shared__ bf16 As[BM][BK];   // [M][K] row-major
    __shared__ bf16 Bs[BK][BN];   // [K][N] row-major（不转置）

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / 2;   // 0..3
    const int warp_n = warp % 2;   // 0..1

    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    float c[2][4] = {{0,0,0,0},{0,0,0,0}};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 协作加载 As[BM][BK]（coalesced）
        #pragma unroll
        for (int i = 0; i < BM * BK / 256; ++i) {
            int idx = i * 256 + tid;
            int r = idx / BK, kc = idx % BK;
            int gr = block_m + r, gc = k0 + kc;
            As[r][kc] = (gr < M && gc < K) ? A[gr * K + gc] : bf16(0.0f);
        }
        // 协作加载 Bs[BK][BN]（不转置，coalesced）
        #pragma unroll
        for (int i = 0; i < BK * BN / 256; ++i) {
            int idx = i * 256 + tid;
            int k = idx / BN, n = idx % BN;
            int gr = k0 + k, gc = block_n + n;
            Bs[k][n] = (gr < K && gc < N) ? B[gr * N + gc] : bf16(0.0f);
        }
        __syncthreads();

        // ---- ldmatrix 加载 A 的 16×16 ----
        const int arow = warp_m * WM;
        const int atile = lane / 8, ar8 = lane % 8;
        uint32_t a[4];
        uint32_t a_addr = (uint32_t)__cvta_generic_to_shared(
            &As[arow + ar8 + (atile % 2) * 8][(atile / 2) * 8]);
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(a_addr));

        // ---- ldmatrix 加载 B（2 个 N 偏移各 16×8），加 mma ----
        #pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            const int brow = warp_n * WN + ni * 8;
            uint32_t b[2];
            uint32_t b_addr = (uint32_t)__cvta_generic_to_shared(
                &Bs[lane % 8 + (lane / 8 % 2) * 8][brow]);
            asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                         : "=r"(b[0]), "=r"(b[1]) : "r"(b_addr));

            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(c[ni][0]), "+f"(c[ni][1]), "+f"(c[ni][2]), "+f"(c[ni][3])
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        }
        __syncthreads();
    }

    // 写回（mma C 布局：每线程 2×2，行 offset 0/8）
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
    int M = argc > 1 ? atoi(argv[1]) : 1024;
    int N = argc > 2 ? atoi(argv[2]) : 1024;
    int K = argc > 3 ? atoi(argv[3]) : 1024;
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
    for (int i = 0; i < 3; ++i) gemm_mma<<<grid, block>>>(dA, dB, dC, M, N, K);
    CK(cudaDeviceSynchronize());

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) gemm_mma<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;

    double flop = 2.0 * M * N * K;
    double tflops = flop / (ms / 1e3) / 1e12;
    printf("[M=%d N=%d K=%d] mma(ldmatrix): %.3f ms, %.4f TFLOPS (峰值 118.37 的 %.2f%%)\n",
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
        printf("正确性 vs cuBLAS(BF16同精度): max_abs_err=%.3e  max_rel_err=%.3e  rel_L2=%.3e  %s\n",
               max_abs, max_rel, std::sqrt(l2) / std::sqrt(l2_ref), max_rel < 1e-3 ? "[PASS]" : "[FAIL]");
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}