// fp8/probe_fp8_layout.cu —— 阶段 A：FP8 mma fragment 布局 dump 探针（criterion 阶段门 A）
//
// 目的：一次锁死 e4m3 GEMM 的三套布局（A fragment / B fragment / C fragment），后续 GEMM 零布局 bug。
// 方法（dump 法，与 v2 时代锁 LDSM 布局同一招）：
//   1. 用 e4m3 精确可表示的小整数（[-4,4]）构造 A[16][32] / B[32][8]，f32 累加下所有和精确
//      （乘积 ≤16，部分和 ≤512，均为二进制精确值）→ 对拍必须逐位一致，任何差异 = 布局错；
//   2. 单 warp 执行一次 mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32；
//   3. 按 fp8/README.md §2.1 的推导装载 A/B（b16 字节对同构 + Bt 转置写侧路线）：
//      A-ldsm x4 非 trans：g=lane/8 → 行=(g%2)*8+lane%8，sub=g/2（bf16 A 路径照搬）；
//      B-ldsm x2 非 trans：行=lane%8（n 维），chunk=(lane/8)*16（k-lo/k-hi 两 16B 段）
//                        —— 依赖量化 kernel 写侧转置（[N][K] n-major，行内 k 连续）；
//   4. C fragment 按 bf16 m16n8k16 同款布局写出（lane/4 行、(lane%4)*2 列、+8 行半区）。
//
// 预期：max_abs_err = 0（阶段门 A）。若 B 布局路线 b 失败，回退试路线 a（k-major + trans ldsm）。
// 编译：nvcc -O3 -arch=sm_121 -o fp8/probe_fp8_layout fp8/probe_fp8_layout.cu

#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// ---- host 端 e4m3 量化（数据本就精确可表示，量化 = 恒等）----
static unsigned char q_e4m3(float v) {
    return (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

__global__ void probe_mma(const unsigned char* dA, const unsigned char* dB, float* dC) {
    __shared__ __align__(16) unsigned char sA[16 * 32];   // [m][k] 行主序，行距 32B
    __shared__ __align__(16) unsigned char sB[8 * 32];   // [n][k] 行主序（写侧已转置），行距 32B
    const int tid = threadIdx.x;
    for (int i = tid; i < 16 * 32; i += 32) sA[i] = dA[i];
    for (int i = tid; i < 8 * 32; i += 32) sB[i] = dB[i];
    __syncthreads();

    // ---- A fragment：ldsm x4 非 trans（b16 视图，e4m3 字节对与 bf16 同构）----
    // x4 四矩阵 = (rows 0-7 k-lo | rows 8-15 k-lo | rows 0-7 k-hi | rows 8-15 k-hi)
    // → e4m3 fragment 期望 (a0: r,k0-15 / a1: r+8,k0-15 / a2: r,k16-31 / a3: r+8,k16-31) ✓
    uint32_t a0, a1, a2, a3;
    {
        const int g = tid / 8;                       // 0..3
        const int row = (g % 2) * 8 + tid % 8;       // g0:0-7 g1:8-15 g2:0-7 g3:8-15
        const int sub = g / 2;                       // 0: k-lo(字节 0-15) 1: k-hi(字节 16-31)
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(sA + row * 32 + sub * 16);
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
    }

    // ---- B fragment：ldsm x2 非 trans（写侧转置后的 Bt：行=n，行内=k 字节）----
    // x2 两矩阵 = (rows n0-7 k-lo | rows n0-7 k-hi)；lane<16 提供地址
    // → b0: n=lane/4, k=4*(lane%4)+0..3；b1: k=16+4*(lane%4)+0..3（小端连续）✓
    uint32_t b0, b1;
    {
        const int nrow = tid % 8;                     // n 维行号
        const int chunk = (tid / 8) * 16;            // 0: k-lo 16: k-hi
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(sB + nrow * 32 + chunk);
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
                     : "=r"(b0), "=r"(b1) : "r"(addr));
    }

    // ---- mma：k32 一发 ----
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

    // ---- C fragment 写出（与 bf16 m16n8k16 同款布局）----
    const int r0 = tid / 4, cc = (tid % 4) * 2;
    dC[r0 * 8 + cc]         = c0;
    dC[r0 * 8 + cc + 1]     = c1;
    dC[(r0 + 8) * 8 + cc]     = c2;
    dC[(r0 + 8) * 8 + cc + 1] = c3;
}

int main() {
    constexpr int M = 16, N = 8, K = 32;

    // host 数据：[-4,4] 小整数（e4m3 精确可表示；f32 累加下全链路精确）
    float hA[M][K], hB[K][N];
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            hA[m][k] = (float)(((m * 31 + k * 17) % 9) - 4);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            hB[k][n] = (float)(((k * 13 + n * 7) % 9) - 4);

    // A 量化 + 按行主序（[m][k]）送入 —— 与 GEMM 的 A 全局布局一致
    unsigned char hAq[M * K];
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            hAq[m * K + k] = q_e4m3(hA[m][k]);

    // B 量化 + 写侧转置（[n][k] 行主序）—— 阶段 A 同时验证"量化 kernel 转置写出"路线
    unsigned char hBq[N * K];
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            hBq[n * K + k] = q_e4m3(hB[k][n]);

    unsigned char *dA, *dB; float* dC;
    CK(cudaMalloc(&dA, M * K));
    CK(cudaMalloc(&dB, N * K));
    CK(cudaMalloc(&dC, sizeof(float) * M * N));
    CK(cudaMemcpy(dA, hAq, M * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hBq, N * K, cudaMemcpyHostToDevice));

    probe_mma<<<1, 32>>>(dA, dB, dC);
    CK(cudaDeviceSynchronize());

    float hC[M * N];
    CK(cudaMemcpy(hC, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    // CPU 参照：f32 累加（全链路精确 → 必须逐位一致）
    double max_abs = 0; int bad = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float ref = 0.f;
            for (int k = 0; k < K; ++k) ref += hA[m][k] * hB[k][n];
            double d = fabs((double)hC[m * N + n] - ref);
            if (d > max_abs) max_abs = d;
            if (d != 0.0) { if (bad < 4) printf("  首个错位示例: C[%d][%d] got %g want %g\n", m, n, hC[m * N + n], ref); ++bad; }
        }

    printf("[阶段门 A] mma.m16n8k32.e4m3 布局对拍: max_abs_err=%.3e  错位数=%d/%d  %s\n",
           max_abs, bad, M * N, bad == 0 ? "[PASS 逐位一致]" : "[FAIL]");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return bad == 0 ? 0 : 1;
}
