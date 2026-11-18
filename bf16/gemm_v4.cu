// gemm_v4.cu —— v4：参数化 tile（BM/BN/BK/WN），评估大 tile 表现
//
// 依据 v3 结论：tile 太小 → compute 利用率 22%、AI 83.6。本版把 tile 做成模板参数，
// 每 warp 算 WM×WN（WM=16 固定 = mma M，WN 为 8 的倍数），block = (BM/WM)×(BN/WN) warps。
//
// 前提：M/N/K 为 8 的倍数且边界不越界。

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

// ---- cp.async ----
__device__ __forceinline__ void cp_async16(bf16* smem_dst, const bf16* gmem_src) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

template <int BM, int BN, int BK, int WN>
__global__ void gemm_mma_v4(const bf16* __restrict__ A,
                            const bf16* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    constexpr int WM = 16;
    constexpr int NUM_WARP_M = BM / WM;
    constexpr int NUM_WARP_N = BN / WN;
    constexpr int NUM_WARPS = NUM_WARP_M * NUM_WARP_N;
    constexpr int NUM_MMA_N = WN / 8;          // 每 warp 的 mma 数
    constexpr int AS_BLOCKS = BM * BK / 8;     // 16 字节块数
    constexpr int BS_BLOCKS = BK * BN / 8;
    constexpr int TOTAL_BLOCKS = AS_BLOCKS + BS_BLOCKS;

    __shared__ bf16 As[2][BM][BK];
    __shared__ bf16 Bs[2][BK][BN];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / NUM_WARP_N;
    const int warp_n = warp % NUM_WARP_N;
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;
    constexpr int NUM_THREADS = NUM_WARPS * 32;

    auto load_async = [&](int stage, int k_start) {
        for (int i = tid; i < TOTAL_BLOCKS; i += NUM_THREADS) {
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

    float c[NUM_MMA_N][4];
    #pragma unroll
    for (int i = 0; i < NUM_MMA_N; ++i) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.0f;

    // prologue
    load_async(0, 0);
    cp_async_commit();

    for (int k0 = 0, s = 0; k0 < K; k0 += BK, s ^= 1) {
        if (k0 + BK < K) {
            load_async(s ^ 1, k0 + BK);
            cp_async_commit();
        }
        if (k0 + BK < K) cp_async_wait<1>();
        else             cp_async_wait<0>();
        __syncthreads();

        // 内层 K 循环：BK=16 时 1 次，BK=32 时 2 次，...（减少 barrier 次数的关键）
        #pragma unroll
        for (int ki = 0; ki < BK / 16; ++ki) {
            // A：16×16，K 偏移 ki*16
            const int arow = warp_m * WM;
            const int atile = lane / 8, ar8 = lane % 8;
            uint32_t a[4];
            uint32_t a_addr = (uint32_t)__cvta_generic_to_shared(
                &As[s][arow + ar8 + (atile % 2) * 8][(atile / 2) * 8 + ki * 16]);
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(a_addr));

            // NUM_MMA_N 个 mma（N 方向）
            #pragma unroll
            for (int ni = 0; ni < NUM_MMA_N; ++ni) {
                const int brow = warp_n * WN + ni * 8;
                uint32_t b[2];
                uint32_t b_addr = (uint32_t)__cvta_generic_to_shared(
                    &Bs[s][lane % 8 + (lane / 8 % 2) * 8 + ki * 16][brow]);
                asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(b[0]), "=r"(b[1]) : "r"(b_addr));

                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(c[ni][0]), "+f"(c[ni][1]), "+f"(c[ni][2]), "+f"(c[ni][3])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            }
        }
        __syncthreads();
    }

    // 写回
    const int crow = lane / 4, ccol = (lane % 4) * 2;
    #pragma unroll
    for (int ni = 0; ni < NUM_MMA_N; ++ni) {
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

// 评测一个 tile 配置
template <int BM, int BN, int BK, int WN>
static float run_one(const bf16* dA, const bf16* dB, float* dC,
                     int M, int N, int K, bool do_check) {
    constexpr int NUM_WARPS = (BM / 16) * (BN / WN);
    dim3 block(NUM_WARPS * 32);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const int iters = 10;
    for (int i = 0; i < 3; ++i) gemm_mma_v4<BM,BN,BK,WN><<<grid, block>>>(dA, dB, dC, M, N, K);
    CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) gemm_mma_v4<BM,BN,BK,WN><<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);   // iters 次总时间
    double flop = 2.0 * M * N * K;
    return (float)(flop * iters / (ms / 1e3) / 1e12);
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 4096;
    int N = argc > 2 ? atoi(argv[2]) : 4096;
    int K = argc > 3 ? atoi(argv[3]) : 4096;

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

    printf("=== tile 评估 (M=N=K=%d) ===\n", M);
    printf("%-14s %-8s %-8s %-10s %s\n", "BMxBNxBK", "WN", "threads", "TFLOPS", "峰值%");

    #define EVAL(BM,BN,BK,WN) do { \
        float tf = run_one<BM,BN,BK,WN>(dA,dB,dC,M,N,K,false); \
        printf("%-14s %-8d %-8d %-10.2f %.2f%%\n", #BM "x" #BN "x" #BK, WN, (BM/16)*(BN/WN)*32, tf, tf/118.37*100); \
    } while(0)

    EVAL(64,  32, 16, 16);   // 基线
    EVAL(64,  32, 32, 16);   // BK 翻倍
    EVAL(64,  32, 64, 16);   // BK 四倍
    EVAL(64,  64, 32, 32);
    EVAL(64,  64, 64, 32);
    EVAL(128,128, 32, 32);   // 大 tile + 大 BK

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}