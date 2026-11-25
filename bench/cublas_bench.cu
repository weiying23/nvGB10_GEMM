// cuBLAS baseline：测 BF16 / FP32(TF32) 的 GEMM 有效算力，作为固定参照（只测一次）
// 2026-11-25 修复：随机数据填充替代 memset 0——GB10 功耗受限，零数据位翻转率低 →
// 时钟更高 → GEMM 快 2-3%（实测 2048³：零填充 96.3 vs 随机 93.3）。基线工具必须与
// 被测 kernel（随机数据）同分布，否则基线虚高。
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstring>
#include <vector>

#define CK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA err %d @%d\n",e,__LINE__);return 1;} } while(0)

template <typename F>
static float time_ms(F&& f, int iters) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    // 长热身（协议修订 2026-11-16）：3 次短热身时小形状整个计时区落在 DVFS 时钟
    // 爬升内（空闲 208MHz → 加速 3003MHz），读数系统性偏低 ≈6% 且散布 ±5%。
    // 热身改为时间制：持续 kernel 活跃 ≥30ms 后才开始计时。
    cudaEventRecord(s);
    for (int i=0;i<100000;i++) {
        f();
        if (i % 64 == 63) {
            cudaEventRecord(e); cudaEventSynchronize(e);
            float wms=0; cudaEventElapsedTime(&wms,s,e);
            if (wms > 30.0f) break;   // 累计活跃 ≥30ms
        }
    }
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i=0;i<iters;i++) f();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms=0; cudaEventElapsedTime(&ms,s,e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;
}

// 测一个形状的 BF16 GEMM (C = A@B, 列主序)

// 随机 bf16 填充（hash(index)，值域约 [-2,2)，混合正负）
__global__ void fill_rand_bf16(__nv_bfloat16* p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        unsigned h = (unsigned)(i * 2654435761u) ^ (unsigned)(i >> 32);
        h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
        // 符号/指数/尾数全部取自 hash——指数 120..127（值域 |x|∈[0.008,2)），
        // 保证位翻转率与真实 [-1,1] 随机数据一致（指数位也要翻）
        unsigned short bits = (unsigned short)(((h & 0x8000))
                                               | ((120u + ((h >> 13) & 0x7u)) << 7)
                                               | ((h >> 5) & 0x7Fu));
        __nv_bfloat16 v;
        memcpy(&v, &bits, 2);
        p[i] = v;
    }
}
static void fill_rand(__nv_bfloat16* p, size_t n) {
    int blocks = (int)((n / 256) + 1);
    fill_rand_bf16<<<blocks > 65535 ? 65535 : blocks, 256>>>(p, n);
}

static float bench_bf16(cublasHandle_t h, int M, int N, int K) {
    size_t sa = (size_t)M*K*2, sb=(size_t)K*N*2, sc=(size_t)M*N*2;
    __nv_bfloat16 *A, *B, *C;
    CK(cudaMalloc(&A,sa)); CK(cudaMalloc(&B,sb)); CK(cudaMalloc(&C,sc));
    fill_rand(A, sa/2); fill_rand(B, sb/2); fill_rand(C, sc/2);   // 随机数据（勿 memset 0，见文件头）
    float alpha=1.0f, beta=0.0f;
    auto run = [&]{
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &alpha, B, CUDA_R_16BF, N, A, CUDA_R_16BF, K,
            &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };
    float ms = time_ms(run, 50);
    double flop = 2.0*M*N*K;
    cudaFree(A); cudaFree(B); cudaFree(C);
    return (float)(flop / (ms/1e3) / 1e12);
}

static float bench_fp32(cublasHandle_t h, int M, int N, int K) {
    size_t sa=(size_t)M*K*4, sb=(size_t)K*N*4, sc=(size_t)M*N*4;
    float *A,*B,*C;
    CK(cudaMalloc(&A,sa)); CK(cudaMalloc(&B,sb)); CK(cudaMalloc(&C,sc));
    CK(cudaMemset(A,0,sa)); CK(cudaMemset(B,0,sb)); CK(cudaMemset(C,0,sc));
    float alpha=1.0f, beta=0.0f;
    auto run = [&]{
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &alpha, B, CUDA_R_32F, N, A, CUDA_R_32F, K,
            &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };
    float ms = time_ms(run, 50);
    double flop = 2.0*M*N*K;
    cudaFree(A); cudaFree(B); cudaFree(C);
    return (float)(flop / (ms/1e3) / 1e12);
}

int main(int argc, char** argv) {
    cublasHandle_t h;
    cublasCreate(&h);
    // 支持命令行指定单个形状：cublas_bench M N K（不传则跑默认形状表）
    if (argc >= 4) {
        int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
        float b = bench_bf16(h, m, n, k);
        printf("[M=%d N=%d K=%d] cuBLAS BF16: %.4f TFLOPS\n", m, n, k, b);
        cublasDestroy(h);
        return 0;
    }
    int shapes[][3] = {{4096,4096,4096},{2048,2048,2048},{1024,1024,1024},{8192,8192,8192}};
    printf("shape (M=N=K) | BF16 TFLOPS | FP32(TF32) TFLOPS\n");
    for (auto& s : shapes) {
        int m=s[0], n=s[1], k=s[2];
        float b = bench_bf16(h, m, n, k);
        float f = bench_fp32(h, m, n, k);
        printf("%d            | %.2f       | %.2f\n", m, b, f);
    }
    cublasDestroy(h);
    return 0;
}