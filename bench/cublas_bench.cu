// cuBLAS baseline：测 BF16 / FP32(TF32) 的 GEMM 有效算力，作为固定参照（只测一次）
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>

#define CK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA err %d @%d\n",e,__LINE__);return 1;} } while(0)

template <typename F>
static float time_ms(F&& f, int iters) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i=0;i<5;i++) f();                    // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i=0;i<iters;i++) f();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms=0; cudaEventElapsedTime(&ms,s,e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;
}

// 测一个形状的 BF16 GEMM (C = A@B, 列主序)
static float bench_bf16(cublasHandle_t h, int M, int N, int K) {
    size_t sa = (size_t)M*K*2, sb=(size_t)K*N*2, sc=(size_t)M*N*2;
    __nv_bfloat16 *A, *B, *C;
    CK(cudaMalloc(&A,sa)); CK(cudaMalloc(&B,sb)); CK(cudaMalloc(&C,sc));
    CK(cudaMemset(A,0,sa)); CK(cudaMemset(B,0,sb)); CK(cudaMemset(C,0,sc));
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

int main() {
    cublasHandle_t h;
    cublasCreate(&h);
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