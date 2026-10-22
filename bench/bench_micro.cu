// Microbench：实测本机峰值算力（BF16/TF32 张量核）与内存带宽
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while(0)

// 峰值：BF16 mma.sync m16n8k16，8 个独立累加器提供 ILP
__global__ void peak_bf16_kernel(float* out, int iters) {
    float c[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) c[i][j] = 0.0f;
    const uint32_t a0 = 0x3f803f80u, a1 = 0x3f803f80u, a2 = 0x3f803f80u, a3 = 0x3f803f80u;
    const uint32_t b0 = 0x3f803f80u, b1 = 0x3f803f80u;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    float s = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) s += c[i][0] + c[i][1] + c[i][2] + c[i][3];
    if (s == -12345.0f) out[threadIdx.x] = s;
}

// 峰值：TF32 mma.sync m16n8k8
__global__ void peak_tf32_kernel(float* out, int iters) {
    float c[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) c[i][j] = 0.0f;
    const uint32_t a0 = 0x3f800000u, a1 = 0x3f800000u, a2 = 0x3f800000u, a3 = 0x3f800000u;
    const uint32_t b0 = 0x3f800000u, b1 = 0x3f800000u;
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    float s = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) s += c[i][0] + c[i][1] + c[i][2] + c[i][3];
    if (s == -12345.0f) out[threadIdx.x] = s;
}

// 带宽：grid-stride float4 读
__global__ void bw_read_kernel(const float4* __restrict__ buf, float* out, long long n4) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (long long j = i; j < n4; j += stride) {
        float4 v = buf[j];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == -12345.0f) out[threadIdx.x] = acc;
}

template <typename F>
static float time_ms(F&& f) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    f(); cudaDeviceSynchronize();          // warmup
    cudaEventRecord(s); f(); cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    int sms = p.multiProcessorCount;
    printf("GPU %s, cc %d.%d, SMs=%d\n", p.name, p.major, p.minor, sms);

    int grid = sms * 4, block = 256;
    int warps = grid * block / 32;
    float* out = nullptr;
    CUDA_CHECK(cudaMalloc(&out, 1024));

    {   // BF16 peak
        int iters = 100000;
        double flop_per_iter_per_warp = 4096.0 * 8;   // m16n8k16 = 4096 FLOP/warp, 8 acc
        double total_flop = (double)iters * flop_per_iter_per_warp * warps;
        auto launch = [&]{ peak_bf16_kernel<<<grid, block>>>(out, iters); };
        float ms = time_ms(launch);
        printf("BF16 peak : %.3f ms, %.2f TFLOPS\n", ms, total_flop / (ms / 1e3) / 1e12);
    }
    {   // TF32 peak
        int iters = 100000;
        double flop_per_iter_per_warp = 2048.0 * 8;   // m16n8k8 = 2048 FLOP/warp, 8 acc
        double total_flop = (double)iters * flop_per_iter_per_warp * warps;
        auto launch = [&]{ peak_tf32_kernel<<<grid, block>>>(out, iters); };
        float ms = time_ms(launch);
        printf("TF32 peak : %.3f ms, %.2f TFLOPS\n", ms, total_flop / (ms / 1e3) / 1e12);
    }
    {   // mem read bandwidth
        size_t bytes = 1024ull * 1024 * 1024;
        float4* buf;
        CUDA_CHECK(cudaMalloc(&buf, bytes));
        long long n4 = bytes / sizeof(float4);
        auto launch = [&]{ bw_read_kernel<<<sms * 32, 256>>>(buf, out, n4); };
        float ms = time_ms(launch);
        printf("Mem read BW : %.3f ms, %.2f GB/s\n", ms, (double)bytes / (ms / 1e3) / 1e9);
        cudaFree(buf);
    }

    cudaFree(out);
    return 0;
}