// 2 的幂步长病理分层判别探针 v2（修 v1 的两个伪影：除法环绕/DRAM 工况进 L2）
// 访问模式：每行读 128B，行距 = stride 字节（模仿 TMA 读 A tile 行段）
//   L2 工况：16384 行(2^14, AND 环绕) × 128B = 2MB 唯一数据（驻留 L2，超 L1）→ 病理在此 = L2 层
//   DRAM 工况：stride × 2M 行单遍扫描，唯一数据 256MB >> L2 24MB → 病理在此 = DRAM 层
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// L2 工况：rep 次循环，行号 = (base + rep*threads) & 16383 —— 两种步长同代价
__global__ void row_read_l2(const uint8_t* __restrict__ buf, float* out,
                            int reps, long long stride_bytes, int nthreads) {
    float acc = 0;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int rep = 0; rep < reps; ++rep) {
        int row = (tid + rep * nthreads) & 16383;
        const float4* p = reinterpret_cast<const float4*>(buf + (long long)row * stride_bytes);
        #pragma unroll
        for (int j = 0; j < 8; ++j) { float4 v = p[j]; acc += v.x + v.y + v.z + v.w; }
    }
    if (acc == -12345.f) out[0] = acc;
}

// DRAM 工况：grid-stride 单遍扫描 2M 行（无环绕、无除法）
__global__ void row_read_dram(const uint8_t* __restrict__ buf, float* out,
                              int rows, long long stride_bytes) {
    float acc = 0;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += gridDim.x * blockDim.x) {
        const float4* p = reinterpret_cast<const float4*>(buf + (long long)r * stride_bytes);
        #pragma unroll
        for (int j = 0; j < 8; ++j) { float4 v = p[j]; acc += v.x + v.y + v.z + v.w; }
    }
    if (acc == -12345.f) out[0] = acc;
}

static double run_l2(long long stride_bytes) {
    size_t buf_bytes = (size_t)16384 * stride_bytes;           // ≈128-132MB
    uint8_t* d; cudaMalloc(&d, buf_bytes); cudaMemset(d, 1, buf_bytes);
    float* out; cudaMalloc(&out, 4);
    int grid = 48 * 32, block = 256, nth = grid * block, reps = 120000;
    row_read_l2<<<grid, block>>>(d, out, 1000, stride_bytes, nth); cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    row_read_l2<<<grid, block>>>(d, out, reps, stride_bytes, nth);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    double rows_s = (double)reps * nth / (ms / 1e3);
    cudaFree(d); cudaFree(out);
    return rows_s * 128 / 1e9;                                 // GB/s
}

static double run_dram(long long stride_bytes) {
    int rows = 2 * 1024 * 1024;                                 // 2M 行
    size_t buf_bytes = (size_t)rows * stride_bytes;             // 16-16.1GB
    uint8_t* d; cudaMalloc(&d, buf_bytes); cudaMemset(d, 1, buf_bytes);
    float* out; cudaMalloc(&out, 4);
    int grid = 48 * 32, block = 256;
    row_read_dram<<<grid, block>>>(d, out, rows, stride_bytes); cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    row_read_dram<<<grid, block>>>(d, out, rows, stride_bytes);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    double rows_s = (double)rows / (ms / 1e3);
    cudaFree(d); cudaFree(out);
    return rows_s * 128 / 1e9;
}

int main() {
    printf("===== L2 工况（唯一数据 2MB 驻留 L2） =====\n");
    double l2_p2  = run_l2(8192);
    double l2_np2 = run_l2(8064);
    printf("  stride 8192B(2^13): %.1f GB/s   stride 8064B(63x128): %.1f GB/s   → 2 幂惩罚 %.1f%%\n",
           l2_p2, l2_np2, (1 - l2_p2 / l2_np2) * 100);
    printf("===== DRAM 工况（唯一数据 256MB，单遍） =====\n");
    double dr_p2  = run_dram(8192);
    double dr_np2 = run_dram(8064);
    printf("  stride 8192B(2^13): %.1f GB/s   stride 8064B(63x128): %.1f GB/s   → 2 幂惩罚 %.1f%%\n",
           dr_p2, dr_np2, (1 - dr_p2 / dr_np2) * 100);
    return 0;
}
