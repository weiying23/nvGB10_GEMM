// fp8/probe_sw64.cu —— v5 前置探针：SWIZZLE_64B 的 TMA 物理布局公式验证
//
// BK=64 后 A/B tile 行距 = 64B（4 个 16B chunk）。SWIZZLE_128B 的公式（chunk ^ (r&7)，v10 验证）
// 不能直接用（XOR 结果会越出 4-chunk 行）。本探针用已知模式数据 TMA 装载后 dump smem，
// 逐位推断 SW64 的 XOR 规律。预期候选：chunk_phys = chunk ^ (row & 3)。
//
// 编译：nvcc -O3 -arch=sm_121 -lcuda -o fp8/probe_sw64 fp8/probe_sw64.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)
#define CK_DRV(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { \
    const char* s; cuGetErrorString(e, &s); fprintf(stderr, "CU error %s @%d\n", s, __LINE__); return 1; } } while(0)

// kernel：TMA 装一个 {64, 8} 的 box（64 K 字节 × 8 行）到 smem（SWIZZLE_64B），然后 dump
__global__ void probe(const __grid_constant__ CUtensorMap tmap, unsigned char* out) {
    __shared__ __align__(128) uint8_t smem[512];
    __shared__ uint64_t bar;
    if (threadIdx.x == 0) {
        uint32_t b = (uint32_t)__cvta_generic_to_shared(&bar);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(b));
        asm volatile("fence.mbarrier_init.release.cluster;\n");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t b = (uint32_t)__cvta_generic_to_shared(&bar);
        uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 512;\n" :: "r"(b));
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {0, 0}], [%2];\n" :: "r"(s), "l"(&tmap), "r"(b));
        // 等 TMA 完成
        uint32_t done = 0;
        while (!done)
            asm volatile("{\n\t.reg .pred p;\n\t"
                         "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n\t"
                         "selp.b32 %0, 1, 0, p;\n\t}\n" : "=r"(done) : "r"(b));
    }
    __syncthreads();
    // dump：行 r 的 4 个 16B chunk 里，每个字节填的是逻辑 (row, k)
    // 数据构造：host 侧 data[r][k] = r*64 + k（r<8, k<64）
    for (int i = threadIdx.x; i < 512; i += blockDim.x)
        out[i] = smem[i];
}

int main() {
    // host 数据：8 行 × 64 字节，data[r][k] = (r<<6) | k  → 可从字节值反解 (r, k)
    std::vector<unsigned char> h(8 * 64);
    for (int r = 0; r < 8; ++r)
        for (int k = 0; k < 64; ++k)
            h[r * 64 + k] = (unsigned char)((r << 6) | k);

    unsigned char* d;
    CK(cudaMalloc(&d, h.size()));
    CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));

    CUtensorMap tmap;
    cuuint64_t gdim[2] = {64, 8};          // 内维 64 字节 × 外维 8 行
    cuuint64_t gstr[1] = {64};             // 行步长 64B
    cuuint32_t box[2] = {64, 8};           // 整 box
    cuuint32_t es[2] = {1, 1};
    CK_DRV(cuTensorMapEncodeTiled(&tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        (void*)d, gdim, gstr, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    unsigned char* dout;
    CK(cudaMalloc(&dout, 512));
    probe<<<1, 128>>>(tmap, dout);
    CK(cudaDeviceSynchronize());
    unsigned char got[512];
    CK(cudaMemcpy(got, dout, 512, cudaMemcpyDeviceToHost));

    // 分析：物理 (row, chunk) 位置上的数据来自逻辑 (r', k')
    // 预期公式：chunk_phys = chunk_logical ^ (r' & 3)？—— 用数据反推
    printf("== SW64 物理布局（行 r 的物理 chunk c 存的逻辑 k 范围）==\n");
    for (int r = 0; r < 8; ++r) {
        printf("行 %d: ", r);
        for (int c = 0; c < 4; ++c) {
            // 物理位置 r*64 + c*16 处的字节 → 逻辑 (lr, lk)
            int byte0 = got[r * 64 + c * 16];
            int lr = byte0 >> 6, lk = byte0 & 63;
            printf("[p%d]=逻辑(r%d,k%d) ", c, lr, lk);
        }
        printf("\n");
    }
    // 验证候选公式：物理 chunk c 的内容 = 逻辑 chunk (c ^ (r & 3))？
    printf("== 候选公式验证：物理 c 的逻辑 chunk = c ^ (r & 3)？ ==\n");
    bool ok = true;
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 4; ++c) {
            int byte0 = got[r * 64 + c * 16];
            int lr = byte0 >> 6, lk = byte0 & 63;
            int logical_chunk = lk / 16;
            int expect = c ^ (r & 3);
            if (lr != r || logical_chunk != expect) {
                printf("  反例 r=%d c=%d: got 逻辑(r%d,chunk%d) 期望 chunk=%d\n",
                       r, c, lr, logical_chunk, expect);
                ok = false;
            }
        }
    printf("%s\n", ok ? "[PASS] SW64 公式 = 物理 chunk c ← 逻辑 chunk (c ^ (r & 3))，行号不变"
                      : "[FAIL] 候选公式不成立，见上方反例");
    return 0;
}
