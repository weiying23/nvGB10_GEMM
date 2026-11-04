// TMA SWIZZLE_128B 落位公式验证探针
// 假设公式：元素 (row r, 内维元素 c)（行距 128B = 64 个 bf16）：
//   物理字节偏移 = r*128 + ((c/8) ^ (r%8))*16 + (c%8)*2     （16B 子槽被行号 XOR 交错）
// 验证法：TMA 加载已知数据 → dump 原始 smem 到 global → host 按公式重建期望布局 → 逐字节比对
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>
using bf16 = __nv_bfloat16;
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;} } while(0)
#define CKD(x) do { CUresult e=(x); if(e!=CUDA_SUCCESS){const char*s;cuGetErrorString(e,&s);fprintf(stderr,"CU err %s @%d\n",s,__LINE__);return 1;} } while(0)

__device__ __forceinline__ void mbar_init_(uint64_t* b, uint32_t c) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(c));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(a), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait_(uint64_t* b, uint32_t ph) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(b), done = 0;
    while (!done)
        asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.b32 %0, 1, 0, p;\n\t}\n" : "=r"(done) : "r"(a), "r"(ph));
}
__device__ __forceinline__ void tma_load_(const void* tmap, void* smem, uint64_t* mbar, int x, int y) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t m = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];\n" :: "r"(s), "l"(tmap), "r"(x), "r"(y), "r"(m));
}

// dump 内核：把 box 加载进 smem（走 TMA + SWIZZLE_128B），再原样拷到 global
__global__ void dump_kernel(const __grid_constant__ CUtensorMap tma, bf16* out, int box_h, int box_w) {
    __shared__ __align__(1024) uint8_t smem[32768];
    __shared__ __align__(8) uint64_t bar;
    if (threadIdx.x == 0) mbar_init_(&bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n");
    __syncthreads();
    uint32_t bytes = box_h * box_w * 2;
    if (threadIdx.x == 0) { mbar_expect_tx(&bar, bytes); tma_load_(&tma, smem, &bar, 0, 0); }
    mbar_wait_(&bar, 0);
    // 原样 dump（不做任何公式解读）
    bf16* sb = reinterpret_cast<bf16*>(smem);
    for (int i = threadIdx.x; i < bytes/2; i += 256) out[i] = sb[i];
}

static int make_map(CUtensorMap* map, const bf16* dptr, cuuint64_t gdim0, cuuint64_t gdim1,
                     cuuint32_t box0, cuuint32_t box1, CUtensorMapSwizzle sw) {
    cuuint64_t gdim[2] = { gdim0, gdim1 };
    cuuint64_t gstr[1] = { gdim0 * 2 };
    cuuint32_t box[2] = { box0, box1 }, es[2] = { 1, 1 };
    CKD(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)dptr,
        gdim, gstr, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return 0;
}

// host 侧按公式重建期望布局
static void expect_layout(const std::vector<bf16>& src, int h, int w, std::vector<bf16>& dst) {
    dst.assign((size_t)h * w * 2, bf16(0));   // 字节粒度
    uint8_t* d = reinterpret_cast<uint8_t*>(dst.data());
    const uint8_t* s = reinterpret_cast<const uint8_t*>(src.data());
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
            int log_off = r * w * 2 + c * 2;             // 逻辑字节偏移
            int phys = r * 128 + ((c / 8) ^ (r % 8)) * 16 + (c % 8) * 2;
            d[phys] = s[log_off]; d[phys+1] = s[log_off+1];
        }
}

int main() {
    // 用例 1：A 型 box {64, 128}（内维 64 元素 = 128B，外维 128 行）
    // 用例 2：B 型 box {64, 64}
    struct { int w, h; const char* name; } cases[] = { {64, 128, "A型 box{64,128}"}, {64, 64, "B型 box{64,64}"} };
    for (auto& cs : cases) {
        int n = cs.w * cs.h;
        std::vector<bf16> h(n);
        for (int i = 0; i < n; ++i) h[i] = __float2bfloat16((float)((i * 37 + 11) % 251));  // 已知模式
        bf16 *d, *out;
        CK(cudaMalloc(&d, n * 2)); CK(cudaMalloc(&out, n * 2));
        CK(cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice));
        CUtensorMap map;
        make_map(&map, d, cs.w, cs.h, cs.w, cs.h, CU_TENSOR_MAP_SWIZZLE_128B);
        dump_kernel<<<1, 256>>>(map, out, cs.h, cs.w);
        CK(cudaDeviceSynchronize());
        std::vector<bf16> got(n);
        CK(cudaMemcpy(got.data(), out, n * 2, cudaMemcpyDeviceToHost));
        std::vector<bf16> expv;
        expect_layout(h, cs.h, cs.w, expv);
        int bad = 0;
        for (int i = 0; i < n; ++i)
            if (got[i] != expv[i]) { if (bad < 4) printf("  首个差异 idx=%d got=%f exp=%f\n", i, __bfloat162float(got[i]), __bfloat162float(expv[i])); ++bad; }
        printf("%-18s: %s（%d/%d 元素不符）\n", cs.name, bad == 0 ? "公式验证通过 ✓" : "公式错误 ✗", bad, n);
        cudaFree(d); cudaFree(out);
    }
    return 0;
}
