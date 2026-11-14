// 2 的幂病理最终判别：TMA box 装载吞吐 vs K 的 2 幂性
// 与 GEMM 相同的 box 形态 {64 K 列, 128 M 行}，swizzle128，mbarrier full/empty 流水
// 无 mma/LDSM/epilogue —— 若 K=4096 比 K=4032 慢，病理定位在 TMA 装载路径本身
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
using bf16 = __nv_bfloat16;
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;} } while(0)
#define CKD(x) do { CUresult e=(x); if(e!=CUDA_SUCCESS){const char*s;cuGetErrorString(e,&s);fprintf(stderr,"CU %s @%d\n",s,__LINE__);return 1;} } while(0)

__device__ __forceinline__ void mbar_init_(uint64_t* b, uint32_t c) {
    uint32_t a=(uint32_t)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"::"r"(a),"r"(c));
}
__device__ __forceinline__ void mbar_expect_(uint64_t* b, uint32_t bytes) {
    uint32_t a=(uint32_t)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"::"r"(a),"r"(bytes));
}
__device__ __forceinline__ void mbar_arr_(uint64_t* b) {
    uint32_t a=(uint32_t)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n"::"r"(a));
}
__device__ __forceinline__ void mbar_wait_(uint64_t* b, uint32_t ph) {
    uint32_t a=(uint32_t)__cvta_generic_to_shared(b), done=0;
    while(!done) asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.b32 %0, 1, 0, p;\n\t}\n":"=r"(done):"r"(a),"r"(ph));
}
__device__ __forceinline__ void tma_load_(const void* t, void* s, uint64_t* m, int x, int y) {
    uint32_t sd=(uint32_t)__cvta_generic_to_shared(s), md=(uint32_t)__cvta_generic_to_shared(m);
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 ::"r"(sd),"l"(t),"r"(x),"r"(y),"r"(md));
}

constexpr int NUM_STAGES = 3;         // 16KB/stage × 3 = 48KB
constexpr int TILE_BYTES = 128 * 64 * 2;  // A box 16KB

__global__ void __launch_bounds__(256, 1) loader(const __grid_constant__ CUtensorMap tma_a,
                                              float* out, int k_blocks, int m_blocks, int reps) {
    extern __shared__ __align__(1024) uint8_t smem[];
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + NUM_STAGES * TILE_BYTES);
    uint64_t* empty = full + NUM_STAGES;
    const int tid = threadIdx.x;
    if (tid == 0) for (int i = 0; i < NUM_STAGES; ++i) { mbar_init_(&full[i], 1); mbar_init_(&empty[i], 256); }
    asm volatile("fence.mbarrier_init.release.cluster;\n");
    __syncthreads();

    // 每 block 固定处理自己的一行 m-tile（block_m = bid % m_blocks），kb 循环 + reps 轮
    const int my_mb = blockIdx.x % m_blocks;
    const long long total = (long long)reps * k_blocks;

    uint32_t ph[NUM_STAGES] = {0};
    if (tid == 0) {
        // 生产者兼消费者：先发 NUM_STAGES 个，之后每步 消费一份 → 补发一个
        uint32_t pe[NUM_STAGES] = {0};
        for (long long i = 0; i < NUM_STAGES && i < total; ++i) {
            int s = (int)(i % NUM_STAGES);
            mbar_expect_(&full[s], TILE_BYTES);
            tma_load_(&tma_a, smem + s * TILE_BYTES, &full[s], (int)((i % k_blocks) * 64), my_mb * 128);
        }
        for (long long c = 0; c < total; ++c) {
            int s = (int)(c % NUM_STAGES);
            mbar_wait_(&full[s], ph[s]); ph[s] ^= 1;   // tid0 的消费份额
            mbar_arr_(&empty[s]);                       // 喂 empty（256 份之一）
            long long next = c + NUM_STAGES;
            if (next < total) {
                mbar_wait_(&empty[s], pe[s]); pe[s] ^= 1;   // 256 到齐 = 可覆写
                mbar_expect_(&full[s], TILE_BYTES);
                tma_load_(&tma_a, smem + s * TILE_BYTES, &full[s], (int)((next % k_blocks) * 64), my_mb * 128);
            }
        }
    } else {
        for (long long c = 0; c < total; ++c) {
            int s = (int)(c % NUM_STAGES);
            mbar_wait_(&full[s], ph[s]); ph[s] ^= 1;
            mbar_arr_(&empty[s]);
        }
    }
    if (tid == 0) out[0] = 1.f;
}

static int make_map(CUtensorMap* m, const bf16* p, int M, int K) {
    cuuint64_t gd[2] = {(cuuint64_t)K, (cuuint64_t)M};
    cuuint64_t gs[1] = {(cuuint64_t)K * 2};
    cuuint32_t bx[2] = {64, 128}, es[2] = {1, 1};
    CKD(cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)p, gd, gs, bx, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return 0;
}

static double bench(int M, int K, int reps) {
    bf16* d; CK(cudaMalloc(&d, (size_t)M * K * 2)); CK(cudaMemset(d, 1, (size_t)M * K * 2));
    CUtensorMap tm; make_map(&tm, d, M, K);
    float* out; CK(cudaMalloc(&out, 4));
    static bool attr_set = false;
    if (!attr_set) { CK(cudaFuncSetAttribute(loader, cudaFuncAttributeMaxDynamicSharedMemorySize, NUM_STAGES * TILE_BYTES + 2 * NUM_STAGES * 8)); attr_set = true; }
    int kb = K / 64, mb = M / 128;
    loader<<<48, 256, NUM_STAGES * TILE_BYTES + 2 * NUM_STAGES * 8>>>(tm, out, kb, mb, reps);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    loader<<<48, 256, NUM_STAGES * TILE_BYTES + 2 * NUM_STAGES * 8>>>(tm, out, kb, mb, reps);
    CK(cudaGetLastError());
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    double boxes = (double)reps * kb * mb * 48;   // 48 个 block 各自装载
    double gbps = boxes * TILE_BYTES / (ms / 1e3) / 1e9;
    cudaFree(d); cudaFree(out);
    return gbps;
}

int main() {
    int reps = 120;
    printf("===== TMA box 装载吞吐（A box {64,128}，48 blocks 同 GEMM） =====\n");
    double p2  = bench(4096, 4096, reps);   // A 行距 8192B = 2^13
    double np2 = bench(4096, 4032, reps);  // A 行距 8064B
    double p2b = bench(4096, 2048, reps);  // A 行距 4096B = 2^12
    double np2b = bench(4096, 2080, reps);// A 行距 4160B = 65×64
    printf("  K=4096 (2^12):  %.1f GB/s     K=4032:        %.1f GB/s   → 惩罚 %.1f%%\n", p2, np2, (1-p2/np2)*100);
    printf("  K=2048 (2^11):  %.1f GB/s     K=2080:        %.1f GB/s   → 惩罚 %.1f%%\n", p2b, np2b, (1-p2b/np2b)*100);
    return 0;
}
