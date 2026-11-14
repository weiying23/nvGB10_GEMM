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

constexpr int NUM_STAGES = 2;         // 48KB/stage × 2 = 96KB（同 GEMM）
constexpr int TILE_BYTES = 128 * 64 * 2 + 4 * (64 * 64 * 2);  // A box 16KB + 4 个 B box 32KB = 48KB

__global__ void __launch_bounds__(256, 1) loader(const __grid_constant__ CUtensorMap tma_a,
                                              const __grid_constant__ CUtensorMap tma_b,
                                              float* out, int k_blocks, int m_blocks, int n_blocks, int reps) {
    extern __shared__ __align__(1024) uint8_t smem[];
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + NUM_STAGES * TILE_BYTES);
    uint64_t* empty = full + NUM_STAGES;
    const int tid = threadIdx.x;
    if (tid == 0) for (int i = 0; i < NUM_STAGES; ++i) { mbar_init_(&full[i], 1); mbar_init_(&empty[i], 256); }
    asm volatile("fence.mbarrier_init.release.cluster;\n");
    __syncthreads();

    // 每 block 固定处理自己的一行 m-tile（block_m = bid % m_blocks），kb 循环 + reps 轮
    const int my_mb = blockIdx.x % m_blocks;
    const int my_nb = (blockIdx.x / m_blocks) % n_blocks;
    const long long total = (long long)reps * k_blocks;
    // 每 stage 装载 = A box(16KB) + 4 个 B box(32KB)，同 GEMM issue_stage
    auto issue_stage = [&](int s, long long idx) {
        int kb = (int)(idx % k_blocks);
        uint8_t* a_dst = smem + s * TILE_BYTES;
        uint8_t* b_dst = a_dst + 128 * 64 * 2;
        mbar_expect_(&full[s], TILE_BYTES);
        tma_load_(&tma_a, a_dst, &full[s], kb * 64, my_mb * 128);
        for (int g = 0; g < 4; ++g)
            tma_load_(&tma_b, b_dst + g * (64 * 64 * 2), &full[s], my_nb * 256 + g * 64, kb * 64);
    };

    uint32_t ph[NUM_STAGES] = {0};
    if (tid == 0) {
        // 生产者兼消费者：先发 NUM_STAGES 个，之后每步 消费一份 → 补发一个
        uint32_t pe[NUM_STAGES] = {0};
        for (long long i = 0; i < NUM_STAGES && i < total; ++i)
            issue_stage((int)(i % NUM_STAGES), i);
        for (long long c = 0; c < total; ++c) {
            int s = (int)(c % NUM_STAGES);
            mbar_wait_(&full[s], ph[s]); ph[s] ^= 1;   // tid0 的消费份额
            mbar_arr_(&empty[s]);                       // 喂 empty（256 份之一）
            long long next = c + NUM_STAGES;
            if (next < total) {
                mbar_wait_(&empty[s], pe[s]); pe[s] ^= 1;   // 256 到齐 = 可覆写
                issue_stage(s, next);
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

static int make_map_B(CUtensorMap* m, const bf16* p, int K, int N) {
    cuuint64_t gd[2] = {(cuuint64_t)N, (cuuint64_t)K};
    cuuint64_t gs[1] = {(cuuint64_t)N * 2};
    cuuint32_t bx[2] = {64, 64}, es[2] = {1, 1};
    CKD(cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)p, gd, gs, bx, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return 0;
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

static double bench(int M, int N, int K, int reps) {
    bf16* d; CK(cudaMalloc(&d, (size_t)M * K * 2)); CK(cudaMemset(d, 1, (size_t)M * K * 2));
    bf16* db; CK(cudaMalloc(&db, (size_t)K * N * 2)); CK(cudaMemset(db, 1, (size_t)K * N * 2));
    CUtensorMap tm, tmb; make_map(&tm, d, M, K); make_map_B(&tmb, db, K, N);
    float* out; CK(cudaMalloc(&out, 4));
    static bool attr_set = false;
    if (!attr_set) { CK(cudaFuncSetAttribute(loader, cudaFuncAttributeMaxDynamicSharedMemorySize, NUM_STAGES * TILE_BYTES + 2 * NUM_STAGES * 8)); attr_set = true; }
    int kb = K / 64, mb = M / 128, nb = N / 256;
    loader<<<48, 256, NUM_STAGES * TILE_BYTES + 2 * NUM_STAGES * 8>>>(tm, tmb, out, kb, mb, nb, reps);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    loader<<<48, 256, NUM_STAGES * TILE_BYTES + 2 * NUM_STAGES * 8>>>(tm, tmb, out, kb, mb, nb, reps);
    CK(cudaGetLastError());
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    double boxes = (double)reps * kb * 48;          // 48 个 block 各自装载 A+B stage
    double gbps = boxes * TILE_BYTES / (ms / 1e3) / 1e9;
    cudaFree(d); cudaFree(db); cudaFree(out);
    return gbps;
}

int main() {
    int reps = 40;
    printf("===== A+B 双流 TMA 装载吞吐（stage = A 1 box + B 4 box，48 blocks 同 GEMM） =====\n");
    double p2  = bench(4096, 4096, 4096, reps);   // A+B=64MB >> L2
    double np2 = bench(4096, 4096, 4032, reps);
    printf("  [DRAM 域] A+B=64MB:  K=4096 %.1f GB/s  vs  K=4032 %.1f GB/s  → 惩罚 %.1f%%\n", p2, np2, (1-p2/np2)*100);
    int reps_l2 = 400;                             // 工作集小，加大 reps
    double q2  = bench(1024, 1024, 1024, reps_l2);  // A+B=16MB < L2 24MB（行距 2048B=2^11）
    double qnp = bench(1024, 1024, 1008, reps_l2);   // A 行距 2016B（63×32）
    printf("  [L2 域]   A+B=16MB:  K=1024 %.1f GB/s  vs  K=1008 %.1f GB/s  → 惩罚 %.1f%%\n", q2, qnp, (1-q2/qnp)*100);
    return 0;
}
