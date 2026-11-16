// bench/jit_kernel.cu —— NVRTC 形状特化版 v12 kernel（探针 ⑦ 第 1 步：常量烘焙）
//
// 本文件不是给 nvcc 编译的，而是被 bench/probe_jit.cu 在运行时读出、交给 NVRTC，
// 以 -DM_GLOB/-DN_GLOB/-DK_GLOB/-DUSE_SMALL 烘焙形状与配置后生成 CUBIN。
//
// 设备代码与 gemm_v12.cu 逐行同源，仅 4 处差异：
//   1) M/N/K 为编译期常量（-D 注入）→ 所有分块/边界/K 块数计算常量折叠，
//      编译器可对主循环做首尾 peeling（1024³ 缺口 = per-K-block 固定成本，主打此项）；
//   2) tensor map 按指针传（NVRTC 侧无 cuda.h，CUtensorMap 自定义 128B 不透明结构，
//      host 侧把 cuTensorMapEncodeTiled 的产物拷到 device 全局内存）；
//   3) 配置二选一经 -DUSE_SMALL 注入（单配置编译，符号 extern "C" 便于 cuModuleGetFunction）；
//   4) min() → jit_umin()（避开 NVRTC/nvcc 双路头依赖）、make_float2 → 本地 F2 结构（同上）。
// 其余（swizzle 公式、mbarrier 协议、fence 位置、epilogue 4-pass）与 v12 完全一致，
// 保证性能差异只来自「常量烘焙」这一个变量。

#include <cuda_bf16.h>

// NVRTC 无标准 C++ 头（cstdint 不可用）：自行定义定宽类型；
// nvcc 诊断路径（-Xptxas -v）下用真头，便于同一文件双路编译
// （注意：NVRTC 不预定义 __NVRTC__，由 probe_jit 显式传 -DJIT_NVRTC）
#ifdef JIT_NVRTC
typedef unsigned char       uint8_t;
typedef unsigned int        uint32_t;
typedef unsigned long long  uint64_t;
#else
#include <cstdint>
#endif

using bf16 = __nv_bfloat16;

#ifndef M_GLOB
#define M_GLOB 512
#endif
#ifndef N_GLOB
#define N_GLOB 512
#endif
#ifndef K_GLOB
#define K_GLOB 512
#endif
#ifndef USE_SMALL
#define USE_SMALL 1
#endif

#define BK 64
#define NUM_STAGES 2

// NVRTC 无 cuda.h：128B 不透明结构，仅按指针使用
struct alignas(64) CUtensorMap { unsigned char __opaque[128]; };

// ---- 形状常量（本探针的唯一自变量）----
constexpr int M = M_GLOB, N = N_GLOB, K = K_GLOB;

// ---- 配置（v12 的 BigCfg/SmallCfg 二选一）----
#if USE_SMALL
constexpr int BM = 128, BN = 64, WM = 16, WN = 64;   // SmallCfg（小形状补填充）
#else
constexpr int BM = 128, BN = 256, WM = 64, WN = 64;  // BigCfg（大形状）
#endif
constexpr int THREADS = 256, NUM_WARPS = THREADS / 32;
constexpr int NUM_M = WM / 16, NUM_N = WN / 8, NUM_B_CHUNKS = BN / 64;
constexpr int WARP_N_COUNT = NUM_B_CHUNKS;
constexpr int WARP_M_COUNT = NUM_WARPS / WARP_N_COUNT;
static_assert(WM == BM / WARP_M_COUNT, "WM must match warp grid");
static_assert(WARP_M_COUNT * WARP_N_COUNT == NUM_WARPS, "warp grid");
constexpr int A_STAGE_BYTES  = BM * 128;
constexpr int B_CHUNK_BYTES  = 64 * 128;
constexpr int B_STAGE_BYTES  = NUM_B_CHUNKS * B_CHUNK_BYTES;
constexpr int TMA_BYTES_PER_STAGE = A_STAGE_BYTES + B_STAGE_BYTES;
constexpr int SMEM_BYTES = NUM_STAGES * TMA_BYTES_PER_STAGE + 2 * NUM_STAGES * 8;

__device__ __forceinline__ uint32_t jit_umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

// ---- mbarrier（与 v12 逐字一致）----
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t c) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(a), "r"(c));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* b, uint32_t bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(a), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(a));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
    uint32_t done = 0;
    while (!done) {
        asm volatile("{\n\t.reg .pred p;\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
                     "selp.b32 %0, 1, 0, p;\n\t}\n" : "=r"(done) : "r"(a), "r"(ph));
    }
}
__device__ __forceinline__ void fence_mbar_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n");
}

// ---- TMA 2D load / store（与 v12 逐字一致，tmap 即指针）----
__device__ __forceinline__ void tma_load_2d(const void* tmap, void* smem_dst,
                                            uint64_t* mbar, int x, int y) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_dst);
    uint32_t m = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(s), "l"(tmap), "r"(x), "r"(y), "r"(m));
}
__device__ __forceinline__ void tma_store_2d(const void* tmap, const void* smem_src, int x, int y) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_src);
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                 :: "l"(tmap), "r"(x), "r"(y), "r"(s));
}
__device__ __forceinline__ void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;\n"); }
template<int NG>
__device__ __forceinline__ void tma_store_wait() { asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(NG)); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n"); }

struct F2 { float x, y; };   // 替代 make_float2（NVRTC 头依赖规避）

extern "C" __global__ void __launch_bounds__(THREADS, 1) gemm_jit(
    const CUtensorMap* tma_a, const CUtensorMap* tma_b, const CUtensorMap* tma_c) {
    // ---- block swizzle（M/N 为编译期常量 → 分块数与组运算全部折叠）----
    constexpr uint32_t GROUP_M = 8;
    const uint32_t num_m_blocks = (M + BM - 1) / BM;
    const uint32_t num_n_blocks = (N + BN - 1) / BN;
    const uint32_t bid = blockIdx.x;
    const uint32_t group_size = GROUP_M * num_n_blocks;
    const uint32_t group_id = bid / group_size;
    const uint32_t in_group = bid % group_size;
    const uint32_t m_first = jit_umin(group_id * GROUP_M, num_m_blocks);
    const uint32_t m_g = jit_umin(GROUP_M, num_m_blocks - m_first);
    const uint32_t m_block_idx = m_first + in_group % m_g;
    const uint32_t n_block_idx = in_group / m_g;
    if (n_block_idx >= num_n_blocks) return;    // 常量折叠后精确形状下为死代码
    const int block_m = m_block_idx * BM;
    const int block_n = n_block_idx * BN;

    // ---- smem 布局：每 stage 连续 [A | B]，再接 mbarrier ----
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(smem_raw + NUM_STAGES * TMA_BYTES_PER_STAGE);
    uint64_t* empty_bar = full_bar + NUM_STAGES;

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / NUM_B_CHUNKS;
    const int warp_n = warp % NUM_B_CHUNKS;
    static_assert(WARP_M_COUNT * WARP_N_COUNT == NUM_WARPS, "warp grid");

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < NUM_STAGES; ++i) {
            mbar_init(&full_bar[i], 1);
            mbar_init(&empty_bar[i], NUM_WARPS);
        }
    }
    fence_mbar_init();
    __syncthreads();

    const int num_k_blocks = (K + BK - 1) / BK;   // 编译期常量 → 主循环可被 peeling 优化

    // producer：装一个 stage（A 1 条 + B NUM_B_CHUNKS 条 TMA）
    auto issue_stage = [&](int s, int kb) {
        uint8_t* a_dst = smem_raw + s * TMA_BYTES_PER_STAGE;
        uint8_t* b_dst = a_dst + A_STAGE_BYTES;
        mbar_arrive_expect_tx(&full_bar[s], TMA_BYTES_PER_STAGE);
        tma_load_2d(tma_a, a_dst, &full_bar[s], kb * BK, block_m);
        #pragma unroll
        for (int g = 0; g < NUM_B_CHUNKS; ++g)
            tma_load_2d(tma_b, b_dst + g * B_CHUNK_BYTES, &full_bar[s],
                        block_n + g * 64, kb * BK);
    };

    // 累加器
    float c[NUM_M][NUM_N][4];
    #pragma unroll
    for (int i = 0; i < NUM_M; ++i)
        #pragma unroll
        for (int j = 0; j < NUM_N; ++j) c[i][j][0] = c[i][j][1] = c[i][j][2] = c[i][j][3] = 0.0f;

    // prologue：预取 stage 0
    if (tid == 0 && num_k_blocks > 0) issue_stage(0, 0);

    uint32_t phase[NUM_STAGES] = {0};
    uint32_t prod_empty_phase[NUM_STAGES] = {0};

    // ---- 主循环（零 __syncthreads，BK=64 分 2 个 pair 处理）----
    for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int s = kb % NUM_STAGES;

        // ① producer（thread 0）：等 empty → 装下一 stage
        if (tid == 0 && kb + 1 < num_k_blocks) {
            int ns = (kb + 1) % NUM_STAGES;
            if (kb >= 1) {
                mbar_wait(&empty_bar[ns], prod_empty_phase[ns]);
                prod_empty_phase[ns] ^= 1;
            }
            issue_stage(ns, kb + 1);
        }

        // ② 等当前 stage 数据就绪
        mbar_wait(&full_bar[s], phase[s]);
        phase[s] ^= 1;

        const uint8_t* a_base = smem_raw + s * TMA_BYTES_PER_STAGE;
        const uint8_t* b_base = a_base + A_STAGE_BYTES;
        const int atile = lane / 8, ar8 = lane % 8;

        // ③ 2 个 pair × (16 ldsm + 64 mma)
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            uint32_t a[2][NUM_M][4];
            #pragma unroll
            for (int kii = 0; kii < 2; ++kii) {
                const int ks = p * 2 + kii;
                #pragma unroll
                for (int mi = 0; mi < NUM_M; ++mi) {
                    const int r = warp_m * WM + mi * 16 + ar8 + (atile % 2) * 8;
                    const int sub = ks * 2 + atile / 2;
                    uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                        a_base + r * 128 + (sub ^ (r & 7)) * 16);
                    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(a[kii][mi][0]), "=r"(a[kii][mi][1]), "=r"(a[kii][mi][2]), "=r"(a[kii][mi][3]) : "r"(addr));
                }
            }
            uint32_t b[2][4][4];
            #pragma unroll
            for (int kii = 0; kii < 2; ++kii) {
                const int ks = p * 2 + kii;
                #pragma unroll
                for (int nj = 0; nj < 4; ++nj) {
                    const int k = ks * 16 + lane % 8 + (lane / 8 % 2) * 8;
                    const int nsub = nj * 2 + (lane / 8 / 2);
                    uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                        b_base + warp_n * B_CHUNK_BYTES + k * 128 + (nsub ^ (k & 7)) * 16);
                    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(b[kii][nj][0]), "=r"(b[kii][nj][1]), "=r"(b[kii][nj][2]), "=r"(b[kii][nj][3]) : "r"(addr));
                }
            }
            #pragma unroll
            for (int kii = 0; kii < 2; ++kii)
                #pragma unroll
                for (int mi = 0; mi < NUM_M; ++mi)
                    #pragma unroll
                    for (int nj = 0; nj < 4; ++nj)
                        #pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            int ni = nj * 2 + half;
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                         : "+f"(c[mi][ni][0]), "+f"(c[mi][ni][1]), "+f"(c[mi][ni][2]), "+f"(c[mi][ni][3])
                                         : "r"(a[kii][mi][0]), "r"(a[kii][mi][1]), "r"(a[kii][mi][2]), "r"(a[kii][mi][3]),
                                           "r"(b[kii][nj][half*2]), "r"(b[kii][nj][half*2+1]));
                        }
        }

        // ④ 释放 stage（fence：SM121 上 arrive 必须在 LDSM 读完成之后）
        asm volatile("fence.acq_rel.cta;\n");
        if (lane == 0) mbar_arrive(&empty_bar[s]);
    }

    // ---- epilogue：smem 暂存 + TMA store（4 pass × 32 行，双 stage 流水）----
    __syncthreads();
    const int crow = lane / 4, ccol = (lane % 4) * 2;
    for (int p = 0; p < (BM + 31) / 32; ++p) {
        float* cstg = reinterpret_cast<float*>(smem_raw + (p % NUM_STAGES) * TMA_BYTES_PER_STAGE)
                      + warp_n * (32 * 64);
        #pragma unroll
        for (int mi = 0; mi < NUM_M; ++mi) {
            const int r0 = warp_m * WM + mi * 16;
            if (r0 < 32 * p || r0 >= 32 * p + 32) continue;
            #pragma unroll
            for (int ni = 0; ni < NUM_N; ++ni) {
                const int rr = r0 - 32 * p + crow;
                const int cc = ni * 8 + ccol;
                F2 v01 = {c[mi][ni][0], c[mi][ni][1]};
                F2 v23 = {c[mi][ni][2], c[mi][ni][3]};
                *reinterpret_cast<F2*>(&cstg[rr * 64 + cc]) = v01;
                *reinterpret_cast<F2*>(&cstg[(rr + 8) * 64 + cc]) = v23;
            }
        }
        fence_proxy_async();
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < NUM_B_CHUNKS; ++g)
                tma_store_2d(tma_c, reinterpret_cast<float*>(smem_raw + (p % NUM_STAGES) * TMA_BYTES_PER_STAGE) + g * (32 * 64),
                             block_n + g * 64, block_m + 32 * p);
            tma_store_commit();
            tma_store_wait<1>();
        }
        __syncthreads();
    }
    if (tid == 0) tma_store_wait<0>();
    __syncthreads();
}
