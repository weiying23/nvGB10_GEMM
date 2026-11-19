// fp8/gemm_fp8_v2.cu —— FP8 GEMM v2：配置族 + 双维分派（tile 尺寸 × 2 幂病理）
//
// v01 后的差距解剖（reports/fp8/v01.md + v01 附加探针）：
//   * 512³ 填充病：单 128×256 → 8 block/48 SM = 17%，实测 18.7 vs cuBLASLt 49.5；
//   * 2 幂病理（K%2^k==0 大形状）：128×256 下 -9.5%（164.5 vs 8064 的 182.7）——
//     128×128 tile（B 足迹减半摊薄 L2 slice 压力）实测病理缩到 -3.4%，绝对值反超（169.3）；
//   * 分派维度新增「2 幂性」：与 BF16 线按大小分派不同，这是按「数据布局病理」分派。
//
// v2 改动（v1 骨架零改动）：
//   1. 配置族：BigCfg(128x256) / MidCfg(128x128) / SmallCfg(128x64) 三模板实例化；
//   2. 分派规则：
//        block 数(BigCfg) < 48 → SmallCfg（填充优先，同 BF16 v12 经验）
//        否则 K 为 2 的幂 → MidCfg（病理规避）
//        否则 → BigCfg（非 2 幂大形状最优）
//   3. SmallCfg 的 WM 联动（v12 踩坑在案）：BN=64 → NUM_B_CHUNKS=1 → 8 warp_m → WM=16。
//
// 预期（全部来自 v01 附加探针实测）：512³ 18.7→40+（SmallCfg 填充 67%）；
//   8192³ 164.5→169.3（MidCfg 病理规避）；非 2 幂大形状零回归（仍 BigCfg）。
//
// 立项推导（fp8/README.md §2 + 阶段 A/B/C 探针）：
//   * A 侧：BK=128 的 e4m3 tile 与 v13 BK=64 的 bf16 tile **smem 几何相同**（128 行 × 128B，
//     8 个 16B chunk）——SWIZZLE_128B 公式与 x4 非 trans LDSM 寻址**逐字节照搬**（阶段 A 验证
//     b16 字节对同构）。每 stage 足迹不变：A 16KB + B 32KB = 48KB，2-stage 96KB。
//   * B 侧：量化 kernel 写侧转置（阶段 B 验证 [K][N]→[N][K]），TMA box {128,64}（内维 k 字节），
//     smem 行 = n、行内 = k 字节——B-LDSM 从 v13 的 x4.trans 改为 x4 非 trans（阶段 A 验证路线 b：
//     每 x4 出 2 个 n-group 的 {b0,b1}）。
//   * mma：m16n8k16.bf16 → m16n8k32.e4m3。每 K-block 指令数与 v13 完全相同（16 A-ldsm +
//     16 B-ldsm + 128 mma），但 K-block 覆盖 128 k（2×）——同等指令流吞吐翻倍的理论来源。
//   * scale：per-tensor（sA·sB 在 epilogue 乘一次——hpc-ops 图纸的简单路线）。
//   * 参照系：cuBLASLt FP8 同精度对拍（阶段 C 实测引擎可用：4096³ 182.97 / 8192³ 200.26）。
//
// 约束：K % 16 == 0（TMA uint8 行步长 16B 对齐——比 bf16 线的 %8 更严），M/N % 8。
// 编译：nvcc -O3 -arch=sm_121 -lcuda -lcublasLt -o fp8/gemm_fp8_v1 fp8/gemm_fp8_v1.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

using bf16 = __nv_bfloat16;

#ifndef SWZ
#define SWZ 1   // 1 = SWIZZLE_128B（正常）；0 = 无 swizzle 对照（判别探针）
#endif

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define CK_DRV(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { \
    const char* s; cuGetErrorString(e, &s); fprintf(stderr, "CU error %s @%d\n", s, __LINE__); exit(1); } } while(0)
#define CKLT(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cublasLt error %d @%d\n", (int)s, __LINE__); exit(1); } } while(0)

// ---- 配置族模板（v12 思想移植 + FP8 BK=128 常量）----
template <int BM_, int BN_, int WM_, int WN_, int THREADS_ = 256, int BPSM_ = 1>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, WM = WM_, WN = WN_;
    static constexpr int THREADS = THREADS_, BLOCKS_PER_SM = BPSM_;
    static constexpr int NUM_WARPS = THREADS_ / 32;
    static constexpr int NUM_M = WM_ / 16;                // warp tile 内 m16 组
    static constexpr int NUM_NG = WN_ / 8;                // warp tile 内 n8 组
    static constexpr int NUM_B_CHUNKS = BN_ / 64;         // B chunk 数（WN 恒 64 锁死）
    static constexpr int WARP_N_COUNT = NUM_B_CHUNKS;
    static constexpr int WARP_M_COUNT = NUM_WARPS / WARP_N_COUNT;
    static_assert(WM_ == BM_ / WARP_M_COUNT, "WM must match warp grid");
    static_assert(WARP_M_COUNT * WARP_N_COUNT == NUM_WARPS, "warp grid");
    static constexpr int BK = 128;                        // e4m3 元素数（内维 128B）
    static constexpr int A_STAGE_BYTES  = BM_ * 128;
    static constexpr int B_CHUNK_BYTES  = 64 * 128;       // 64 n 行 × 128B
    static constexpr int B_STAGE_BYTES  = NUM_B_CHUNKS * B_CHUNK_BYTES;
    static constexpr int TMA_BYTES_PER_STAGE = A_STAGE_BYTES + B_STAGE_BYTES;
    static constexpr int NUM_STAGES = 2;
    static constexpr int SMEM_BYTES = NUM_STAGES * TMA_BYTES_PER_STAGE + 2 * NUM_STAGES * 8;
};

// ---- mbarrier / TMA 原语（与 v13 逐字一致）----
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

__device__ __forceinline__ uint32_t umin_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// ---- kernel（配置模板化：C 决定 tile/warp 网格/smem 几何）----
template <typename C>
__global__ void __launch_bounds__(C::THREADS, C::BLOCKS_PER_SM) gemm_fp8_v2(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_c,
    int M, int N, int K, float alpha
#if defined(NOTMA) || defined(DUMP)
    , const unsigned char* gA_raw, const unsigned char* gB_raw
#endif
) {
    // 配置常量展开（kernel 体与 v1 逐行同构，仅常量来源改为模板）
    constexpr int BM = C::BM, BN = C::BN, WM = C::WM, WN = C::WN;
    constexpr int THREADS = C::THREADS, NUM_WARPS = C::NUM_WARPS;
    constexpr int NUM_M = C::NUM_M, NUM_NG = C::NUM_NG, NUM_B_CHUNKS = C::NUM_B_CHUNKS;
    constexpr int BK = C::BK, NUM_STAGES = C::NUM_STAGES;
    constexpr int A_STAGE_BYTES = C::A_STAGE_BYTES, B_CHUNK_BYTES = C::B_CHUNK_BYTES;
    constexpr int TMA_BYTES_PER_STAGE = C::TMA_BYTES_PER_STAGE, SMEM_BYTES = C::SMEM_BYTES;
#ifdef DUMP
    unsigned char* dump_buf = const_cast<unsigned char*>(gB_raw);
#endif

    // ---- block swizzle（同 v13）----
#ifndef GM
#define GM 8
#endif
    constexpr uint32_t GROUP_M = GM;
    const uint32_t num_m_blocks = (M + BM - 1) / BM;
    const uint32_t num_n_blocks = (N + BN - 1) / BN;
    const uint32_t bid = blockIdx.x;
    const uint32_t group_size = GROUP_M * num_n_blocks;
    const uint32_t group_id = bid / group_size;
    const uint32_t in_group = bid % group_size;
    const uint32_t m_first = umin_u32(group_id * GROUP_M, num_m_blocks);
    const uint32_t m_g = umin_u32(GROUP_M, num_m_blocks - m_first);
    const uint32_t m_block_idx = m_first + in_group % m_g;
    const uint32_t n_block_idx = in_group / m_g;
    if (n_block_idx >= num_n_blocks) return;
    const int block_m = m_block_idx * BM;
    const int block_n = n_block_idx * BN;

    // ---- smem：每 stage 连续 [A 16KB | B 32KB]，接 mbarrier ----
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(smem_raw + NUM_STAGES * TMA_BYTES_PER_STAGE);
    uint64_t* empty_bar = full_bar + NUM_STAGES;

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / NUM_B_CHUNKS;
    const int warp_n = warp % NUM_B_CHUNKS;

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < NUM_STAGES; ++i) {
            mbar_init(&full_bar[i], 1);
            mbar_init(&empty_bar[i], NUM_WARPS);
        }
    }
    fence_mbar_init();
    __syncthreads();

    const int num_k_blocks = (K + BK - 1) / BK;

#ifdef NOTMA
    // 调试路径：绕过 TMA，按 ldsm 期望的布局直写 smem（判别 TMA 配置 vs fragment 逻辑）
    const unsigned char* gA = gA_raw;   // [M][K]
    const unsigned char* gB = gB_raw;   // [N][K]
    auto issue_stage = [&](int s, int kb) {
        uint8_t* a_dst = smem_raw + s * TMA_BYTES_PER_STAGE;
        uint8_t* b_dst = a_dst + A_STAGE_BYTES;
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            const int r = idx / BK, c16 = (idx % BK) / 16, b16 = idx % 16;
            a_dst[r * 128 + (SWZ ? ((c16 ^ (r & 7)) * 16) : (c16 * 16)) + b16] =
                gA[(size_t)(block_m + r) * K + kb * BK + c16 * 16 + b16];
        }
        for (int idx = tid; idx < NUM_B_CHUNKS * 64 * BK; idx += THREADS) {
            const int g = idx / (64 * BK), rem = idx % (64 * BK);
            const int row = rem / BK, c16 = (rem % BK) / 16, b16 = rem % 16;
            b_dst[g * B_CHUNK_BYTES + row * 128 + (SWZ ? ((c16 ^ (row & 7)) * 16) : (c16 * 16)) + b16] =
                gB[(size_t)(block_n + g * 64 + row) * K + kb * BK + c16 * 16 + b16];
        }
        __syncthreads();                     // 直写完成 = full
        if (tid == 0) mbar_arrive(&full_bar[s]);   // arrive（count=1，非 expect_tx 路径）
    };
#else
    // producer：A 1 条（box {128,128}）+ B 4 条（box {128,64}，n 行分 chunk）
    auto issue_stage = [&](int s, int kb) {
        uint8_t* a_dst = smem_raw + s * TMA_BYTES_PER_STAGE;
        uint8_t* b_dst = a_dst + A_STAGE_BYTES;
        mbar_arrive_expect_tx(&full_bar[s], TMA_BYTES_PER_STAGE);
        tma_load_2d(&tma_a, a_dst, &full_bar[s], kb * BK, block_m);
        // 注意坐标序：本 tmap 内维 = K、外维 = N → x = K 坐标、y = N 行
        // （v13 的 B tmap 内维是 N，坐标序相反——翻转 tmap 时必须同步交换调用坐标，
        //  chunk 0 因 x=y=0 恰好掩盖了这个 bug，chunk 1-3 全错）
        #pragma unroll
        for (int g = 0; g < NUM_B_CHUNKS; ++g)
            tma_load_2d(&tma_b, b_dst + g * B_CHUNK_BYTES, &full_bar[s],
                        kb * BK, block_n + g * 64);
    };
#endif

    // 累加器：4 m16 × 8 n8 × 4 f32 = 128
    float c[NUM_M][NUM_NG][4];
    #pragma unroll
    for (int i = 0; i < NUM_M; ++i)
        #pragma unroll
        for (int j = 0; j < NUM_NG; ++j)
            c[i][j][0] = c[i][j][1] = c[i][j][2] = c[i][j][3] = 0.0f;

#ifdef NOTMA
    if (num_k_blocks > 0) issue_stage(0, 0);           // 全线程（内含 __syncthreads）
#else
    if (tid == 0 && num_k_blocks > 0) issue_stage(0, 0);
#endif

    uint32_t phase[NUM_STAGES] = {0};
    uint32_t prod_empty_phase[NUM_STAGES] = {0};

    // ---- 主循环（结构同 v13：零 syncthreads + unroll 2 跨迭代调度）----
    #pragma unroll 2
    for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int s = kb % NUM_STAGES;

#ifdef NOTMA
        if (kb + 1 < num_k_blocks) {                    // 全线程：parity wait 多观察者安全
            int ns = (kb + 1) % NUM_STAGES;
            if (kb >= 1) {
                mbar_wait(&empty_bar[ns], prod_empty_phase[ns]);
                prod_empty_phase[ns] ^= 1;
            }
            issue_stage(ns, kb + 1);
        }
#else
        if (tid == 0 && kb + 1 < num_k_blocks) {
            int ns = (kb + 1) % NUM_STAGES;
            if (kb >= 1) {
                mbar_wait(&empty_bar[ns], prod_empty_phase[ns]);
                prod_empty_phase[ns] ^= 1;
            }
            issue_stage(ns, kb + 1);
        }
#endif

        mbar_wait(&full_bar[s], phase[s]);
        phase[s] ^= 1;

#ifdef DUMP
        if (kb == 0) {
            const uint8_t* a_base_dbg = smem_raw + s * TMA_BYTES_PER_STAGE;
            const uint8_t* b_base_dbg = a_base_dbg + A_STAGE_BYTES;
            for (int i = tid; i < 4 * 128; i += THREADS)
                dump_buf[i] = a_base_dbg[(i / 128) * 128 + i % 128];
            for (int g = 0; g < 4; ++g)
                for (int i = tid; i < 2 * 128; i += THREADS)
                    dump_buf[512 + g * 256 + i] =
                        b_base_dbg[g * B_CHUNK_BYTES + (i / 128) * 128 + i % 128];
            __syncthreads();
        }
#endif
        const uint8_t* a_base = smem_raw + s * TMA_BYTES_PER_STAGE;
        const uint8_t* b_base = a_base + A_STAGE_BYTES;
        const int atile = lane / 8, ar8 = lane % 8;
        const int gl = lane / 8;                       // B 侧：x4 的 4 矩阵角色 0-3

        // ---- 4 个 k32 slice（每 slice：4 A-ldsm.x4 + 4 B-ldsm.x4 + 32 mma）----
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            // A fragment（寻址与 v13 逐字节同构：sub = ks*2 + atile/2 覆盖 chunk 0-7）
            uint32_t a[NUM_M][4];
            #pragma unroll
            for (int mi = 0; mi < NUM_M; ++mi) {
                const int r = warp_m * WM + mi * 16 + ar8 + (atile % 2) * 8;
                const int sub = ks * 2 + atile / 2;
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                    a_base + r * 128 + (SWZ ? ((sub ^ (r & 7)) * 16) : (sub * 16)));
                asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3]) : "r"(addr));
            }
            // B fragment（n-major 非 trans：每 x4 = 2 个 n-group 的 {b0,b1}）
            //   lane 角色映射：gl=0→(ng_lo,k-lo) gl=1→(ng_lo,k-hi) gl=2→(ng_hi,k-lo) gl=3→(ng_hi,k-hi)
            uint32_t b[NUM_NG][2];
            #pragma unroll
            for (int npj = 0; npj < NUM_NG / 2; ++npj) {
                const int ng = npj * 2 + gl / 2;             // 本 lane 覆盖的 n-group
                const int row = ng * 8 + lane % 8;           // chunk 内 n 行
                const int chunk = ks * 2 + (gl % 2);         // k-lo / k-hi 的 16B chunk
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                    b_base + warp_n * B_CHUNK_BYTES + row * 128 + (SWZ ? ((chunk ^ (row & 7)) * 16) : (chunk * 16)));
                uint32_t r0, r1, r2, r3;
                asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
                b[npj * 2][0] = r0;  b[npj * 2][1] = r1;
                b[npj * 2 + 1][0] = r2;  b[npj * 2 + 1][1] = r3;
            }
            // mma：m16n8k32 e4m3（4 m × 8 n = 32 发）
            #pragma unroll
            for (int mi = 0; mi < NUM_M; ++mi)
                #pragma unroll
                for (int ng = 0; ng < NUM_NG; ++ng)
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+f"(c[mi][ng][0]), "+f"(c[mi][ng][1]),
                                   "+f"(c[mi][ng][2]), "+f"(c[mi][ng][3])
                                 : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]),
                                   "r"(b[ng][0]), "r"(b[ng][1]));
        }

        // 释放 stage（SM121：arrive 前显式 fence 排序 LDSM 读）
        asm volatile("fence.acq_rel.cta;\n");
        if (lane == 0) mbar_arrive(&empty_bar[s]);
    }

    // ---- epilogue：v13 同款（4-pass × 32 行 TMA store 流水）+ per-tensor scale ----
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
            for (int ng = 0; ng < NUM_NG; ++ng) {
                const int rr = r0 - 32 * p + crow;
                const int cc = ng * 8 + ccol;
                *reinterpret_cast<float2*>(&cstg[rr * 64 + cc]) =
                    make_float2(c[mi][ng][0] * alpha, c[mi][ng][1] * alpha);
                *reinterpret_cast<float2*>(&cstg[(rr + 8) * 64 + cc]) =
                    make_float2(c[mi][ng][2] * alpha, c[mi][ng][3] * alpha);
            }
        }
        fence_proxy_async();
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int g = 0; g < NUM_B_CHUNKS; ++g)
                tma_store_2d(&tma_c,
                             reinterpret_cast<float*>(smem_raw + (p % NUM_STAGES) * TMA_BYTES_PER_STAGE) + g * (32 * 64),
                             block_n + g * 64, block_m + 32 * p);
            tma_store_commit();
            tma_store_wait<1>();
        }
        __syncthreads();
    }
    if (tid == 0) tma_store_wait<0>();
    __syncthreads();
}

// ---- tensor map（uint8 + SWIZZLE_128B）----
static void create_tmap_A(CUtensorMap* map, const unsigned char* dA, int M, int K) {
    cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)M};
    cuuint64_t gstr[1] = {(cuuint64_t)K};                 // 行步长 K 字节（K%16==0 保证 16B 对齐）
    cuuint32_t box[2] = {128, 128};                       // 内维 128 k 字节 × 外维 128 m 行（恒定）
    cuuint32_t es[2] = {1, 1};
    CK_DRV(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        (void*)dA, gdim, gstr, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, SWZ ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
static void create_tmap_B(CUtensorMap* map, const unsigned char* dB, int N, int K) {
    cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)N};  // [N][K]（写侧已转置）
    cuuint64_t gstr[1] = {(cuuint64_t)K};
    cuuint32_t box[2] = {128, 64};                        // 内维 128 k 字节 × 外维 64 n 行（恒定）
    cuuint32_t es[2] = {1, 1};
    CK_DRV(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        (void*)dB, gdim, gstr, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, SWZ ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
static void create_tmap_C(CUtensorMap* map, const float* dC, int M, int N) {
    cuuint64_t gdim[2] = {(cuuint64_t)N, (cuuint64_t)M};
    cuuint64_t gstr[1] = {(cuuint64_t)N * sizeof(float)};
    cuuint32_t box[2] = {64, 32};
    cuuint32_t es[2] = {1, 1};
    CK_DRV(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
        (void*)dC, gdim, gstr, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// ---- cuBLASLt FP8 参照（阶段 C canonical 布局 + 同 scale）----
// dBq 是写侧转置过的 [N][K]（行主序），dAq 是 [M][K]（行主序）：
//   D_row[m][n] = Σ A[m][k]·B[k][n] = Σ Aq[m][k]·Bq[n][k]
//   ⇔ D_col[N][M] = opT(Bq_col[K×N]) · opN(Aq_col[K×M])
//   其中 Bq_col 用 layout (K,N,K)（dBq 的 col-major 视图），opT 后 = Bq(N×K)；
//         Aq_col 用 layout (K,M,K)（dAq 的 col-major 视图）= Aqᵀ(K×M)。
static void cublaslt_fp8_ref(const unsigned char* dAq, const unsigned char* dBq,
                             float* dRef, int M, int N, int K,
                             float sA, float sB, cublasLtHandle_t lt) {
    cublasLtMatmulDesc_t op;
    CKLT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    cublasLtMatrixLayout_t lB, lA, ld;
    CKLT(cublasLtMatrixLayoutCreate(&lB, CUDA_R_8F_E4M3, K, N, K));   // dBq[N][K] 的 col-major 视图
    CKLT(cublasLtMatrixLayoutCreate(&lA, CUDA_R_8F_E4M3, K, M, K));   // dAq[M][K] 的 col-major 视图
    CKLT(cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, N, M, N));
    float *dSA, *dSB;
    CK(cudaMalloc(&dSA, 4)); CK(cudaMalloc(&dSB, 4));
    CK(cudaMemcpy(dSA, &sA, 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSB, &sB, 4, cudaMemcpyHostToDevice));
    cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm)));
    CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm)));
    CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dSA, sizeof(dSA)));
    CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dSB, sizeof(dSB)));
    cublasLtMatmulPreference_t pref;
    CKLT(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = 64 << 20;
    CKLT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    cublasLtMatmulHeuristicResult_t h[8]; int nr = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, op, lB, lA, ld, ld, pref, 8, h, &nr);
    if (nr == 0) { fprintf(stderr, "cublasLt 参照无 algo (status=%d)\n", (int)st); exit(1); }
    float alpha = 1.f, beta = 0.f;
    bool ok = false;
    for (int i = 0; i < nr; ++i) {
        st = cublasLtMatmul(lt, op, &alpha, dBq, lB, dAq, lA, &beta, dRef, ld, dRef, ld,
                            &h[i].algo, nullptr, 0, nullptr);
        if (st == CUBLAS_STATUS_SUCCESS) { ok = true; break; }
    }
    if (!ok) { fprintf(stderr, "cublasLt 参照 matmul 失败\n"); exit(1); }
    CK(cudaDeviceSynchronize());
    cudaFree(dSA); cudaFree(dSB);
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 4096;
    int N = argc > 2 ? atoi(argv[2]) : 4096;
    int K = argc > 3 ? atoi(argv[3]) : 4096;
    bool do_check = (argc <= 4);
    if (K % 16 != 0 || M % 8 != 0 || N % 8 != 0) {
        fprintf(stderr, "Error: 需 M/N%%8==0 且 K%%16==0（TMA uint8 步长对齐）\n");
        return 1;
    }

    // ---- host 数据 + 动态 per-tensor 量化（阶段 B 语义：amax/448）----
    std::vector<bf16> hA((size_t)M * K), hB((size_t)K * N);
    srand(42);
    auto randf = []() { return (float)(rand() % 2000 - 1000) / 1000.0f; };
    for (auto& v : hA) v = __float2bfloat16(randf());
    for (auto& v : hB) v = __float2bfloat16(randf());

    auto amax_of = [](const std::vector<bf16>& v) {
        float m = 0;
        for (auto x : v) m = std::max(m, fabsf(__bfloat162float(x)));
        return m;
    };
    float sA = amax_of(hA) / 448.0f, sB = amax_of(hB) / 448.0f;
    float invA = 1.0f / (sA + 1e-8f), invB = 1.0f / (sB + 1e-8f);
    float alpha = (1.0f / invA) * (1.0f / invB);      // = sA*sB（去量化因子）

    std::vector<unsigned char> hAq((size_t)M * K), hBq((size_t)N * K);   // B 写侧转置 [N][K]
    for (size_t i = 0; i < hA.size(); ++i)
        hAq[i] = (unsigned char)__nv_cvt_float_to_fp8(
            __bfloat162float(hA[i]) * invA, __NV_SATFINITE, __NV_E4M3);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            hBq[(size_t)n * K + k] = (unsigned char)__nv_cvt_float_to_fp8(
                __bfloat162float(hB[(size_t)k * N + n]) * invB, __NV_SATFINITE, __NV_E4M3);

    unsigned char *dAq, *dBq; float* dC;
    CK(cudaMalloc(&dAq, hAq.size()));
    CK(cudaMalloc(&dBq, hBq.size()));
    CK(cudaMalloc(&dC, sizeof(float) * (size_t)M * N));
    CK(cudaMemcpy(dAq, hAq.data(), hAq.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBq, hBq.data(), hBq.size(), cudaMemcpyHostToDevice));

    unsigned char* dDump = nullptr;
#ifdef DUMP
    CK(cudaMalloc(&dDump, 1024));
#endif
    CUtensorMap tma_a, tma_b, tma_c;
    create_tmap_A(&tma_a, dAq, M, K);
    create_tmap_B(&tma_b, dBq, N, K);
    create_tmap_C(&tma_c, dC, M, N);

    // ---- 配置族 + 双维分派（v2 核心）----
    using BigCfg   = Cfg<128, 256, 64, 64>;   // 非幂大形状最优
    using MidCfg   = Cfg<128, 128, 32, 64>;   // 2 幂大形状：B 足迹减半摊薄 L2 slice 压力（病理 -9.5%→-3.4%）
    using SmallCfg = Cfg<128, 64, 16, 64>;    // 小形状填充（WM 联动：BN=64→1 chunk→8 warp_m→WM=16）
    CK(cudaFuncSetAttribute((gemm_fp8_v2<BigCfg>),   cudaFuncAttributeMaxDynamicSharedMemorySize, BigCfg::SMEM_BYTES));
    CK(cudaFuncSetAttribute((gemm_fp8_v2<MidCfg>),   cudaFuncAttributeMaxDynamicSharedMemorySize, MidCfg::SMEM_BYTES));
    CK(cudaFuncSetAttribute((gemm_fp8_v2<SmallCfg>), cudaFuncAttributeMaxDynamicSharedMemorySize, SmallCfg::SMEM_BYTES));

    // 分派规则（填充优先，其次 2 幂病理规避——仅 K≥8192：病理深度随规模增长，
    //   4096³ 实测病理较浅，MidCfg 的 A 流量 ×1.33 代价反超规避收益，净 -5.2%）：
    const uint32_t big_blocks = ((M + 127) / 128) * ((N + 255) / 256);
    const bool is_pow2_k = (K & (K - 1)) == 0 && K >= 8192;
    const int which = (big_blocks < 48) ? 2 : (is_pow2_k ? 1 : 0);   // 0=Big 1=Mid 2=Small

    const uint32_t bn_used = which == 0 ? 256 : (which == 1 ? 128 : 64);
    const uint32_t num_m_blocks = (M + 127) / 128;
    const uint32_t num_n_blocks = (N + bn_used - 1) / bn_used;
    dim3 grid(num_m_blocks * num_n_blocks);
    auto run1 = [&] {
        switch (which) {
        case 2: gemm_fp8_v2<SmallCfg><<<grid, SmallCfg::THREADS, SmallCfg::SMEM_BYTES>>>(
                    tma_a, tma_b, tma_c, M, N, K, alpha); break;
        case 1: gemm_fp8_v2<MidCfg><<<grid, MidCfg::THREADS, MidCfg::SMEM_BYTES>>>(
                    tma_a, tma_b, tma_c, M, N, K, alpha); break;
        default: gemm_fp8_v2<BigCfg><<<grid, BigCfg::THREADS, BigCfg::SMEM_BYTES>>>(
                    tma_a, tma_b, tma_c, M, N, K, alpha); break;
        }
    };

#ifdef DUMP
    run1(); CK(cudaDeviceSynchronize());
    unsigned char hDump[1024];
    CK(cudaMemcpy(hDump, dDump, 1024, cudaMemcpyDeviceToHost));
    auto dqd = [](unsigned char b) { __nv_fp8_e4m3 v; memcpy(&v, &b, 1); return (float)v; };
    printf("== TMA smem dump（A 前 2 行 + B 4 chunk 各前 2 行，数值 got|want）==\n");
    // A
    for (int r = 0; r < 2; ++r) {
        printf("[A 行%d] ", r);
        for (int j = 0; j < 8; ++j) {
            int phys = j, logk = -1;
            for (int c = 0; c < 8; ++c)
                if (((c ^ (r & 7)) * 16) <= phys && phys < ((c ^ (r & 7)) * 16) + 16)
                    logk = c * 16 + phys % 16;
            printf("(%g|%g) ", dqd(hDump[r * 128 + phys]), dqd(hAq[(size_t)r * 128 + logk]));
        }
        printf("\n");
    }
    // B chunk 0-3
    for (int g = 0; g < 4; ++g)
        for (int r = 0; r < 2; ++r) {
            printf("[B ch%d 行%d] ", g, r);
            for (int j = 0; j < 8; ++j) {
                int phys = j, logk = -1;
                for (int c = 0; c < 8; ++c)
                    if (((c ^ (r & 7)) * 16) <= phys && phys < ((c ^ (r & 7)) * 16) + 16)
                        logk = c * 16 + phys % 16;
                printf("(%g|%g) ", dqd(hDump[512 + g * 256 + r * 128 + phys]),
                       dqd(hBq[(size_t)(g * 64 + r) * 128 + logk]));
            }
            printf("\n");
        }
    return 0;
#endif
    // 时间制热身（协议 v2）：kernel 活跃累计 ≥30ms 即停——不写死次数
    // （写死 3000 次时 8192³ 每轮热身 3.7s，是需求量的 120 倍）
    {
        cudaEvent_t ws, we; cudaEventCreate(&ws); cudaEventCreate(&we);
        cudaEventRecord(ws);
        int cnt = 0;
        do {
            for (int i = 0; i < 32; ++i) run1();
            cnt += 32;
            cudaEventRecord(we); cudaEventSynchronize(we);
            float wms = 0; cudaEventElapsedTime(&wms, ws, we);
            if (wms > 300.0f) break;   // 大形状需更长热身：8192³ 在 30ms 停点时钟未满（教训实测）
        } while (cnt < 1000000);
        CK(cudaDeviceSynchronize());
        cudaEventDestroy(ws); cudaEventDestroy(we);
    }
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    // 短内核加大迭代（协议 v2 修正 2026-11-18）：计时窗 ≥20ms——
    // 1024³ 单发仅 11.7μs，20 迭代的计时窗 234μs 太薄，一次干扰占比过大。
    // 估时用独立事件对，绝不污染正式计时窗口（教训：Record(s) 若先于估时，
    // 估时那一发 kernel 会被计入正式窗口 → ms 虚高）
    int iters = 20;
    {
        cudaEvent_t es, ee; cudaEventCreate(&es); cudaEventCreate(&ee);
        cudaEventRecord(es);
        run1();
        cudaEventRecord(ee); cudaEventSynchronize(ee);
        float one = 0; cudaEventElapsedTime(&one, es, ee);
        if (one > 0.001f && one < 1.0f) {
            int want = (int)(20.0f / one) + 1;
            iters = want > 2000 ? 2000 : (want < 20 ? 20 : want);
        }
        cudaEventDestroy(es); cudaEventDestroy(ee);
    }
    cudaEventRecord(s);                       // 正式计时起点：估时之后
    for (int i = 0; i < iters; ++i) run1();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;

    double flop = 2.0 * M * N * K;
    double tflops = flop / (ms / 1e3) / 1e12;
    const char* cfg_name = which == 0 ? "128x256" : (which == 1 ? "128x128pow2" : "128x64");
    printf("[M=%d N=%d K=%d] fp8_v2(%s): %.3f ms, %.4f TFLOPS "
           "(FP8 峰值 235.79 的 %.2f%% | BF16 v13 同形状的 %.2f 倍)\n",
           M, N, K, cfg_name, ms, tflops, tflops / 235.79 * 100, tflops / 95.04);

    if (do_check) {
        // CPU 参照（小形状直接对拍，dump 错位分布定位布局 bug）
        if ((long)M * N * K <= (long)64 * 1024 * 1024) {
            std::vector<float> hC((size_t)M * N);
            CK(cudaMemcpy(hC.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
            auto dq = [](unsigned char b) { __nv_fp8_e4m3 v; memcpy(&v, &b, 1); return (float)v; };
            double max_abs = 0; int bad = 0;
            int bad_m_lo = 0, bad_m_hi = 0, bad_n_lo = 0, bad_n_hi = 0;
            int bad_ch[8] = {0}, tot_ch[8] = {0};   // 按 n/64 分桶（B chunk 维度）
            for (int m = 0; m < M; ++m)
                for (int n = 0; n < N; ++n) {
                    float ref = 0;
                    for (int k = 0; k < K; ++k)
                        ref += dq(hAq[(size_t)m * K + k]) * dq(hBq[(size_t)n * K + k]);
                    ref *= alpha;
                    double d = fabs((double)hC[(size_t)m * N + n] - ref);
                    tot_ch[n / 64]++;
                    if (d > 1e-2 * std::max(1.0, fabs((double)ref))) {
                        ++bad;
                        if (m < M / 2) ++bad_m_lo; else ++bad_m_hi;
                        if (n < N / 2) ++bad_n_lo; else ++bad_n_hi;
                        bad_ch[n / 64]++;
                        if (bad <= 3)
                            printf("  错位: C[%d][%d] got %.4f want %.4f\n",
                                   m, n, hC[(size_t)m * N + n], ref);
                    }
                    max_abs = std::max(max_abs, d);
                }
            printf("[CPU 对拍] max_abs=%.3e 错位=%d/%d  (m<%d: %d / m≥: %d | n<%d: %d / n≥: %d)\n",
                   max_abs, bad, M * N, M / 2, bad_m_lo, bad_m_hi, N / 2, bad_n_lo, bad_n_hi);
            printf("[B chunk 分桶] ");
            for (int i = 0; i < (N + 63) / 64 && i < 8; ++i) printf("ch%d:%d/%d ", i, bad_ch[i], tot_ch[i]);
            printf("\n");
        }

        cublasLtHandle_t lt; cublasLtCreate(&lt);
        float* dRef;
        CK(cudaMalloc(&dRef, sizeof(float) * (size_t)M * N));
        cublaslt_fp8_ref(dAq, dBq, dRef, M, N, K, 1.0f / invA, 1.0f / invB, lt);

        std::vector<float> hC((size_t)M * N), hR((size_t)M * N);
        CK(cudaMemcpy(hC.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hR.data(), dRef, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        double max_abs = 0, max_rel = 0;
        for (size_t i = 0; i < hC.size(); ++i) {
            double d = fabs((double)hC[i] - hR[i]);
            max_abs = std::max(max_abs, d);
            max_rel = std::max(max_rel, d / std::max(1e-3, fabs((double)hR[i])));
        }
        printf("正确性 vs cuBLASLt FP8(同精度同 scale): max_abs_err=%.3e  max_rel_err=%.3e  %s\n",
               max_abs, max_rel, max_rel < 1e-3 ? "[PASS]" : "[FAIL]");
        cublasLtDestroy(lt); cudaFree(dRef);
    }

    cudaFree(dAq); cudaFree(dBq); cudaFree(dC);
    return 0;
}
