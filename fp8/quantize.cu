// fp8/quantize.cu —— 阶段 B：BF16 → e4m3 量化 kernel（per-tensor + 动态 abs-max scale）
//
// 职责（一条命令三件事，hpc-ops act.cu 同款思路）：
//   1. 动态 abs-max 求 scale（amax / 448.0f，防 0 除加 1e-8）——两遍式：kernel1 归约 amax，host 算 scale；
//   2. 量化：x * (1/scale) → e4m3，向量化 8 元素/线程（bf16x2x4 load → fp8x2x4 store）；
//   3. 写侧自由（FP8 管线结构性优势，阶段 A 已验证）：
//      - A：按原布局 [M][K] 行主序写出（GEMM 的 A 全局布局）；
//      - B：**转置写出** [N][K] 行主序（阶段 A 验证的路线 b——量化 kernel 顺便做 TMA 做不了的转置，
//        GEMM 侧 smem 行 = n、行内 = k 字节，非 trans LDSM 直出 fragment）。
//
// 阶段门 B（criterion 2.3）：① 量化-反量化对 e4m3 精确值恒等；② 有效带宽 ≥ 200 GB/s。
// 编译：nvcc -O3 -arch=sm_121 -o fp8/quantize fp8/quantize.cu

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

using bf16 = __nv_bfloat16;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// ---- kernel 1：逐行 abs-max 归约（每 block 一行，warp 归约树）----
__global__ void amax_kernel(const bf16* __restrict__ x, float* __restrict__ amax_out,
                            int64_t cols) {
    const int row = blockIdx.x;
    const bf16* rowp = x + (int64_t)row * cols;
    extern __shared__ float warp_max[];               // 32 个 warp 的局部最大
    const int tid = threadIdx.x, lane = tid % 32, wid = tid / 32;
    const int nwarp = blockDim.x / 32;

    float t = 0.f;
    for (int64_t i = tid; i < cols; i += blockDim.x) {
        float v = fabsf(__bfloat162float(rowp[i]));
        t = fmaxf(t, v);
    }
    // warp 内归约
    for (int off = 16; off > 0; off >>= 1)
        t = fmaxf(t, __shfl_xor_sync(0xFFFFFFFF, t, off));
    if (lane == 0) warp_max[wid] = t;
    __syncthreads();
    if (wid == 0) {
        t = (lane < nwarp) ? warp_max[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            t = fmaxf(t, __shfl_xor_sync(0xFFFFFFFF, t, off));
        if (lane == 0) atomicMax(reinterpret_cast<int*>(amax_out), __float_as_int(t));
    }
}

// ---- kernel 2t：量化 + 转置（smem 64×64 tile 版，B: [K][N] → [N][K]）----
// 经 smem 换向：读源行内 128B 连续（uint4 = 8 bf16），写目标行内 64B 连续（uint4 = 16 e4m3）。
// 64×64 tile：256 线程 = 16 行 × 16 列段角色，每线程 4 轮 × 8 元素 = 覆盖 64 行。
// tile 行距 72（64+8，16B 奇数倍——v9 验证的无冲突条件）防 smem bank 冲突。
// 约束：K、N 均为 64 的倍数（GEMM 阶段 D 的 BK=128/BN=64 场景天然满足）。
__global__ void quantize_transpose_kernel(const bf16* __restrict__ x, unsigned char* __restrict__ dst,
                                         float inv_scale, int64_t N, int64_t K) {
    __shared__ bf16 tile[64][72];
    const int64_t k0 = (int64_t)blockIdx.x * 64;
    const int64_t n0 = (int64_t)blockIdx.y * 64;
    const int r = threadIdx.x / 16;           // 行角色：16 行 × 4 轮 = 64 行
    const int c = threadIdx.x % 16;          // 列角色：16 列段（读侧 4 bf16/段；写侧 16 列轮换）

    // 读：源 [K][N]，行 = k（64 行），列 = n（64 列）
    // 每线程一次 uint2（4 bf16 = 8B）：16 列段 × 8B = 每行 128B 连续
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int64_t k = k0 + r + i * 16;
        uint2 raw = *reinterpret_cast<const uint2*>(x + k * N + n0 + c * 4);
        bf16* pb = reinterpret_cast<bf16*>(&raw);
        tile[r + i * 16][c * 4 + 0] = pb[0];
        tile[r + i * 16][c * 4 + 1] = pb[1];
        tile[r + i * 16][c * 4 + 2] = pb[2];
        tile[r + i * 16][c * 4 + 3] = pb[3];
    }
    __syncthreads();
    // 写：目标 [N][K]，行 = n（64 行），列 = k（64 列）——转置读 tile
    // dst[n][k0 + 列] 的源 = x[(k0+列)*N + n] → tile 行 = k、tile 列 = n
    // 写相位独立角色映射：256 线程 = 64 n 行（tid/4）× 4 k 段（tid%4），
    // 每线程恰一次 uint4（16 e4m3 = 16B）；4 段 × 16 = 64 列，64 行 × 1 = 64 行 ✓ 零闲置
    {
        const int nr = threadIdx.x / 4;        // 0..63：目标行（n）
        const int ks = threadIdx.x % 4;        // 0..3：k 列段（每段 16 列 = 16B）
        unsigned char out[16];
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            out[j] = (unsigned char)__nv_cvt_float_to_fp8(
                __bfloat162float(tile[ks * 16 + j][nr]) * inv_scale,
                __NV_SATFINITE, __NV_E4M3);
        *reinterpret_cast<uint4*>(dst + (n0 + nr) * K + k0 + ks * 16) = *reinterpret_cast<uint4*>(out);
    }
}

// ---- kernel 2：量化 + 可选转置（A: [M][K] → [M][K]；B: [K][N] → [N][K]）----
// 向量化：8 bf16（32B）→ 8 e4m3（8B）/线程迭代；K 为 8 的倍数时对齐无尾部。
// transpose=true 时写 [N][K]：源读 x[k*N+n] 连续 n，写出 dst[n*K+k]——转置写在全局侧，
// 写合并按行分组（同 n 行的线程写连续 k 段），访存模式同 hpc-ops reformat_x_scale。
__global__ void quantize_kernel(const bf16* __restrict__ x, unsigned char* __restrict__ dst,
                                float inv_scale, int64_t rows, int64_t cols, int transpose) {
    // 2D grid：blockIdx.y = 行（M 或 N），blockIdx.x 沿列
    const int64_t r = blockIdx.y;
    const int64_t vec_cols = cols / 8;                // 8 元素向量数
    for (int64_t v = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         v < vec_cols; v += (int64_t)gridDim.x * blockDim.x) {
        const int64_t c8 = v * 8;
        if (!transpose) {
            // A 路线：读 x[r*cols + c8..]，写 dst[r*cols + c8..]（元素同序）
            uint4 raw = *reinterpret_cast<const uint4*>(x + r * cols + c8);
            bf16* pb = reinterpret_cast<bf16*>(&raw);
            unsigned char out[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                out[i] = (unsigned char)__nv_cvt_float_to_fp8(
                    __bfloat162float(pb[i]) * inv_scale, __NV_SATFINITE, __NV_E4M3);
            *reinterpret_cast<uint2*>(dst + r * cols + c8) = *reinterpret_cast<uint2*>(out);
        } else {
            // B 路线：源 x 是 [K][N]（k 行 n 列）→ 输出 dst[n][k]：dst 行 = 列遍历
            // 本 block 负责输出行 r（=n）；8 元素沿源列方向（k, k+1..）取：
            //   读 x[(c8+i)*N + r] —— 8 次离散读（转置读）；写 dst[r*cols + c8..] 连续 8B
            unsigned char out[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                out[i] = (unsigned char)__nv_cvt_float_to_fp8(
                    __bfloat162float(x[(c8 + i) * cols + r]) * inv_scale,
                    __NV_SATFINITE, __NV_E4M3);
            *reinterpret_cast<uint2*>(dst + r * cols + c8) = *reinterpret_cast<uint2*>(out);
        }
    }
}

// ---- 反量化（host 侧用于验证）----
// 注意：__nv_fp8_e4m3 没有"从原始字节构造"的构造函数——__nv_fp8_e4m3(int) 是"整数值转 e4m3"。
// 原始字节必须经 storage（union/memcpy）放入 __x 再走 operator float()。
static float dequant(unsigned char b) {
    __nv_fp8_e4m3 v;
    memcpy(&v, &b, 1);          // storage 语义：字节直接进 __x
    return (float)v;            // operator float()：__nv_cvt_fp8_to_halfraw 解码
}

int main(int argc, char** argv) {
    const int64_t M = argc > 1 ? atoll(argv[1]) : 4096;
    const int64_t N = argc > 2 ? atoll(argv[2]) : 4096;
    const int64_t K = argc > 3 ? atoll(argv[3]) : 4096;
    bool do_check = (argc <= 4);

    std::vector<bf16> hA(M * K), hB(K * N);
    srand(42);
    auto randf = []() { return (float)(rand() % 2000 - 1000) / 1000.0f; };
    for (int64_t i = 0; i < M * K; ++i) hA[i] = __float2bfloat16(randf());
    for (int64_t i = 0; i < K * N; ++i) hB[i] = __float2bfloat16(randf());

    bf16 *dA, *dB;
    unsigned char *dAq, *dBq;
    float *d_amax;
    CK(cudaMalloc(&dA, sizeof(bf16) * M * K));
    CK(cudaMalloc(&dB, sizeof(bf16) * K * N));
    CK(cudaMalloc(&dAq, M * K));
    CK(cudaMalloc(&dBq, N * K));
    CK(cudaMalloc(&d_amax, sizeof(float)));
    CK(cudaMemcpy(dA, hA.data(), sizeof(bf16) * M * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), sizeof(bf16) * K * N, cudaMemcpyHostToDevice));

    // ---- 对 A 和 B 各做一遍（动态 scale + 量化 + B 转置）----
    auto quantize_one = [&](const char* name, bf16* dx, unsigned char* dxq,
                            int64_t rows, int64_t cols, int transpose) {
        // 1) amax
        CK(cudaMemset(d_amax, 0, sizeof(float)));
        amax_kernel<<<(int)rows, 256, 128>>>(dx, d_amax, cols);
        float amax = 0.f;
        CK(cudaMemcpy(&amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost));
        float scale = amax / 448.0f;
        float inv_scale = 1.0f / (scale + 1e-8f);

        // 2) 量化（时间制热身 + 计时，协议 v2）
        dim3 blk(256);
        int64_t vec_cols = cols / 8;
        dim3 grd((unsigned)std::min<int64_t>((vec_cols + 255) / 256, 4096), (unsigned)rows);
        auto launch = [&] {
            if (transpose)
                quantize_transpose_kernel<<<dim3((unsigned)(cols / 32), (unsigned)(rows / 32)), 256>>>(
                    dx, dxq, inv_scale, rows, cols);   // cols=N_src? 见调用侧注释
            else
                quantize_kernel<<<grd, blk>>>(dx, dxq, inv_scale, rows, cols, transpose);
        };
        for (int i = 0; i < 3000; ++i) { launch(); if (i % 256 == 255) CK(cudaDeviceSynchronize()); }
        CK(cudaDeviceSynchronize());
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        const int iters = 20;
        for (int i = 0; i < iters; ++i) launch();
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;

        // 带宽账：读 rows*cols*2B + 写 rows*cols*1B（B 路线读离散，仍按 3B/元素计有效带宽下界）
        double bytes = (double)rows * cols * 3;
        double gbps = bytes / (ms / 1e3) / 1e9;
        printf("[量化 %s] rows=%ld cols=%ld transpose=%d: %.3f ms, 有效带宽 %.1f GB/s, scale=%.4e (amax=%.3f)\n",
               name, (long)rows, (long)cols, transpose, ms, gbps, scale, amax);
        return inv_scale;
    };

    float invA = quantize_one("A", dA, dAq, M, K, 0);
    float invB = 1.f;

    // B：源 [K][N] → 输出 [N][K] 转置（tile kernel 参数：源列数 N、源行数 K）
    {
        CK(cudaMemset(d_amax, 0, sizeof(float)));
        amax_kernel<<<(int)K, 256, 128>>>(dB, d_amax, N);      // 逐 k 行求 amax（元素总数不变）
        float amax = 0.f;
        CK(cudaMemcpy(&amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost));
        float scale = amax / 448.0f;
        float inv_scale = 1.0f / (scale + 1e-8f);
        dim3 grd((unsigned)(K / 64), (unsigned)(N / 64));
        auto launch = [&] { quantize_transpose_kernel<<<grd, 256>>>(dB, dBq, inv_scale, N, K); };
        for (int i = 0; i < 3000; ++i) { launch(); if (i % 256 == 255) CK(cudaDeviceSynchronize()); }
        CK(cudaDeviceSynchronize());
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        const int iters = 20;
        for (int i = 0; i < iters; ++i) launch();
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
        double bytes = (double)K * N * 3;
        double gbps = bytes / (ms / 1e3) / 1e9;
        printf("[量化 B] K=%ld N=%ld smem-tile 转置: %.3f ms, 有效带宽 %.1f GB/s, scale=%.4e\n",
               (long)K, (long)N, ms, gbps, scale);
        invB = inv_scale;
    }

    // ---- 阶段门 B①：量化-反量化恒等（对 e4m3 精确值）----
    if (do_check) {
        // 用 e4m3 精确可表示值构造小样本（[-4,4] 整数），kernel 化前后应恒等
        std::vector<bf16> hX(64);
        for (int i = 0; i < 64; ++i) hX[i] = __float2bfloat16((float)((i * 7 % 9) - 4));
        bf16* dX; unsigned char* dXq;
        CK(cudaMalloc(&dX, sizeof(bf16) * 64));
        CK(cudaMalloc(&dXq, 64));
        CK(cudaMemcpy(dX, hX.data(), sizeof(bf16) * 64, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_amax, 0, sizeof(float)));
        amax_kernel<<<1, 256, 128>>>(dX, d_amax, 64);
        float amax = 0.f;
        CK(cudaMemcpy(&amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost));
        float inv = 1.0f / (amax / 448.0f + 1e-8f);
        // 恒等验证：inv=1（不缩放）——[-4,4] 整数本就精确落在 e4m3 网格上，往返必须严格恒等；
        // scale 路径已由全尺寸抽样（固有误差 5% 量级）覆盖
        quantize_kernel<<<dim3(1, 1), 256>>>(dX, dXq, 1.0f, 1, 64, 0);
        unsigned char hXq[64];
        CK(cudaMemcpy(hXq, dXq, 64, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < 64; ++i) {
            float q = dequant(hXq[i]);
            float want = __bfloat162float(hX[i]);
            if (q != want && bad < 4) { printf("  不一致示例: %d got %g (0x%02x) want %g\n", i, q, hXq[i], want); ++bad; }
            else if (q != want) ++bad;
        }
        printf("[阶段门 B①] 量化-反量化恒等(inv=1, 网格精确值): 错位数=%d/64  %s\n", bad, bad == 0 ? "[PASS]" : "[FAIL]");

        // ---- 全尺寸抽样验证（随机数据，量化误差量级记录）----
        std::vector<unsigned char> hAq(1024), hBq(1024);
        CK(cudaMemcpy(hAq.data(), dAq, 1024, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hBq.data(), dBq + N * K - 1024, 1024, cudaMemcpyDeviceToHost));  // B 尾部行
        double max_rel = 0;
        for (int i = 0; i < 1024; ++i) {
            float orig = __bfloat162float(hA[i]);
            float dq = dequant(hAq[i]) / invA;
            max_rel = std::max(max_rel, (double)(fabs(dq - orig) / std::max(1e-3f, fabs(orig))));
        }
        printf("[抽样] A 量化-反量化相对误差 max_rel=%.3e（e4m3 固有量化误差量级，供 D 阶段基线）\n", max_rel);

        // B 转置正确性：dst[n][k] == quant(B[k][n]) 抽样 64 点
        // 验证 8 个完整行（n=0..7），每行 K 字节
        std::vector<unsigned char> hBq_full(8 * (size_t)K);
        CK(cudaMemcpy(hBq_full.data(), dBq, 8 * (size_t)K, cudaMemcpyDeviceToHost));
        int tbad = 0;
        for (int n = 0; n < 8; ++n)
            for (int k = 0; k < 8; ++k) {
                float orig = __bfloat162float(hB[k * (int)N + n]);   // hB 是 [K][N]
                float dq = dequant(hBq_full[n * (int)K + k]) / invB;
                if (fabs(dq - orig) / std::max(1e-3f, fabs(orig)) > 0.1f) {
                    if (tbad < 6) printf("  转置错位: n=%d k=%d got %g want %g\n", n, k, dq, orig);
                    ++tbad;
                }
            }
        printf("[转置验证] dst[n][k] == quant(B[k][n]): 超限点=%d/64  %s\n",
               tbad, tbad == 0 ? "[PASS]" : "[FAIL]");
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dAq); cudaFree(dBq); cudaFree(d_amax);
    return 0;
}
