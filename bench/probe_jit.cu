// bench/probe_jit.cu —— 探针 ⑦ 第 1 步：NVRTC 形状特化收益测量（autogemm 路线 B）
//
// 问题：小形状缺口 = 填充（pow2 tile 天花板）× 短 K 固定成本。1024³ 去同步消融显示
//       计算核已超 cuBLAS 26%，缺口全在 per-K-block 串行暴露。
// 假设：形状常量烘焙（NVRTC JIT，nvjet 同款能力的第 1 步）→ 编译器折叠边界检查、
//       对主循环做首尾 peeling、消死分支 → 固定成本项收窄。
// 对照：同会话交替测 ./gemm_v12（运行时参数版，代码逐行同源）。
// 用法：./bench/probe_jit M N K [skip]     编译：见文件头注释
//
// 编译：nvcc -O3 -arch=sm_121 -o bench/probe_jit bench/probe_jit.cu -lcuda -lcublas -lnvrtc

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nvrtc.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>

using bf16 = __nv_bfloat16;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define CK_DRV(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { \
    const char* s; cuGetErrorString(e, &s); fprintf(stderr, "CU error %s @%d\n", s, __LINE__); exit(1); } } while(0)
#define CK_NVRTC(x) do { nvrtcResult r = (x); if (r != NVRTC_SUCCESS) { \
    fprintf(stderr, "NVRTC error %s @%d\n", nvrtcGetErrorString(r), __LINE__); exit(1); } } while(0)

// ---- tensor map 创建（与 gemm_v12.cu 逐字一致）----
static void create_tensor_map_C(CUtensorMap* map, const float* dC, int M, int N) {
    cuuint64_t gdim[2] = {(cuuint64_t)N, (cuuint64_t)M};
    cuuint64_t gstr[1] = {(cuuint64_t)N * sizeof(float)};
    cuuint32_t box[2] = {64, 32};
    cuuint32_t es[2] = {1, 1};
    CK_DRV(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
        (void*)dC, gdim, gstr, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
static void create_tensor_map_A(CUtensorMap* map, const bf16* dA, int M, int K, int box_m) {
    cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)M};
    cuuint64_t gstr[1] = {(cuuint64_t)K * sizeof(bf16)};
    cuuint32_t box[2] = {64, (cuuint32_t)box_m};
    cuuint32_t es[2] = {1, 1};
    CK_DRV(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        (void*)dA, gdim, gstr, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
static void create_tensor_map_B(CUtensorMap* map, const bf16* dB, int K, int N) {
    cuuint64_t gdim[2] = {(cuuint64_t)N, (cuuint64_t)K};
    cuuint64_t gstr[1] = {(cuuint64_t)N * sizeof(bf16)};
    cuuint32_t box[2] = {64, 64};
    cuuint32_t es[2] = {1, 1};
    CK_DRV(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        (void*)dB, gdim, gstr, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 512;
    int N = argc > 2 ? atoi(argv[2]) : 512;
    int K = argc > 3 ? atoi(argv[3]) : 512;
    bool do_check = (argc <= 4);
    if (M % 8 != 0 || N % 8 != 0 || K % 8 != 0) {
        fprintf(stderr, "Error: M/N/K must be multiples of 8\n");
        return 1;
    }

    // ---- 读取 NVRTC kernel 源（repo 根目录运行：./bench/probe_jit）----
    FILE* f = fopen("bench/jit_kernel.cu", "rb");
    std::string src;
    if (!f) f = fopen("jit_kernel.cu", "rb");
    if (!f) { fprintf(stderr, "找不到 bench/jit_kernel.cu（请在 repo 根目录运行）\n"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    src.resize(sz);
    if (fread(&src[0], 1, sz, f) != (size_t)sz) { fprintf(stderr, "读源失败\n"); return 1; }
    fclose(f);

    // ---- 分派（与 v12 同规则）----
    bool use_small = ((M + 127) / 128) * ((N + 255) / 256) < 48;
    const uint32_t bn = use_small ? 64 : 256;
    // SMEM 公式与 kernel 内一致（BigCfg: 98336B / SmallCfg: 49184B）
    const int smem_bytes = 2 * (128 * 128 + (use_small ? 1 : 4) * 64 * 128) + 2 * 2 * 8;

    // ---- NVRTC 编译（计时；这就是 autogemm 的「每次形状特化成本」）----
    char dM[32], dN[32], dK[32], dS[32];
    snprintf(dM, 32, "-DM_GLOB=%d", M);
    snprintf(dN, 32, "-DN_GLOB=%d", N);
    snprintf(dK, 32, "-DK_GLOB=%d", K);
    snprintf(dS, 32, "-DUSE_SMALL=%d", use_small ? 1 : 0);
    const char* opts[] = {"--gpu-architecture=sm_121", "--std=c++17", "-default-device",
                          "-I/usr/local/cuda/include", "-DJIT_NVRTC",
                          dM, dN, dK, dS};
    auto t0 = std::chrono::steady_clock::now();
    nvrtcProgram prog;
    CK_NVRTC(nvrtcCreateProgram(&prog, src.c_str(), "jit_kernel.cu", 0, nullptr, nullptr));
    nvrtcResult cr = nvrtcCompileProgram(prog, 9, opts);
    if (cr != NVRTC_SUCCESS) {
        size_t logsz; nvrtcGetProgramLogSize(prog, &logsz);
        std::vector<char> log(std::max(logsz, (size_t)1), 0);
        nvrtcGetProgramLog(prog, log.data());
        fprintf(stderr, "NVRTC 编译失败:\n%s\n", log.data());
        return 1;
    }
    size_t cubinSz; CK_NVRTC(nvrtcGetCUBINSize(prog, &cubinSz));
    std::vector<char> cubin(cubinSz);
    CK_NVRTC(nvrtcGetCUBIN(prog, cubin.data()));
    CK_NVRTC(nvrtcDestroyProgram(&prog));
    auto t1 = std::chrono::steady_clock::now();
    double compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    CK_DRV(cuInit(0));

    // ---- 数据准备（与 gemm_v12 一致）----
    std::vector<bf16> hA(M * K), hB(K * N);
    std::vector<float> hC(M * N);
    srand(42);
    auto randf = []() { return (float)(rand() % 2000 - 1000) / 1000.0f; };
    for (int i = 0; i < M * K; ++i) hA[i] = __float2bfloat16_rn(randf());
    for (int i = 0; i < K * N; ++i) hB[i] = __float2bfloat16_rn(randf());
    bf16 *dA, *dB; float *dC;
    CK(cudaMalloc(&dA, sizeof(bf16) * M * K));
    CK(cudaMalloc(&dB, sizeof(bf16) * K * N));
    CK(cudaMalloc(&dC, sizeof(float) * M * N));
    CK(cudaMemcpy(dA, hA.data(), sizeof(bf16) * M * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), sizeof(bf16) * K * N, cudaMemcpyHostToDevice));

    // ---- tensor map：host 创建 → 拷到 device 全局内存（JIT kernel 按指针取）----
    CUtensorMap tmap_a, tmap_b, tmap_c;
    create_tensor_map_A(&tmap_a, dA, M, K, 128);
    create_tensor_map_B(&tmap_b, dB, K, N);
    create_tensor_map_C(&tmap_c, dC, M, N);
    CUtensorMap *d_ta, *d_tb, *d_tc;
    CK(cudaMalloc(&d_ta, sizeof(CUtensorMap)));
    CK(cudaMalloc(&d_tb, sizeof(CUtensorMap)));
    CK(cudaMalloc(&d_tc, sizeof(CUtensorMap)));
    CK(cudaMemcpy(d_ta, &tmap_a, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tb, &tmap_b, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tc, &tmap_c, sizeof(CUtensorMap), cudaMemcpyHostToDevice));

    // ---- 加载 CUBIN + 设 smem 上限 ----
    CUmodule mod; CK_DRV(cuModuleLoadData(&mod, cubin.data()));
    CUfunction kern; CK_DRV(cuModuleGetFunction(&kern, mod, "gemm_jit"));
    CK_DRV(cuFuncSetAttribute(kern, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes));

    const uint32_t num_m_blocks = (M + 127) / 128;
    const uint32_t num_n_blocks = (N + bn - 1) / bn;
    void* params[] = { &d_ta, &d_tb, &d_tc };
    auto run1 = [&] {
        CK_DRV(cuLaunchKernel(kern, num_m_blocks * num_n_blocks, 1, 1,
                              256, 1, 1, smem_bytes, 0, params, nullptr));
    };

    for (int i = 0; i < 3; ++i) run1();
    CK(cudaDeviceSynchronize());

    const int iters = 10;
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) run1();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;

    double flop = 2.0 * M * N * K;
    double tflops = flop / (ms / 1e3) / 1e12;
    printf("[M=%d N=%d K=%d] jit(%s): %.3f ms, %.4f TFLOPS (峰值 118.37 的 %.2f%%)  [NVRTC 编译 %.0f ms]\n",
           M, N, K, use_small ? "128x64 特化" : "128x256 特化", ms, tflops, tflops / 118.37 * 100, compile_ms);

    if (do_check) {
        CK(cudaMemcpy(hC.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        float* dC_cublas;
        CK(cudaMalloc(&dC_cublas, sizeof(float) * M * N));
        cublasHandle_t h; cublasCreate(&h);
        float ca = 1.0f, cb = 0.0f;
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                     &ca, dB, CUDA_R_16BF, N, dA, CUDA_R_16BF, K,
                     &cb, dC_cublas, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        std::vector<float> hC_cublas(M * N);
        CK(cudaMemcpy(hC_cublas.data(), dC_cublas, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
        cublasDestroy(h); cudaFree(dC_cublas);
        double max_abs = 0, max_rel = 0, l2 = 0, l2_ref = 0;
        for (int i = 0; i < M * N; ++i) {
            double d = (double)hC[i] - hC_cublas[i];
            max_abs = std::max(max_abs, std::fabs(d));
            max_rel = std::max(max_rel, std::fabs(d) / std::max(1.0, std::fabs((double)hC_cublas[i])));
            l2 += d * d; l2_ref += (double)hC_cublas[i] * hC_cublas[i];
        }
        printf("正确性 vs cuBLAS: max_abs_err=%.3e  max_rel_err=%.3e  rel_L2=%.3e  %s\n",
               max_abs, max_rel, std::sqrt(l2) / std::sqrt(l2_ref), max_rel < 1e-3 ? "[PASS]" : "[FAIL]");
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(d_ta); cudaFree(d_tb); cudaFree(d_tc);
    return 0;
}
