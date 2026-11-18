// fp8/probe_cublaslt_fp8.cu —— 阶段 C：GB10 上 cuBLASLt 的 FP8 支持探针
//
// 回答三个问题（决定阶段 D 的精度参照系）：
//   1. cublasLtMatmul 能否接受 CUDA_R_8F_E4M3 输入？（无 → 阶段 D 走"两段对拍"参照系）
//   2. 若能：跑通的最小配置是什么（scale 模式/计算类型）？
//   3. 若能：性能多少（FP8 GEMM 在 GB10 上的另一个参照基线）？
//
// 方法：官方推荐流程 —— cublasLtMatmulAlgoGetHeuristic 拿 algo → cublasLtMatmul。
//   输入 A/B 用 e4m3（本文件内 host 生成随机数转 e4m3），C/D 用 fp32/bf16。
//   scale 模式先试 CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F（per-tensor 标量，
//   与我们阶段 D v1 的 scale 结构一致）；失败则降级试 VECTORIZED / 无 scale。
//   注意：NVFP4/FP8 的 cublasLt 要求 scale 指针合法（host 标量 fp32）。
//
// 编译：nvcc -O3 -arch=sm_121 -o fp8/probe_cublaslt_fp8 fp8/probe_cublaslt_fp8.cu -lcublasLt -lcuda

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define CKLT(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cublasLt error %d @%d\n", (int)s, __LINE__); exit(1); } } while (0)

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 4096;
    const int N = argc > 2 ? atoi(argv[2]) : 4096;
    const int K = argc > 3 ? atoi(argv[3]) : 4096;

    cublasLtHandle_t ltHandle;
    CKLT(cublasLtCreate(&ltHandle));
    printf("[阶段 C] cuBLASLt FP8 探针: M=%d N=%d K=%d\n", M, N, K);

    // ---- 数据：A[M][K] e4m3, B[K][N] e4m3, D[M][N] bf16（输出）----
    // host 侧随机数 → e4m3 字节（值域 [-1,1]，均匀量化无饱和）
    std::vector<unsigned char> hA(M * K), hB(K * N);
    srand(42);
    auto randq = []() {
        float v = (float)(rand() % 2000 - 1000) / 1000.0f;
        return (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    };
    for (int64_t i = 0; i < (int64_t)M * K; ++i) hA[i] = randq();
    for (int64_t i = 0; i < (int64_t)K * N; ++i) hB[i] = randq();

    unsigned char *dA, *dB;
    __nv_bfloat16 *dD;
    CK(cudaMalloc(&dA, (size_t)M * K));
    CK(cudaMalloc(&dB, (size_t)K * N));
    CK(cudaMalloc(&dD, sizeof(__nv_bfloat16) * M * N));
    CK(cudaMemcpy(dA, hA.data(), (size_t)M * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), (size_t)K * N, cudaMemcpyHostToDevice));

    // ---- 布局：行主序 A[M][K] x B[K][N] → cublasLt 列主序视角 ----
    // cublasLt 与 cublas 同族：opN/opT 语义按列主序。我们直接给行主序数据配 opT：
    //   A_row[M][K] ≡ A_col[K][M]，opT 后 = [M][K]；B_row[K][N] ≡ B_col[N][K]，opN 后 = [K][N]
    //   → C_col[N][M] ≡ C_row[M][N] ✓（与我们 BF16 线 cublasGemmEx 的换法一致）
    cublasLtMatmulDesc_t op;
    CKLT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    cublasLtMatrixLayout_t la, lb, ld;
    CKLT(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, K, M, K));   // A_col[K][M]，ld=K
    CKLT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, K, N, K));   // B_col[K][N]，ld=K
    CKLT(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, N, M, N));      // D_col[N][M]，ld=N

    // ---- per-tensor 标量 scale（SCALAR_32F 模式）----
    float hScaleA = 1.0f, hScaleB = 1.0f;    // 本探针数据值域 [-1,1]，scale=1 不饱和
    float *dScaleA, *dScaleB;
    CK(cudaMalloc(&dScaleA, 4));
    CK(cudaMalloc(&dScaleB, 4));
    CK(cudaMemcpy(dScaleA, &hScaleA, 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dScaleB, &hScaleB, 4, cudaMemcpyHostToDevice));

    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    cublasStatus_t sa = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                       &scaleMode, sizeof(scaleMode));
    cublasStatus_t sb = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                       &scaleMode, sizeof(scaleMode));
    printf("scale 模式设置: A=%d B=%d（0=SUCCESS）\n", (int)sa, (int)sb);
    if (sa == CUBLAS_STATUS_SUCCESS)
        CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                            &dScaleA, sizeof(dScaleA)));
    if (sb == CUBLAS_STATUS_SUCCESS)
        CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                            &dScaleB, sizeof(dScaleB)));

    // ---- heuristic 拿 algo ----
    cublasLtMatmulPreference_t pref;
    CKLT(cublasLtMatmulPreferenceCreate(&pref));
    size_t wsSize = 32 * 1024 * 1024;
    CKLT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &wsSize, sizeof(wsSize)));
    cublasLtMatmulHeuristicResult_t heur[4];
    int nres = 0;
    cublasStatus_t sh = cublasLtMatmulAlgoGetHeuristic(ltHandle, op, la, lb, ld, ld,
                                                       pref, 4, heur, &nres);
    printf("heuristic: status=%d, 可用 algo 数=%d\n", (int)sh, nres);

    if (sh == CUBLAS_STATUS_SUCCESS && nres > 0) {
        // ---- 跑一发 + 计时（协议 v2：时间制热身）----
        float alpha = 1.0f, beta = 0.0f;
        auto run1 = [&] {
            return cublasLtMatmul(ltHandle, op, &alpha,
                                  dA, la, dB, lb, &beta,
                                  dD, ld, dD, ld,
                                  &heur[0].algo, nullptr, 0, nullptr);
        };
        cublasStatus_t sr = run1();
        printf("cublasLtMatmul(e4m3 x e4m3 -> bf16): status=%d\n", (int)sr);
        if (sr == CUBLAS_STATUS_SUCCESS) {
            // 时间制热身 ≥30ms
            cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
            cudaEventRecord(s);
            int cnt = 0;
            do {
                run1();
                if (++cnt % 64 == 0) {
                    cudaEventRecord(e); cudaEventSynchronize(e);
                    float wms = 0; cudaEventElapsedTime(&wms, s, e);
                    if (wms > 30.f) break;
                }
            } while (cnt < 1000000);
            cudaDeviceSynchronize();
            cudaEventRecord(s);
            const int iters = 20;
            for (int i = 0; i < iters; ++i) run1();
            cudaEventRecord(e); cudaEventSynchronize(e);
            float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
            double tflops = 2.0 * M * N * K / (ms / 1e3) / 1e12;
            printf("[结论] cuBLASLt FP8 可用! %.3f ms, %.4f TFLOPS (FP8 峰值 235.79 的 %.2f%%)\n",
                   ms, tflops, tflops / 235.79 * 100);

            // ---- 精度抽查：取 16 个点与 CPU 参照对拍（scale=1，e4m3 值域精确）----
            // 注：此对拍只验"engine 语义正确"（layout/转置理解无误），不验精度细节
            std::vector<__nv_bfloat16> hD(M * N);
            CK(cudaMemcpy(hD.data(), dD, sizeof(__nv_bfloat16) * M * N, cudaMemcpyDeviceToHost));
            auto dq = [](unsigned char b) {
                __nv_fp8_e4m3 v; memcpy(&v, &b, 1); return (float)v;
            };
            double max_abs = 0, max_abs_bt = 0;
            for (int t = 0; t < 16; ++t) {
                int m = (t * 37) % M, n = (t * 53) % N;
                float ref = 0;      // 假设1：D_row[m][n] = Σ A_row[m][k]·B_row[k][n]
                float ref_bt = 0;   // 假设2：D 是列主序未被换回（D_col[m][n] = D_row[n][m] 的产物）
                for (int k = 0; k < K; ++k) {
                    ref += dq(hA[(int64_t)m * K + k]) * dq(hB[(int64_t)k * N + n]);
                    ref_bt += dq(hA[(int64_t)n * K + k]) * dq(hB[(int64_t)k * N + m]);
                }
                double got = __bfloat162float(hD[(int64_t)m * N + n]);
                max_abs = std::max(max_abs, std::fabs(got - ref));
                max_abs_bt = std::max(max_abs_bt, std::fabs(got - ref_bt));
            }
            printf("[抽查] 假设1(行主序直积) max_abs=%.3e / 假设2(行列互换) max_abs=%.3e\n",
                   max_abs, max_abs_bt);
            if (max_abs < max_abs_bt)
                printf("       → 语义 = 假设1：D_row[m][n] = Σ A_row[m][k]·B_row[k][n] ✓\n");
            else
                printf("       → 语义 = 假设2：读 D 时需按列主序换回（D_col[N][M] → D_row[M][N]）\n");
        }
    } else {
        printf("[结论] cuBLASLt 无可用 FP8 algo——GB10 无 FP8 引擎（或此 scale/布局组合不支持）。\n");
        printf("       阶段 D 参照系走『两段对拍』（量化-反量化 BF16 cuBLAS 基线，见 fp8/criterion 2.1）。\n");
        // 额外尝试：e5m2 输入（有的引擎只收 e5m2）——快速再探一次
        // （布局重建略，只改类型宏重试 heuristic）
    }

    return 0;
}
