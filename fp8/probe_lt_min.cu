// fp8/probe_lt_min.cu —— 阶段 C 修正版：cuBLASLt FP8 语义探针（非对称随机数据全量对拍）
//
// 教训（记录）：第一版单位阵检验（A=B=2I，M=N=K）通过是**假阳性**——
//   错误布局 (K,N,K) 读 buffer[k+n*K]，在平方对角数据上 k*N+k = k+n*K ⟺ k=n，
//   恰好在对角线自洽，4I 照样出现。对称输入无法暴露转置错误！
// 正确换法（canonical）：行主序 C_row[M][N] = A_row·B_row
//   ⇔ 列主序 C_col[N][M] = B_rowᵀ · A_rowᵀ
//   → cublasLt 第 1 操作数 = B 缓冲（layout (N,K,N)，opN，挂 B 的 scale）
//     第 2 操作数 = A 缓冲（layout (K,M,K)，opN，挂 A 的 scale）
//     D layout (N,M,N)（≡ D_row[M][N]）
// 验证：非对称形状（M≠N≠K）+ 随机 e4m3 精确值 + 全量 CPU 对拍（K 小，f32 累加精确 → 逐位一致）。
//
// 编译：nvcc -O3 -arch=sm_121 -o fp8/probe_lt_min fp8/probe_lt_min.cu -lcublasLt

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
    fprintf(stderr, "CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)
#define CKLT(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cublasLt error %d @%d\n", (int)s, __LINE__); return 1; } } while (0)

static float dq(unsigned char b) { __nv_fp8_e4m3 v; memcpy(&v, &b, 1); return (float)v; }

int main(int argc, char** argv) {
    // 非对称默认形状：任两维不相等（防对称盲区）
    const int M = argc > 1 ? atoi(argv[1]) : 64;
    const int N = argc > 2 ? atoi(argv[2]) : 96;
    const int K = argc > 3 ? atoi(argv[3]) : 48;
    const int perf = argc > 4 ? atoi(argv[4]) : 0;   // 1 = 大形状性能模式（跳过全量对拍）

    cublasLtHandle_t lt; CKLT(cublasLtCreate(&lt));

    // 随机 e4m3 精确值（[-4,4] 整数）：K ≤ 64 时 f32 累加精确 → 对拍必须逐位一致
    std::vector<unsigned char> hA((size_t)M * K), hB((size_t)K * N);
    srand(42);
    auto rq = []() {
        float v = (float)((rand() % 9) - 4);
        return (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    };
    for (auto& b : hA) b = rq();
    for (auto& b : hB) b = rq();

    unsigned char *dA, *dB; __nv_bfloat16* dD;
    CK(cudaMalloc(&dA, hA.size()));
    CK(cudaMalloc(&dB, hB.size()));
    CK(cudaMalloc(&dD, sizeof(__nv_bfloat16) * M * N));
    CK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));

    // ---- canonical 布局：D_col[N][M] = opN(B_col[N][K]) · opN(A_col[K][M]) ----
    cublasLtMatmulDesc_t op;
    CKLT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opN = CUBLAS_OP_N;
    CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    cublasLtMatrixLayout_t lB, lA, ld;   // 第 1 操作数 = B，第 2 操作数 = A
    CKLT(cublasLtMatrixLayoutCreate(&lB, CUDA_R_8F_E4M3, N, K, N));   // B_row ≡ B_col[N][K] ld=N
    CKLT(cublasLtMatrixLayoutCreate(&lA, CUDA_R_8F_E4M3, K, M, K));   // A_row ≡ A_col[K][M] ld=K
    CKLT(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, N, M, N));       // D_col[N][M] ≡ D_row[M][N]

    // scale=1（数据值域已可表示；scale 结构仍完整挂上——与阶段 D per-tensor 同构）
    float hS = 1.0f, *dS;
    CK(cudaMalloc(&dS, 4)); CK(cudaMemcpy(dS, &hS, 4, cudaMemcpyHostToDevice));
    cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm)));
    CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm)));
    CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dS, sizeof(dS)));
    CKLT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dS, sizeof(dS)));

    cublasLtMatmulPreference_t pref;
    CKLT(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = 32 << 20;
    CKLT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    cublasLtMatmulHeuristicResult_t h[8]; int nr = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, op, lB, lA, ld, ld, pref, 8, h, &nr);
    if (nr == 0) { printf("heuristic 无 algo (status=%d) M=%d N=%d K=%d\n", (int)st, M, N, K); return 1; }

    float alpha = 1.f, beta = 0.f;
    int ok = -1;
    for (int i = 0; i < nr; ++i) {
        st = cublasLtMatmul(lt, op, &alpha, dB, lB, dA, lA, &beta, dD, ld, dD, ld,
                            &h[i].algo, nullptr, 0, nullptr);
        if (st == CUBLAS_STATUS_SUCCESS) { ok = i; break; }
    }
    if (ok < 0) { printf("全部 algo 失败 M=%d N=%d K=%d\n", M, N, K); return 1; }

    // ---- 全量 CPU 对拍（K≤64 精确 → 逐位一致门限；perf 模式跳过：大形状全量对拍是 CPU 分钟级）----
    if (!perf) {
    std::vector<__nv_bfloat16> hD((size_t)M * N);
    CK(cudaMemcpy(hD.data(), dD, sizeof(__nv_bfloat16) * M * N, cudaMemcpyDeviceToHost));
    double max_abs = 0; int bad = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float ref = 0;
            for (int k = 0; k < K; ++k)
                ref += dq(hA[(size_t)m * K + k]) * dq(hB[(size_t)k * N + n]);
            double got = __bfloat162float(hD[(size_t)m * N + n]);
            double d = fabs(got - ref);
            if (d > max_abs) max_abs = d;
            if (d != 0.0) ++bad;
        }
    printf("[语义对拍 M=%d N=%d K=%d 非对称] max_abs=%.3e  逐位差=%d/%d  %s\n",
           M, N, K, max_abs, bad, M * N, bad == 0 ? "[PASS 逐位一致]" : "[检查]");
    }

    // ---- 性能模式（大形状，协议 v2 时间制热身）----
    if (perf) {
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        int cnt = 0;
        do {
            cublasLtMatmul(lt, op, &alpha, dB, lB, dA, lA, &beta, dD, ld, dD, ld,
                           &h[ok].algo, nullptr, 0, nullptr);
            if (++cnt % 64 == 0) {
                cudaEventRecord(e); cudaEventSynchronize(e);
                float wms = 0; cudaEventElapsedTime(&wms, s, e);
                if (wms > 30.f) break;
            }
        } while (cnt < 1000000);
        cudaDeviceSynchronize();
        cudaEventRecord(s);
        const int iters = 20;
        for (int i = 0; i < iters; ++i)
            cublasLtMatmul(lt, op, &alpha, dB, lB, dA, lA, &beta, dD, ld, dD, ld,
                           &h[ok].algo, nullptr, 0, nullptr);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;
        double tflops = 2.0 * M * N * K / (ms / 1e3) / 1e12;
        printf("[性能 M=%d N=%d K=%d] %.3f ms, %.4f TFLOPS (FP8 峰值 235.79 的 %.2f%%)\n",
               M, N, K, ms, tflops, tflops / 235.79 * 100);
    }
    return 0;
}
