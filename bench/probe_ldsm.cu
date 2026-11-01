// LDSM bank conflict 探针：分别测 v7/v8 的 A 模式（x4，行距64B）与 B 模式（x4.trans，行距512B）
// 对照组：A-padded（行距80B=16B×5，与8互质→无冲突）、理想连续（行距16B）
// 输出每 SM 每周期完成的 ldsm.x4 条数（无冲突时 ~1/4clk 为理论上限形态，比较相对值即可）
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
using bf16 = __nv_bfloat16;

__device__ __forceinline__ void ldsm_x4(uint32_t* d, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4t(uint32_t* d, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(addr));
}

// MODE: 0=A(64B行距) 1=A-padded(80B) 2=B(512B行距,trans) 3=理想(16B行距)
template<int MODE>
__global__ void __launch_bounds__(256, 1) probe(uint32_t* out, int iters) {
    __shared__ bf16 As[128][32];    // 行距 64B（v7/v8 的 A）
    __shared__ bf16 Ap[128][40];    // 行距 80B（无冲突对照）
    __shared__ bf16 Bs[32][256];    // 行距 512B（v7/v8 的 B）
    __shared__ bf16 Id[128][16];    // 行距 32B… 改用真正的 16B 行距见下
    __shared__ bf16 Ic[2048][8];   // 行距 16B（理想：8×8 连续 128B）
    for (int i = threadIdx.x; i < 128*32; i += 256) As[i/32][i%32] = bf16(1.f);
    for (int i = threadIdx.x; i < 128*40; i += 256) Ap[i/40][i%40] = bf16(1.f);
    for (int i = threadIdx.x; i < 32*256; i += 256) Bs[i/256][i%256] = bf16(1.f);
    for (int i = threadIdx.x; i < 2048*8; i += 256) Ic[i/8][i%8] = bf16(1.f);
    __syncthreads();

    const int lane = threadIdx.x % 32;
    uint32_t d[4];
    float acc = 0;
    for (int it = 0; it < iters; ++it) {
        // 每 8 次换一个基址（防 CSE），冲突结构相同
        int rot = (it & 7);
        if (MODE == 0) {   // A 模式：&As[arow + r8 + (t%2)*8 + rot][(t/2)*8]，行距 64B
            int t = lane / 8, r8 = lane % 8;
            uint32_t a = (uint32_t)__cvta_generic_to_shared(
                &As[(r8 + (t % 2) * 8 + rot) % 128][(t / 2) * 8]);
            ldsm_x4(d, a);
        } else if (MODE == 1) {  // A-padded：同模式，行距 80B
            int t = lane / 8, r8 = lane % 8;
            uint32_t a = (uint32_t)__cvta_generic_to_shared(
                &Ap[(r8 + (t % 2) * 8 + rot) % 128][(t / 2) * 8]);
            ldsm_x4(d, a);
        } else if (MODE == 2) {  // B 模式：&Bs[r8 + (t%2)*8][(t/2)*8 + rot*16]，行距 512B
            int t = lane / 8, r8 = lane % 8;
            uint32_t a = (uint32_t)__cvta_generic_to_shared(
                &Bs[(r8 + (t % 2) * 8) % 32][(t / 2) * 8 + rot * 16]);
            ldsm_x4t(d, a);
        } else {             // 理想：Ic[r8 + t%2*8 + rot][...]，行距 16B
            int t = lane / 8, r8 = lane % 8;
            uint32_t a = (uint32_t)__cvta_generic_to_shared(
                &Ic[(r8 + (t % 2) * 8 + rot) % 2048][t / 2]);
            ldsm_x4(d, a);
        }
        acc += __uint_as_float(d[0]) + __uint_as_float(d[3]);
    }
    if (acc == -12345.f) out[0] = (uint32_t)acc;
}

int main() {
    uint32_t* d; cudaMalloc(&d, 8);
    int iters = 100000;
    const char* names[] = {"A模式(行距64B)", "A-padded(行距80B)", "B模式(行距512B,trans)", "理想(行距16B)"};
    for (int mode = 0; mode < 4; ++mode) {
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        auto run = [&]{ switch (mode) {
            case 0: probe<0><<<48,256>>>(d,iters); break;
            case 1: probe<1><<<48,256>>>(d,iters); break;
            case 2: probe<2><<<48,256>>>(d,iters); break;
            case 3: probe<3><<<48,256>>>(d,iters); break; } };
        run(); cudaDeviceSynchronize();
        cudaEventRecord(s); run(); cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        // 总 ldsm 数 = 48 block × 8 warp × iters；每 SM 每 clk 条数
        double total = 48.0 * 8 * iters;
        double per_sm_clk = total / (ms / 1e-3) / 48.0 / 2.405e9;
        printf("%-24s: %8.2f ms, ldsm.x4/SM/clk = %.3f（相对理想 = %.0f%%）\n",
               names[mode], ms, per_sm_clk, per_sm_clk / 0.25 * 100 > 999 ? 999 :
               per_sm_clk * 400);
        cudaEventDestroy(s); cudaEventDestroy(e);
    }
    return 0;
}
