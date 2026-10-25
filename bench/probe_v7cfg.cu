#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
using bf16 = __nv_bfloat16;
// v7 配置的消融探针：128×256×32 tile, 8 warps, 64×64 warp tile, 64 mma/warp, 128 acc
// INCLUDE 位掩码：bit0=A-ldsm, bit1=B-ldsm, bit2=__syncthreads
// 目的：精确定位 v7 的 55 TFLOPS（vs 纯 mma 116）差距到底在哪
template<int INCLUDE>
__global__ void __launch_bounds__(256, 1) probe(float* out, int Kb) {
    __shared__ bf16 As[128][32];
    __shared__ bf16 Bs[32][256];
    for (int i = threadIdx.x; i < 128*32; i += 256) As[i/32][i%32] = bf16(1.0f);
    for (int i = threadIdx.x; i < 32*256; i += 256) Bs[i/256][i%256] = bf16(1.0f);
    __syncthreads();
    int tid = threadIdx.x, warp = tid/32, lane = tid%32;
    int warp_m = warp/4, warp_n = warp%4;
    float c[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) c[i][j][0]=c[i][j][1]=c[i][j][2]=c[i][j][3]=0.f;
    uint32_t fa[4] = {0x3f803f80u,0x3f803f80u,0x3f803f80u,0x3f803f80u};
    uint32_t fb[2] = {0x3f803f80u,0x3f803f80u};
    int atile = lane/8, ar8 = lane%8;
    for (int k = 0; k < Kb; k++) {
        uint32_t a[2][4][4];
        if constexpr (INCLUDE & 1) {
            #pragma unroll
            for (int ki = 0; ki < 2; ki++)
                #pragma unroll
                for (int mi = 0; mi < 4; mi++) {
                    int arow = warp_m*64 + mi*16;
                    uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                        &As[arow + ar8 + (atile%2)*8][ki*16 + (atile/2)*8]);
                    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                        :"=r"(a[ki][mi][0]),"=r"(a[ki][mi][1]),"=r"(a[ki][mi][2]),"=r"(a[ki][mi][3]):"r"(addr));
                }
        }
        uint32_t b[2][4][4];
        if constexpr (INCLUDE & 2) {
            #pragma unroll
            for (int ki = 0; ki < 2; ki++)
                #pragma unroll
                for (int nj = 0; nj < 4; nj++) {
                    int brow = warp_n*64 + nj*16;
                    uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                        &Bs[lane%8 + (lane/8%2)*8 + ki*16][(lane/8/2)*8 + brow]);
                    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n"
                        :"=r"(b[ki][nj][0]),"=r"(b[ki][nj][1]),"=r"(b[ki][nj][2]),"=r"(b[ki][nj][3]):"r"(addr));
                }
        }
        #pragma unroll
        for (int ki = 0; ki < 2; ki++)
            #pragma unroll
            for (int mi = 0; mi < 4; mi++)
                #pragma unroll
                for (int nj = 0; nj < 4; nj++)
                    #pragma unroll
                    for (int half = 0; half < 2; half++) {
                        int ni = nj*2 + half;
                        uint32_t a0=(INCLUDE&1)?a[ki][mi][0]:fa[0], a1=(INCLUDE&1)?a[ki][mi][1]:fa[1];
                        uint32_t a2=(INCLUDE&1)?a[ki][mi][2]:fa[2], a3=(INCLUDE&1)?a[ki][mi][3]:fa[3];
                        uint32_t b0=(INCLUDE&2)?b[ki][nj][half*2]:fb[0];
                        uint32_t b1=(INCLUDE&2)?b[ki][nj][half*2+1]:fb[1];
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                            :"+f"(c[mi][ni][0]),"+f"(c[mi][ni][1]),"+f"(c[mi][ni][2]),"+f"(c[mi][ni][3])
                            :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
                    }
        if constexpr (INCLUDE & 4) __syncthreads();
    }
    float s=0;
    #pragma unroll
    for (int i=0;i<4;i++)
        #pragma unroll
        for (int j=0;j<8;j++) s+=c[i][j][0]+c[i][j][1]+c[i][j][2]+c[i][j][3];
    if(s==-12345.f) out[threadIdx.x]=s;
}
int main(){
    float* d; cudaMalloc(&d,1024);
    // 48 blocks（每 SM 一个，1 block/SM 匹配 v7 occupancy），Kb=128（匹配 4096³/BK）
    dim3 grid(48), block(256); int Kb=128;
    // FLOPs = 48 blocks × 8 warps × Kb × 64 mma × 4096 FLOPs
    double flop = (double)48*8*Kb*64*4096;
    struct {int inc; const char* name;} cases[] = {
        {0,"0 纯mma（上限）"}, {1,"1 +A-ldsm"}, {2,"2 +B-ldsm"},
        {3,"3 +A+B-ldsm"}, {7,"7 +ldsm+sync(完整计算)"}
    };
    for (auto& c : cases) {
        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        auto run=[&]{ switch(c.inc){
            case 0: probe<0><<<grid,block>>>(d,Kb); break;
            case 1: probe<1><<<grid,block>>>(d,Kb); break;
            case 2: probe<2><<<grid,block>>>(d,Kb); break;
            case 3: probe<3><<<grid,block>>>(d,Kb); break;
            case 7: probe<7><<<grid,block>>>(d,Kb); break;
        }};
        run(); cudaDeviceSynchronize();
        cudaEventRecord(s);
        for(int i=0;i<10;i++) run();
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms,s,e); ms/=10;
        printf("%-22s: %7.2f TFLOPS\n", c.name, flop/(ms/1e3)/1e12);
    }
    return 0;
}
