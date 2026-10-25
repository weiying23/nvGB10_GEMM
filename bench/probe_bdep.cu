#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
using bf16 = __nv_bfloat16;
// B 依赖判别：A 恒固定、A-ldsm 恒执行(死)、B-ldsm 恒执行，mma 是否消费 B 结果决定依赖
// MODE 0: mma 消费固定 B（B-ldsm 死）——基线
// MODE 1: mma 消费 B-ldsm x2.trans 结果（复现更正后的主因工况）
// MODE 2: mma 消费 B-ldsm x2 非trans 结果（判别 trans 特异性；布局无所谓，只测依赖延迟）
template<int MODE>
__global__ void k(float* out, int Kb) {
  __shared__ bf16 As[64][16]; __shared__ bf16 Bs[16][32];
  for(int i=threadIdx.x;i<64*16;i+=256) As[i/16][i%16]=bf16(1.0f);
  for(int i=threadIdx.x;i<16*32;i+=256) Bs[i/32][i%32]=bf16(1.0f);
  __syncthreads();
  int tid=threadIdx.x, lane=tid%32, warp=tid/32;
  int warp_m=warp/2, warp_n=warp%2;
  float c[2][4]={{0,0,0,0},{0,0,0,0}};
  uint32_t fa0=0x3f803f80u,fa1=0x3f803f80u,fa2=0x3f803f80u,fa3=0x3f803f80u;
  uint32_t fb0=0x3f803f80u,fb1=0x3f803f80u;
  for(int kk=0;kk<Kb;kk++){
    uint32_t a[4];
    int arow=warp_m*16, at=lane/8, ar=lane%8;
    uint32_t aa=(uint32_t)__cvta_generic_to_shared(&As[arow+ar+(at%2)*8][(at/2)*8]);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(aa));
    if(MODE==4){
      uint32_t b4[4];
      {
        int brow=warp_n*16;
        uint32_t ba=(uint32_t)__cvta_generic_to_shared(&Bs[lane%8+(lane/8%2)*8][(lane/8/2)*8+brow]);
        asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n":"=r"(b4[0]),"=r"(b4[1]),"=r"(b4[2]),"=r"(b4[3]):"r"(ba));
      }
      #pragma unroll
      for(int ni=0;ni<2;ni++){
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
          :"+f"(c[ni][0]),"+f"(c[ni][1]),"+f"(c[ni][2]),"+f"(c[ni][3])
          :"r"(fa0),"r"(fa1),"r"(fa2),"r"(fa3),"r"(b4[ni*2]),"r"(b4[ni*2+1]));
      }
    } else if(MODE==3){
      uint32_t bv[2][2];
      #pragma unroll
      for(int ni=0;ni<2;ni++){
        int brow=warp_n*16+ni*8;
        uint32_t ba=(uint32_t)__cvta_generic_to_shared(&Bs[lane%8+(lane/8%2)*8][brow]);
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(bv[ni][0]),"=r"(bv[ni][1]):"r"(ba));
      }
      #pragma unroll
      for(int ni=0;ni<2;ni++){
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
          :"+f"(c[ni][0]),"+f"(c[ni][1]),"+f"(c[ni][2]),"+f"(c[ni][3])
          :"r"(fa0),"r"(fa1),"r"(fa2),"r"(fa3),"r"(bv[ni][0]),"r"(bv[ni][1]));
      }
    } else {
    #pragma unroll
    for(int ni=0;ni<2;ni++){
      uint32_t b[2];
      int brow=warp_n*16+ni*8;
      uint32_t ba=(uint32_t)__cvta_generic_to_shared(&Bs[lane%8+(lane/8%2)*8][brow]);
      if(MODE==2)
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(b[0]),"=r"(b[1]):"r"(ba));
      else
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(b[0]),"=r"(b[1]):"r"(ba));
      uint32_t b0=(MODE==0)?fb0:b[0], b1=(MODE==0)?fb1:b[1];
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        :"+f"(c[ni][0]),"+f"(c[ni][1]),"+f"(c[ni][2]),"+f"(c[ni][3])
        :"r"(fa0),"r"(fa1),"r"(fa2),"r"(fa3),"r"(b0),"r"(b1));
    }
    }
    __syncthreads();
  }
  float s=0;for(int i=0;i<2;i++)s+=c[i][0]+c[i][1]+c[i][2]+c[i][3];
  if(s==-12345.f)out[threadIdx.x]=s;
}
int main(){
  float* d; cudaMalloc(&d,1024);
  dim3 grid(64,32), block(256); int Kb=128;
  double flop=2.0*4096*8*Kb*grid.x*grid.y;
  const char* nm[]={"0 B-ldsm死(基线)","1 B依赖 x2.trans","2 B依赖 x2非trans","3 B依赖+片段提前发出","4 B依赖 x4.trans(v6结构)"};
  for(int m=0;m<5;m++){
    cudaEvent_t s,e; cudaEventCreate(&s);cudaEventCreate(&e);
    auto run=[&]{ if(m==0)k<0><<<grid,block>>>(d,Kb); if(m==1)k<1><<<grid,block>>>(d,Kb); if(m==2)k<2><<<grid,block>>>(d,Kb); if(m==3)k<3><<<grid,block>>>(d,Kb); if(m==4)k<4><<<grid,block>>>(d,Kb); };
    run(); cudaDeviceSynchronize();
    cudaEventRecord(s); for(int i=0;i<10;i++) run(); cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e); ms/=10;
    printf("%-22s: %.2f TFLOPS\n", nm[m], flop/(ms/1e3)/1e12);
  }
  return 0;
}
