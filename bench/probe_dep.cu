#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
using bf16 = __nv_bfloat16;
// 判别探针：A-ldsm→mma 依赖代价为何从 -9%(无B ldsm) 变 -51%(有死B ldsm)?
// MODE 0: 无 B ldsm（复现 locate2 基准）
// MODE 1: 2 条死 B-ldsm x2.trans 在 mma 前（复现 locate3）
// MODE 2: 2 条死 B-ldsm x2 非 trans 在 mma 前（判别 .trans 特异性）
// MODE 3: 8 条死 FFMA 在 mma 前（判别通用指令槽）
// MODE 4: 2 条死 B-ldsm x2.trans 在 mma 后（判别位置：是否必须在生产者-消费者窗口内）
template<int MODE>
__global__ void k(float* out, int Kb) {
  __shared__ bf16 As[64][16]; __shared__ bf16 Bs[16][32];
  for(int i=threadIdx.x;i<64*16;i+=256) As[i/16][i%16]=bf16(1.0f);
  for(int i=threadIdx.x;i<16*32;i+=256) Bs[i/32][i%32]=bf16(1.0f);
  __syncthreads();
  int tid=threadIdx.x, lane=tid%32, warp=tid/32;
  int warp_m=warp/2, warp_n=warp%2;
  float c[2][4]={{0,0,0,0},{0,0,0,0}};
  uint32_t fb0=0x3f803f80u,fb1=0x3f803f80u;
  float dummy=1.0f;
  auto deadB = [&](){
    #pragma unroll
    for(int ni=0;ni<2;ni++){
      uint32_t b[2];
      int brow=warp_n*16+ni*8;
      uint32_t ba=(uint32_t)__cvta_generic_to_shared(&Bs[lane%8+(lane/8%2)*8][brow]);
      if(MODE==2)
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(b[0]),"=r"(b[1]):"r"(ba));
      else
        asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];\n":"=r"(b[0]),"=r"(b[1]):"r"(ba));
    }
  };
  for(int kk=0;kk<Kb;kk++){
    uint32_t a[4];
    int arow=warp_m*16, at=lane/8, ar=lane%8;
    uint32_t aa=(uint32_t)__cvta_generic_to_shared(&As[arow+ar+(at%2)*8][(at/2)*8]);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];\n":"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(aa));
    if(MODE==1||MODE==2) deadB();
    if(MODE==3){
      #pragma unroll
      for(int i=0;i<8;i++) asm volatile("fma.rn.f32 %0,%0,%0,%0;\n":"+f"(dummy));
    }
    #pragma unroll
    for(int ni=0;ni<2;ni++){
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        :"+f"(c[ni][0]),"+f"(c[ni][1]),"+f"(c[ni][2]),"+f"(c[ni][3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(fb0),"r"(fb1));
    }
    if(MODE==4) deadB();
    __syncthreads();
  }
  float s=0;for(int i=0;i<2;i++)s+=c[i][0]+c[i][1]+c[i][2]+c[i][3];
  s+=dummy; if(s==-12345.f)out[threadIdx.x]=s;
}
int main(){
  float* d; cudaMalloc(&d,1024);
  dim3 grid(64,32), block(256); int Kb=128;
  double flop=2.0*4096*8*Kb*grid.x*grid.y;
  const char* nm[]={"0 无B ldsm(基准)","1 死B x2.trans@mma前","2 死B x2非trans@mma前","3 8条死FFMA@mma前","4 死B x2.trans@mma后"};
  for(int m=0;m<5;m++){
    cudaEvent_t s,e; cudaEventCreate(&s);cudaEventCreate(&e);
    auto run=[&]{ switch(m){
      case 0:k<0><<<grid,block>>>(d,Kb);break; case 1:k<1><<<grid,block>>>(d,Kb);break;
      case 2:k<2><<<grid,block>>>(d,Kb);break; case 3:k<3><<<grid,block>>>(d,Kb);break;
      case 4:k<4><<<grid,block>>>(d,Kb);} };
    run(); cudaDeviceSynchronize();
    cudaEventRecord(s); for(int i=0;i<10;i++) run(); cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e); ms/=10;
    printf("%-24s: %.2f TFLOPS\n", nm[m], flop/(ms/1e3)/1e12);
  }
  return 0;
}
