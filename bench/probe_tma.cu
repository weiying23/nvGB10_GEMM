#include <cuda_runtime.h>
#include <cstdint>
// 探针：sm_121 是否支持 TMA（cp.async.bulk.tensor）+ mbarrier expect_tx 模式（编译检查）
__global__ void k(float* out) {
    __shared__ __align__(128) float smem[64];
    __shared__ __align__(8) uint64_t mbar;
    uint32_t mb = (uint32_t)__cvta_generic_to_shared(&mbar);
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(smem);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(mb));
        asm volatile("fence.mbarrier_init.release.cluster;\n");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // TMA 标准模式：leader arrive_and_expect_tx(字节数)，TMA 硬件 complete_tx
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 256;\n" :: "r"(mb));
        const void* tmap = out;  // 仅编译检查（真实代码是 __grid_constant__ CUtensorMap）
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];\n"
            :: "r"(dst), "l"(tmap), "r"(0), "r"(0), "r"(mb));
    }
    uint32_t done = 0;
    while (!done)
        asm volatile("{\n\t.reg .pred p;\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n\t"
                     "selp.b32 %0,1,0,p;\n\t}\n" : "=r"(done) : "r"(mb));
    out[threadIdx.x] = smem[0];
}
int main(){ return 0; }
