# nvGB10_GEMM — NVIDIA GB10 上的 BF16 GEMM 优化（v0 → v7）

在 NVIDIA GB10（DGX Spark，SM120/121 消费级 Blackwell）上，用纯 CUDA C++ 从零实现 BF16 GEMM，
经过 7 轮「消融探针定位 → 假设 → 实测证伪/采纳」迭代，从标量 1.26 TFLOPS 优化到 **54.3 TFLOPS（cuBLAS 的 62%）**，
全程与 cuBLAS 对拍精度（`max_rel_err = 0`，逐位一致）。

## 测试运行环境

| 项目 | 配置 |
|---|---|
| 硬件 | NVIDIA GB10（DGX Spark），48 SM，SM121（cc 12.1），Blackwell 消费级 |
| 峰值算力（BF16 mma.sync，实测） | 118.37 TFLOPS |
| 内存带宽（实测） | 255.6 GB/s（单向读，理论 273） |
| L2 cache | 24 MB |
| shared memory / SM | 100 KB |
| 操作系统 | Linux (aarch64) |
| CUDA Toolkit | 13.0（`nvcc -arch=sm_121`） |
| 编译器 | gcc 13.3（C++20） |
| cuBLAS | 13.1（作为精度参照与性能 baseline） |

**本机硬件边界（重要）**：
- SM121 只有 `mma.sync`（m16n8k16 BF16），**没有** `tcgen05`/`TMEM`（数据中心 Blackwell）、**没有** `WGMMA`（Hopper）、**没有** FP8/FP4 指令；
- 因此 DeepGEMM 的「smem 直读张量核」路径在本机不可用，必须走 `smem → ldmatrix → 寄存器 → mma.sync`。

## 如何编译与测试

```bash
# 所有版本同一编译方式（-lcublas 用于精度对拍）
nvcc -O3 -arch=sm_121 -lcublas -o gemm_v0 gemm_v0.cu
nvcc -O3 -arch=sm_121 -lcublas -o gemm_v1 gemm_v1.cu
# ... gemm_v2 ~ gemm_v7 同理

# 运行（默认 4096³，第 4 个参数任意值 = 跳过 CPU 正确性验证，只测性能）
./gemm_v7                          # 4096³，性能 + vs cuBLAS 精度对拍
./gemm_v7 2048 2048 2048           # 自定义尺寸
./gemm_v7 4096 4096 4096 skip      # 只测性能（大尺寸时跳过慢的 CPU 验证）

# 基准压测（测本机峰值算力与带宽）
cd bench
nvcc -O3 -arch=sm_121 -o bench_micro bench_micro.cu && ./bench_micro
nvcc -O3 -arch=sm_121 -lcublas -o cublas_bench cublas_bench.cu && ./cublas_bench

# tile 参数评估（gemm_v4，模板参数化 BM/BN/BK/WN）
nvcc -O3 -arch=sm_121 -lcublas -o gemm_v4 gemm_v4.cu && ./gemm_v4 4096 4096 4096
```

每个版本的输出包含：
- **性能**：`TFLOPS`、占峰值百分比（峰值 = 118.37 实测）
- **精度**：`max_abs_err / max_rel_err / rel_L2`（参照：同精度 cuBLAS，判据 `max_rel_err < 1e-3`）

## 优化守则（criterion.md）

每轮迭代严格走四步循环，产出 `reports/vN.md`：

```
Step 1 分析瓶颈（Roofline + 消融探针 + ptxas -v）
Step 2 制定方案（一次只改一个变量，写预期与回滚条件）
Step 3 保存版本（git commit 打点）
Step 4 验证（性能双百分比：峰值利用率 + 相对 cuBLAS；精度 vs 同精度 cuBLAS）
      → 回到 Step 1
```

两个铁律：**所有性能数字以压测为准，禁止推测**；**精度参照 = 同精度 cuBLAS**（不是 FP32 真值）。

## V1–V8 优化轨迹

### 成绩总览（4096³，同机实测）

| 版本 | 关键改动 | TFLOPS | 峰值% | cuBLAS% | 采纳 |
|---|---|---|---|---|---|
| v0 | 朴素标量（每线程 1 元素，global 直算） | 1.26 | 1.1% | 1.4% | ✅ |
| v1 | 标量 tile + shared memory | 3.24 | 2.7% | 3.7% | ✅ |
| v2 | **mma.sync m16n8k16 + ldmatrix**（上张量核） | 11.05 | 9.3% | 12.7% | ✅ |
| v3 | **cp.async 双缓冲流水**（重叠加载与计算） | 21.37 | 18.0% | 24.5% | ✅ |
| v4 | tile 参数化评估（4 组配置） | — | — | — | ❌ 假设证伪 |
| v5 | 加大 BK / 4-stage 多缓冲 | — | — | — | ❌ 假设证伪 |
| v6 | B 换 x4 ldmatrix / WN=32 | — | — | — | ❌ 假设证伪 |
| **v7** | **照抄 cuBLAS 配置 + block swizzle** | **54.31** | **45.9%** | **62.4%** | ✅ |
| **v8** | **TMA + mbarrier full/empty（零 `__syncthreads`）** | **59.2** | **50.0%** | **68.0%** | ✅ |

### 每一轮的瓶颈定位 → 结论

**v0 → v1：算术强度太低（memory-bound）**
- Roofline：naive 的 AI ≈ 4.9 FLOP/byte（数据被重复读 434 倍），转折点 463。
- v1 引入 tile + smem 复用，AI → 12.7，性能 ∝ AI（roofline 铁律实测验证）。

**v1 → v2：标量 CUDA core 是天花板**
- 实测：标量 FFMA 峰值仅 14.7 TFLOPS（张量核 118 的 1/8）。
- v2 换 `mma.sync` + `ldmatrix`（A 用 x4 的 2×2 象限地址、B 用 x4.trans，布局经 dump 逐位验证），跳到 11.05，精度 0 误差。

**v2 → v3：加载与计算串行**
- SASS 显示加载/计算完全串行、2 次 `__syncthreads` 硬阻塞。
- v3 用 `cp.async`（16B 粒度异步拷贝）+ 双缓冲，加载 stage k+1 与 mma stage k 重叠，1.9×。

**v4/v5/v6：三次假设被实测推翻（本项目最有价值的部分）**

| 版本 | 假设 | 实测 | 为什么错 |
|---|---|---|---|
| v4 | tile 太小 → 加大 BM/BN | compute-bound 下**最小 tile 反而最好** | 真瓶颈不在 tile 复用 |
| v5 | barrier 次数多 → 加大 BK / 4-stage | 无效甚至变差 | 2-stage 总等待时间 = 总加载时间，与 BK 无关 |
| v6 | ldsm 条数多 → B 换 x4 | -2% | x4 与 2×x2 **字节数相同**，省的只是 issue 槽 |
| v6 | ilp 探针 4 mma 摊薄延迟 → WN=32 | -20% | 探针是纯计算；真实 kernel B 访存翻倍 + occupancy 掉 29% |

三次失败后用**消融探针**（逐项去掉开销）+ **nsys 抓 cuBLAS kernel 名**定位到真相：

- 瓶颈不是带宽/条数/bank conflict/syncthreads，而是 **A 的 ldmatrix 延迟**（探针实测：纯 mma+sync 110 TFLOPS，加 A 依赖掉到 55）；
- **cuBLAS 4096³ 的 87 TFLOPS 用的就是相同的 `mma.sync` 指令**（kernel 名 `cutlass_80_tensorop_bf16_s16816gemm_128x256_32x3`），差距全在配置策略。

**v7：照抄 cuBLAS 配置，一步到位**

| 维度 | cuBLAS（CUTLASS 128×256×32） | v3 及以前 |
|---|---|---|
| warp tile | **64×64（32 个 mma、128 累加器/线程）** | 16×16（2 mma、8 累加器） |
| occupancy | 极低（198 寄存器，1-2 block/SM） | 高（36 寄存器，7 block/SM） |
| 策略 | **低 occupancy 换极高 ILP，摊薄 ldmatrix 延迟** | 高 occupancy 低 ILP |

v7 配置：BM=128, BN=256, BK=32，8 warps（256 线程，2×4 排布），每 warp 64×64，3 stage，smem 72KB（dynamic）。
再叠加 **block swizzle（GROUP_M=8）**：默认 n 优先调度下 B tile 复用距离太远（L2 24MB 装不下 64MB 数据），swizzle 组内 m 优先让 B tile 连续复用——8192³ 直接 +43%。

**v7 → v8：sync-bound → TMA + mbarrier 流水（+8.5%）**

- v7 配置消融探针：纯 mma 116.2 → +ldsm 106.2 → **+1×`__syncthreads` 71.0（-33%，主瓶颈）**——1 block/SM 下 barrier 期间整个 SM 空转；
- v8 用 mbarrier full/empty 双 barrier 替代全部 `__syncthreads`（producer=thread 0 发 TMA，consumer=warp leader 释放），4096³ 54.57→59.2，全尺寸 0 误差；
- SM121 上 `mbarrier.arrive` 的隐式 release 不排序先前 LDSM 读完成——释放 stage 前必须显式 `fence.acq_rel.cta`（正确性必需，代价约 -4%）；
- 4-stage 探针：+1.6%（噪声内）不采纳，TMA 到货延迟假设关闭。剩余 ~47 差距主嫌疑：mbarrier 自旋开销、epilogue 串行、grid 尾波。

### 核心方法论收获

1. **消融探针是定位指令级瓶颈的唯一手段**（本机 ncu counter 权限被锁）：逐项排除法最终定位到「ldsm 延迟依赖」，这是 Roofline 完全看不到的一层；
2. **nsys 抓 kernel 名 = 看到标准答案**：cuBLAS 的 kernel 名直接暴露了 tile/stage 配置；
3. **探针结论不能直接外推**：探针隔离单一变量，真实 kernel 有耦合副作用（v6 的教训）；
4. **每轮实测、不预设结论**：v4/v5/v6 三次推翻假设恰是这套流程的价值——失败也被记录、被证伪、沉淀成下一步依据。

## 目录结构

```
criterion.md        优化守则（四步循环 + 双百分比判定 + 精度标准）
gemm_v0.cu ~ v8.cu  9 个版本的 GEMM 实现（每个可独立编译运行）
reports/v0.md~v8.md 每版结论文档（瓶颈→方案→实测→教训）
bench/bench_micro.cu    峰值算力/带宽压测（标定 100% 标尺）
bench/cublas_bench.cu   cuBLAS baseline（性能参照）
bench/probe_v7cfg.cu    v7 配置消融探针（mma/ldsm/sync 逐项排除）
```

## License

MIT