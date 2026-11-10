# nvGB10_GEMM — NVIDIA GB10 上的 BF16 GEMM 优化（v0 → v11）

在 NVIDIA GB10（DGX Spark，SM121 消费级 Blackwell）上，用纯 CUDA C++ 从零实现 BF16 GEMM，
经过 11 轮「roofline/消融探针定位 → 假设 → 实测证伪/采纳」迭代，从标量 1.26 TFLOPS 优化到 **88.9 TFLOPS（cuBLAS 87.10 的 102%）**，
全程与 cuBLAS 对拍精度（`max_rel_err = 0`，逐位一致）。每轮的完整记录见 `reports/`（v0–v12），
硬件全景见 `hardware.md`。

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
- SM121 的张量核指令面 = `mma.sync` 家族（BF16/FP16/TF32；**FP8 e4m3/e5m2 实测支持且峰值 235.79 TFLOPS = BF16 的 2 倍**；FP4 未验证），**没有** `tcgen05`/`TMEM`（数据中心 Blackwell）、**没有** `WGMMA`（Hopper）；
- 因此 DeepGEMM 的「smem 直读张量核」路径在本机不可用，必须走 `smem → ldmatrix → 寄存器 → mma.sync`；
- 完整硬件参数、实测怪癖（LDSM bank conflict 规律、2 的幂全局步长惩罚等）见 `hardware.md`。

## 如何编译与测试

```bash
# v0 ~ v7（-lcublas 用于精度对拍）
nvcc -O3 -arch=sm_121 -lcublas -o gemm_v1 gemm_v1.cu        # v0~v7 同理

# v8 及以后：TMA 需要 -lcuda（tensor map 走驱动 API）
nvcc -O3 -arch=sm_121 -lcuda -lcublas -o gemm_v11 gemm_v11.cu

# 运行（默认 4096³，第 4 个参数任意值 = 跳过对拍，只测性能）
./gemm_v11                         # 4096³，性能 + vs cuBLAS 精度对拍
./gemm_v11 4096 4096 4096 skip     # 只测性能
./gemm_v11 2048 2016 4032          # 非方形/非 2 幂（M/N/K 需为 8 的倍数）

# 基准压测与探针（bench/ 下 20+ 份，每份可独立编译复现对应报告的测量）
cd bench
nvcc -O3 -arch=sm_121 -o bench_micro bench_micro.cu && ./bench_micro      # 峰值标定
nvcc -O3 -arch=sm_121 -lcublas -o cublas_bench cublas_bench.cu && ./cublas_bench
nvcc -O3 -arch=sm_121 -o probe_v7cfg probe_v7cfg.cu && ./probe_v7cfg      # v7 消融探针
nvcc -O3 -arch=sm_121 -lcuda -o probe_tmaswiz probe_tmaswiz.cu && ./probe_tmaswiz  # TMA swizzle 公式验证
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

## V1–V12 优化轨迹

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
| v7 | 照抄 cuBLAS 配置 + block swizzle | 54.31 | 45.9% | 62.4% | ✅ |
| **v8** | TMA + mbarrier full/empty（零 `__syncthreads`） | 59.2 | 50.0% | 68.0% | ✅ |
| v9 | padded smem 布局消 LDSM bank conflict（A 4路/B 8路→0） | 67.9 | 57.4% | 78.0% | ✅ |
| v10 | TMA + SWIZZLE_128B + BK=64（nvjet 路线） | 84.1 | 71.1% | 96.6% | ✅ |
| v11 | epilogue：smem 暂存 + TMA store | 88.9 | 75.1% | 102% | ✅ |
| **v12** | **配置族 + 形状分派（小 tile 128×64）** | **82.1** | **69.4%** | **94.3%** | ✅ |

> **⚠️ "cuBLAS%" 列的口径说明（读数前必看）**：
> - 该列以 **4096³** 为代表形状，对照 2026-10-21 缓存的 cuBLAS 基线（87.10）；
> - **"反超 cuBLAS"（≥100%）目前仅成立于 4096³**——8192³ 为 94.6/96.15 = **98.4%（落后 2–5%）**，8064³（非 2 幂）为 95.1/98.8 = **96.2%（落后 3.8%）**；
> - 4096³ 的反超含**2 的幂步长病理**成分：K=4096 时 cuBLAS 受 L2/DRAM 冲突影响比方更重（当日复测 77.6–86.8、方差巨大；我方稳定 87.4–89.0），在 cuBLAS 不踩坑的非 2 幂形状上我们仍落后；
> - 8192³/8064³ 落后的主体已定位为**写回（-5.5 消融实测）+ 同步残余**，且 8192³ 是 cuBLAS 最强的 nvjet 引擎主场；
> - v12 行的 94.3% 为当日 4096³ 三次均值 vs 历史基线，与 v11 同 kernel（差异为会话方差）。

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
- 4-stage 探针：+1.6%（噪声内）不采纳，TMA 到货延迟假设关闭。

**v8 → v9：LDSM bank conflict（+14.8%）**

- 发现链：片段双缓冲实验零提升 → "cuBLAS 87 > 零同步墙 77"之谜 → 纸面分析旧布局 A 行距 64B→**4 路冲突**、B 行距 512B→**8 路冲突**（微基准实测 4.0×/8.0×）→ kernel 级验证：零冲突 padded 探针 76.8→**112.4（+46%）**，"77 墙"完全是冲突伪影；
- 修复 = padding（A 行距 80B、B 行距 528B，均与 128B 银行周期互质），只改 smem 数组声明；4096³ 59.2→67.9（cuBLAS 78.0%）、8192³ 63.9→75.3；
- 方法论收获：**探针的寻址模式必须与真实 kernel 逐位一致**（v5 的"ldsm 带宽排除"用的是无冲突模式，掩盖了冲突）；
- 下一步：v8 的 mbarrier 零同步叠加（+8.5% 预期）→ TMA SWIZZLE_128B + BK=64（nvjet 正统路线）。

**v9 → v10：TMA + SWIZZLE_128B + BK=64（+24%）——大形状追平 cuBLAS**

- 合流点：padded（零冲突）与 mbarrier 零同步原本互斥（TMA 写不了 padded 布局），TMA 的 `SWIZZLE_128B` 模式一次解决——硬件写入时自动 XOR 交错，同时拿到「零 bank conflict + TMA + 零 `__syncthreads`」；
- 落位公式 `物理偏移 = r*128 + ((c/8) ^ (r%8))*16 + (c%8)*2` 先用探针逐位验证（12288 元素 0 差异），主 kernel 一次通过零调试；
- BK=64 + 2-stage（nvjet 同款）、B 拆 4 个 {64,64} box（warp_n 天然对应 chunk）、mma 分 2 pair 处理保片段寄存器不膨胀（190 regs）；
- 4096³ 67.9→**84.1（cuBLAS 96.6%）**、8192³ 75.3→**91.7（95.4%）**、全尺寸 +19–24%；
- 剩余缺口转移：1024³ 的 30%（SM 填充病，autotune/split-K 地盘）> 2048³ 的 20% > 大形状的 3–5%。

**v10 → v11：epilogue 优化（smem 暂存 + TMA store，+9%）——4096³ 反超 cuBLAS（8192³ 仍落后 2%，见口径说明）**

- 指示：v10 的 epilogue 消融探针（`bench/probe_epi10.cu`）实测写回是第一大单项瓶颈——2048³ -32%、4096³ -20%（store 碎片化：每线程 2×4B 对散写，实测写带宽仅 ≈183 GB/s；且 1 block/SM 下与 mma 完全串行）；
- 两个变体同场对比：float2 向量写（5 行改动，小尺寸 +9–39%）与 smem 暂存 + TMA store（4-pass 双缓冲流水 + bulk group 完成跟踪，中大尺寸赢家）——采纳后者，前者保留为 `gemm_v11a.cu`（512³/8192³ 略优，epilogue 风格记为 autotune 维度）；
- 踩到并修掉一个新 race：smem 从读侧（ldsm）转写侧（暂存）缺屏障，warp 错位时落后 warp 用 C 数据算 mma——**奇数 K 块必触发**（判别实验锁定规律后根因自明）；
- 实测：4096³ 84.1→**88.9（cuBLAS 87.10 的 102%，仅此尺寸反超）**、8192³ 94.6 = cuBLAS 的 98.4%（仍落后——写回+同步残余，见上方口径说明）、2048³ +11.5%、1024³ +27%（写回对短 K 的摊薄伤害最大，正是本轮收益来源）；精度全尺寸 + 奇数 K 块回归 + 非方形全 PASS。

**v11 → v12：配置族 + 形状分派（小形状 +55–126%）**

- 指示：单配置打天下是中小形状最大缺口——512³ 仅 8 block/48 SM（cuBLAS 的 31%）；
- 扫描（吸取 v4 教训先列全维度）：结构约束 WN=64 恒定（warp_n↔B chunk）、WM 派生，六配置 BM×BN 全扫 + 边界形状定阈值；
- 分派规则一行：大 tile block 数 < 48（SM 数）→ 128×64（88 regs/48KB smem），否则 128×256（=v11，零回归）；
- 实测：512³ **+123%**（11.8→26.4，cuBLAS 31%→69%）、768³ +55%、256³ +126%；2048³+ 同 kernel 噪声内；
- 同期证伪：warp 特化生产者全尺寸 -3–6%（sm121 小 SM 上专职 warp 反成负担，见 v11.md 附加探针）。

### 核心方法论收获

1. **消融探针是定位指令级瓶颈的唯一手段**（本机 ncu counter 权限被锁）：逐项排除法最终定位到「ldsm 延迟依赖」，这是 Roofline 完全看不到的一层；
2. **nsys 抓 kernel 名 = 看到标准答案**：cuBLAS 的 kernel 名直接暴露了 tile/stage 配置；
3. **探针结论不能直接外推**：探针隔离单一变量，真实 kernel 有耦合副作用（v6 的教训）；
4. **每轮实测、不预设结论**：v4/v5/v6 三次推翻假设恰是这套流程的价值——失败也被记录、被证伪、沉淀成下一步依据。

## 目录结构

```
criterion.md          优化守则（四步循环 + 双百分比判定 + 精度标准）
hardware.md           GB10 硬件全景（组织架构图 + 全实测参数 + 8 条怪癖 + cuBLAS 引擎地图）
gemm_v0.cu ~ v11.cu   12 个版本的 GEMM 实现（每个可独立编译运行）
gemm_v11a.cu          v11 的 float2 写回变体（小尺寸/8192³ 略优）
reports/v00.md~v11.md  每版完整记录（roofline 指示→方案→实测→关键代码→下一步）
bench/                基准与探针（每份对应报告中某个具体测量的可复现证据）：
  bench_micro.cu        峰值算力/带宽标定（118.37 / 255.6）
  cublas_bench.cu       cuBLAS baseline
  probe_v7cfg.cu        v7 配置消融（mma/ldsm/sync 逐项排除）
  probe_ldsm.cu         LDSM bank conflict 微基准（4 路/8 路实测）
  probe_tmaswiz.cu      TMA SWIZZLE_128B 落位公式逐位验证
  probe_epi10/11.cu     v10/v11 写回消融
  probe_sync11.cu       v11 同步协议消融（-17 的定位证据）
  ...（其余见各报告引用）
```

## License

MIT