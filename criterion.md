# GEMM 优化迭代守则（Criterion）

> 本文件定义在本机（NVIDIA GB10 / Blackwell SM120）上优化矩阵乘内核时，每一轮迭代必须遵循的四步流程与判定标准。
> 目标：让每一次改动**可量化、可回溯、可复现**，避免"凭感觉调参"。
>
> **两条铁律**：
> 1. 所有性能数字**以实测（压测）为准，禁止使用推测值**。
> 2. 每一轮结果用**两个百分比**判定：达成峰值算力的百分比，以及达成 cuBLAS 的百分比。

---

## 0. 目的与基线

### 0.1 核心评价指标（两个百分比，缺一不可）

**① 峰值利用率**（相对硬件上限）：

```
有效算力 (TFLOPS) = 2 × M × N × K / t
峰值利用率 (%) = 有效算力 / 峰值算力(实测) × 100%
```

**② 相对 cuBLAS 的性能**（相对成熟库）：

```
相对 cuBLAS (%) = 有效算力(本实现) / 有效算力(cuBLAS baseline) × 100%
```

- 峰值利用率目标：大形状（如 4096³）**≥ 90%**（最终目标）；中等形状 70~85%；小形状（任一维 < 512）目标放宽，但必须记录。
- 相对 cuBLAS 目标：**≥ 100% = "追平"**，是每轮及格线。参照：本机 cuBLAS BF16 约 81% 峰值（8192³，见 §0.4），故**峰值利用率超过 81% 即已超过 cuBLAS**。
- **判定口径**：本轮是否有效，以这两个百分比的**变化**为准（见 §5.3）。

### 0.2 硬件基线（实测，勿改）

| 参数 | 值 |
|---|---|
| GPU | NVIDIA GB10，compute capability **12.1** |
| SM 数量 | **48** |
| SM 时钟 | 2.418 GHz（boost 可至 3.0 GHz，会波动） |
| 内存 | 统一内存 LPDDR5X，256-bit @ 8533 GT/s |
| 理论内存带宽 | 273 GB/s（**单向**峰值，NVIDIA 官方；实测读 ~256 GB/s，见 §0.3） |
| shared memory / SM | 100 KB |
| 寄存器 / SM | 65536 |
| L2 cache | 24 MB |

### 0.3 峰值算力：以压测为准（不推测）

峰值算力**只以压测结果为准，禁止引用任何推测值**。下表为 2026-10-21 实测结果（压测代码见 `bench/bench_micro.cu`）；每次评估前用附录 A 重新校准，作为本轮 100% 标尺。

| 精度 | 峰值算力（实测 TFLOPS） | 备注 |
|---|---|---|
| BF16 | **118.37** | 与 FP16 同量级 |
| TF32 | **59.19** | 恰好为 BF16 一半 |
| FP8 | **不支持** | sm_121 不支持 `kind::f8f6f4` mma.sync，也无 tcgen05 |
| FP4 | **不支持** | 同上 |

| 项目 | 实测值 | 备注 |
|---|---|---|
| 实际内存带宽（**单向**读） | **255.6 GB/s** | 官方 273 GB/s 的 93.6% |

**官方数据对照**（来源：nvidia.com DGX Spark 产品页 / TechPowerUp）：

| 项目 | NVIDIA 官方 | 本机实测 | 说明 |
|---|---|---|---|
| 内存带宽 | 273 GB/s（单向） | 255.6 GB/s 读 | 吻合 |
| SM boost 时钟 | 2418 MHz | 2418 MHz | 吻合 |
| 张量核 | 5th Gen，FP4 达 1 PFLOP（**sparse**） | — | dense ≈ 500 TFLOPS FP4 |
| BF16 峰值 | 未直接公布（由 FP4 sparse 推算约 125 TFLOPS） | 118.37 TFLOPS | 达推算值 94.7% |

> **守则**：峰值算力与内存带宽一律以上表实测为准；重新压测后需更新本表。不压测的利用率不可信。
>
> **本机约束（重要）**：GB10（sm_121）手写 PTX 层面，张量核仅支持 BF16/FP16/TF32 的 `mma.sync`（m16n8k16 / m16n8k8）。`kind::f8f6f4`、`mxf8f6f4`、`mxf4`、`tcgen05.mma` 均被 ptxas 报 "not supported on sm_121"——官方宣传的 FP4 1 PFLOP 是稀疏值且依赖 cuDNN/cuBLAS 软件栈，**手写 GEMM 无法使用 FP8/FP4 原生张量核**。故本机 GEMM 优化只针对 BF16（可选 FP16/TF32）。

### 0.4 cuBLAS 基线（固定参照，按形状缓存，只测一次）

- 每个**首次评测到的形状**，用 cuBLAS / cuBLASLt 实测一次有效算力，作为该形状的固定 baseline。
- **只测一次并缓存**：结果写入 `baselines/cublas.json`，后续迭代直接查表引用，**不重测**。
- 缓存记录：形状、dtype、有效算力、日期、commit、计时方法。

已实测的 cuBLAS baseline（2026-10-21，压测代码见 `bench/cublas_bench.cu`）：

| 形状 (M=N=K) | dtype | cuBLAS 有效算力 (TFLOPS) | 相对实测峰值 (118.4 TFLOPS) |
|---|---|---|---|
| 4096 | BF16 | 87.10 | 73.6% |
| 2048 | BF16 | 87.16 | 73.6% |
| 1024 | BF16 | 65.31 | 55.2% |
| 8192 | BF16 | **96.15** | 81.2% |

> **解读**：cuBLAS 在本机 BF16 大形状下做到峰值约 **81%**（8192³）。这是"追平线"——你的内核相对 cuBLAS ≥100% 即追平，相对峰值 ≥81% 即超过 cuBLAS。

```json
// baselines/cublas.json 示例
{ "shape": "4096x4096x4096", "dtype": "bf16", "tflops": 87.10, "date": "2026-10-21" }
```

### 0.5 工具链（已就绪）

- `nvcc 13.0`，`gcc 13.3`（C++20）
- Nsight Compute `ncu` 2025.3、Nsight Systems `nsys` 2025.3
- 参考实现：cuBLAS / cuBLASLt（`torch.matmul` 或 CUTLASS 示例）

---

## 1. 总体迭代流程

每一轮迭代严格按以下四步循环，**任何优化改动都必须走完整圈**，禁止跳步：

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1 分析瓶颈   Step 2 制定方案   Step 3 实施   Step 4 验证 │
│  (Roofline+Ncu)  →  (映射优化措施) →  (保存+改动) → (性能+精度) │
│        ↑                                              │       │
│        └──────────── 通过后回到 Step 1  ←──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

**每轮迭代的必备产出物**（记录到第 6 节的迭代日志）：

1. 本轮瓶颈结论（来自 Step 1）
2. 采取的优化措施与预期（来自 Step 2）
3. 改动前后版本号（git commit hash）
4. 性能对比（TFLOPS / 峰值利用率 / 相对 cuBLAS，Step 4）
5. 精度对比（误差指标，Step 4）
6. 下一步瓶颈假设（回到 Step 1 时写下）

---

## 2. Step 1 — 分析瓶颈

**目的**：回答"当前 kernel 卡在哪"，并留下可复现的证据。本步不做任何代码改动。

### 2.0 参照实现侦察（必做，先于一切分析）

在分析自己的 kernel 之前，先采集最优参照（如厂商库）在**同一负载**下实际使用的 kernel 信息：

```bash
nsys profile -o ref ./参照程序
nsys stats --report cuda_gpu_kern_sum ref.nsys-rep
```

- 参照 kernel 的名字通常直接编码了 tile / stage / 指令配置，是最便宜的「标准答案」；
- 若参照与自己使用**相同指令**而性能差距大，瓶颈在配置与调度，优先对比配置，而非怀疑硬件上限；
- 此步成本分钟级，必须在任何瓶颈假设之前完成。

### 2.1 Roofline：先分 compute-bound / memory-bound

计算算术强度（Arithmetic Intensity）：

```
AI (FLOP/byte) = 2 × M × N × K / 搬运的总字节数
```

转折点**用 §0.3 实测值代入，禁止用推测值**：

```
转折点 (FLOP/byte) = 峰值算力(实测 TFLOPS) × 1e12 / 内存带宽(实测 GB/s) × 1e9
```

本机 BF16 实测转折点：

```
118.37e12 / 255.6e9 ≈ 463 FLOP/byte
```

判定：

| 条件 | 结论 | 优化方向 |
|---|---|---|
| AI < 转折点 | **memory-bound** | 优先优化访存（Step 2 → 3.1 的访存类） |
| AI > 转折点 | **compute-bound** | 优先优化计算（Step 2 → 3.1 的计算类） |

> **本机注意**：GB10 带宽远低于数据中心 H100（约 1/6），memory-bound 区间比想象中大得多。小 K / 大 M×N 场景尤其容易落到 memory-bound，不要想当然假设是 compute-bound。

> **推导指标的有效性**：由间接测量推导的指标（如 `TFLOPS ÷ 峰值带宽` 反推的 AI）依赖前提假设（如带宽饱和）；假设未经验证时，该指标只能作为界（上界/下界）使用，不得用于精确定位瓶颈。可用纯带宽 microbench 的实测吞吐对照，判断带宽是否真的饱和。

### 2.2 Nsight Compute：用硬件 counter 精确定位

```bash
# 先看 speed-of-light（一屏判定算力/带宽/占用率谁拖后腿）
ncu --set speed-of-light ./your_gemm

# 全量采集，用于深挖 stall 原因
ncu --set full -o report_iterN ./your_gemm

# 列出本机可用 metric（名字以实际输出为准）
ncu --query-metrics | grep -iE "tensor|dram|bank_conflict|stall|local_mem|warps_active"
```

必须采集并记录以下五类：

1. **张量核利用率** `sm__pipe_tensor_op*...pct_of_peak_sustained_active`
   → 决定是否 compute-bound，数值 < 60% 说明算力没喂满。
2. **DRAM 带宽利用率** `dram__throughput.avg.pct_of_peak_sustained_elapsed`
   → 决定是否 memory-bound，数值 > 85% 说明带宽是瓶颈。
3. **warp stall 原因**（`--set full` 的 *Warp State Statistics*，或 `smsp__average_warp_latency*`）：
   - `long_scoreboard` → 等 TMA/global load，访存延迟没藏住 → 加 stage / 加深 pipeline
   - `barrier` → 等 mbarrier 同步 → 流水线不均衡
   - `lg_throttle` / `drain` → 指令发射瓶颈 → 寄存器/依赖问题
   - `wait` → 固定延迟（mma 依赖链）
   - `not_selected` / `selected` → 占用率（occupancy）不足 / 正常发射
4. **shared memory bank conflict** `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld/st`
   → 非零则调 swizzle。
5. **寄存器溢出** `l1tex__t_sectors_pipe_lsu_mem_local_op_ld/st`
   → 非零则说明有 local memory spill，必须优先修。

> **本机注意**：消费级 GPU 的 ncu counter 权限可能被锁。若 `ncu --set speed-of-light` 报 `ERR_NVGPUCTRPERM`，需要 root 运行或设置 `NVreg_RestrictProfilingToAdminUsers=0`。若确实无法采到 counter，改用 2.4 的消融法替代。

### 2.3 编译期静态检查（零成本，每次编译都做）

```bash
nvcc -arch=sm_120 -Xptxas -v -c your_kernel.cu -o /dev/null
```

记录三项：

- **寄存器数** `registers used = N`：需满足 occupancy 预算。每 SM 65536 寄存器，若目标 4 wave/CTA，每线程不能超过 `65536 / 总线程数`。
- **`spilled registers`**：一旦出现即为性能断崖，本轮最高优先级修复。
- **local memory 使用**：同上。

### 2.4 兜底：消融法（无 profiler 时 / 想量化每个优化贡献时）

按第 7 节附录 B 的消融表，从朴素 kernel 逐项叠加优化，记录每一步两个百分比的增量。**每个增量的来源，就是该优化消掉的那一类瓶颈。**

### 2.5 Step 1 产出

在迭代日志中写下结论，格式：

```
【瓶颈结论】
- Roofline 判定: compute-bound / memory-bound (AI = xxx FLOP/byte)
- 张量核利用率: xx%     DRAM 带宽利用率: xx%
- 主要 stall 原因: long_scoreboard / barrier / ...
- bank conflict: 有/无    spill: 有/无    寄存器: N (预算 M)
- 一句话结论: 当前卡在 ________
```

---

## 3. Step 2 — 制定优化方案

**目的**：根据 Step 1 的瓶颈，选定**唯一**一个优化措施，并写下预期收益。禁止在本步动手改代码。

### 3.1 瓶颈 → 措施映射表

| Step 1 结论 | 候选优化措施 |
|---|---|
| memory-bound / `long_scoreboard` | 增加 pipeline stage、TMA multicast、swizzle 消 bank conflict、block swizzle（L2 复用）、调整 tile 提高算术强度 |
| compute-bound / 张量核利用率低 | 增加 mma 指令密度、减少同步（`barrier` stall）、提升 occupancy、消除 spill、stage 合并（减少 `wait`） |
| `barrier` stall | 调整 stage 数、均衡 TMA 与计算节奏、异步 barrier（`arrive_and_expect_tx`） |
| 寄存器溢出 | 减小 tile、调整累加器布局、拆分 epilogue |
| bank conflict | 改 swizzle mode（32/64/128B）、调整 swizzle atom |

### 3.2 纪律

1. **一次只改一个变量**。若同时改多个，无法归因。
2. 每项优化要写**预期收益**（例："预期峰值利用率 +10%，因为消除 long_scoreboard 等待"）。
3. 写下**可回滚的判断依据**（例："若两个百分比均未提升或精度劣化，回滚到 commit X"）。
4. **设计空间完整性**：对多维设计空间（tile 的各维度、stage 数、布局等）做扫描时，必须显式列出全部维度并声明本次覆盖的子空间；「X 不是瓶颈」类结论只能在已覆盖的子空间内声明。

### 3.3 Step 2 产出

```
【优化方案】
- 措施: 增加 pipeline stage 从 4 → 6
- 依据: Step 1 显示 long_scoreboard 占 stall 40%
- 预期: 峰值利用率 72% → 82%
- 回滚条件: 两个百分比不升或精度超容差
```

---

## 4. Step 3 — 保存版本并实施优化

**目的**：保证任何时刻都能精确回到上一个已知良好（known-good）状态。

### 4.1 版本管理（强制）

1. 当前目录纳入 git 管理（若尚未初始化则 `git init`）。
2. **每轮开始前先 commit** 当前已知良好版本，commit message 带迭代号和基线指标：

```bash
git add -A
git commit -m "iterN baseline: fp8_gemm peak72% cublas98% @ 4096x4096x4096"
```

3. 实施优化后，再 commit，message 说明改动与结果：

```bash
git commit -m "iterN: +stage 4->6, peak 72%->81%, cublas 98%->112%, err 1.2e-3"
```

4. 关键版本打 tag：`git tag iterN-known-good`。

### 4.2 实施纪律

- 只做 Step 2 选定的**那一个**改动，禁止夹带其它"顺手"修改。
- 改动后先编译通过（含 `-Xptxas -v` 无新增 spill），再进入 Step 4。

---

## 5. Step 4 — 验证性能与精度

**目的**：用数据确认"更快了，且没算错"。两者缺一不可。

### 5.1 性能验证

按第 1 节的计时规范（预热 ≥5 次 + 多次取平均/中位数），记录**两个百分比**：

```
【性能】
- 形状: M×N×K
- 改动前: x.x TFLOPS (峰值 xx% / cuBLAS xx%)
- 改动后: y.y TFLOPS (峰值 yy% / cuBLAS yy%)
- 变化: +z% / 持平 / 回退
```

**噪声底线**：性能结论必须显著超过测量噪声（时钟漂移、run-to-run 方差）；差异小于噪声量级（经验值 ~3%）时判定为持平，需重复测量（≥3 次取中位数）后再下结论，禁止基于噪声级差异做采纳/否决。

### 5.2 精度验证

**精度参照 = 同精度的 cuBLAS 结果**（不是 FP32 真值）。即：BF16 GEMM 与「cuBLAS 的 BF16 GEMM（同样 BF16 输入、FP32 累加）」对拍。

> **为什么**：与 FP32 真值对比，测出的是「BF16 量化的固有损失」（本机实测 ~6.5%@K=512），跟你的 kernel 实现无关；与**同精度 cuBLAS** 对比，测出的才是「实现误差」——这才是正确性/回归该看的（只剩累加顺序差异，本机 naive 实测 ~1e-5）。

至少覆盖三类输入：

1. 常规形状（4096³、128 对齐）
2. 非对齐 / 奇数形状（M=4095, N=127, K=1023）
3. 极端小尺寸（M=1,2,4,8；N=1,2）与空组（分组 GEMM）

统一记录误差指标（**vs 同精度 cuBLAS**）：

```
【精度】参考: cuBLAS(同 dtype 输入, FP32 累加)
- max abs error / max relative error
- L2 相对误差  ||C - C_ref||_2 / ||C_ref||_2
```

容差（正确性判据，vs 同精度 cuBLAS）：

| 精度 | 相对误差容差（vs 同精度 cuBLAS） |
|---|---|
| FP32 / TF32 | < 1e-4 |
| BF16 | < 1e-3（本机 naive 实测 ~1e-5） |

> **量化损失**（vs FP32 真值）不作为正确性判据，仅作参考值记录（本机 BF16 实测 ~6.5%@K=512，随 K 增大而增大）。

### 5.3 判定与闭环

> **口径**：下文"性能"指 §0.1 的**两个百分比**（峰值利用率 + 相对 cuBLAS）。通常二者同向变化；若背离（一个升一个降），必须记录并解释，不能只看一个。

| 结果 | 动作 |
|---|---|
| 两个百分比提升 **且** 精度达标 | ✅ 通过，commit + tag，**回到 Step 1** 开始下一轮 |
| 性能提升 **但** 精度超容差 | ❌ 定位精度来源（量化/对齐/溢出），回滚或修复后重验 |
| 性能持平或回退 | ❌ 回滚到上一 known-good，重新分析瓶颈（Step 1） |
| 精度达标但性能无变化 | ❌ 记录"该优化在本机无效"，回滚，换下一候选措施 |

> **守则**：无论通过与否，**每轮都必须回到 Step 1 重新分析**，用新一轮的 roofline/ncu 数据确认瓶颈是否真的被消掉、是否转移到了别处。

---

## 6. 迭代日志模板（每轮填写）

```markdown
## iter N — 日期 YYYY-MM-DD

### Step 1 瓶颈
- Roofline: ____-bound (AI=__ FLOP/byte)
- 张量核 util: __%    DRAM util: __%
- 主要 stall: ______    bank conflict: __  spill: __   regs: __
- 结论: 卡在 ________

### Step 2 方案
- 措施: ________
- 预期: 峰值利用率 __% → __%
- 回滚条件: ________

### Step 3 版本
- baseline commit: ______ (峰值 __% / cuBLAS __%)
- change commit: ______

### Step 4 验证
- 形状 M×N×K = ______
- 性能: __ TFLOPS (峰值 __% / cuBLAS __%) → __ TFLOPS (峰值 __% / cuBLAS __%)
- 精度: max_rel_err=__, RMSE=__, cosine=__   (参考: ____)
- 判定: 通过 / 回滚 / 换方案

### 下一步假设
- 下一轮待验证的瓶颈: ________
```

---

## 7. 附录

### 附录 A — 基准测量（压测，非推测）

> 压测代码已存在于本仓库：`bench/bench_micro.cu`（峰值算力 + 带宽）、`bench/cublas_bench.cu`（cuBLAS baseline）。重新压测时直接复用并更新 §0.3 / §0.4。

1. **实际峰值算力**：写"纯计算" kernel，数据全部驻留寄存器，死循环发射 `mma.sync`（不读 smem/gmem），实测 TFLOPS 即本轮 100% 标尺。**不引用任何理论/推测值。**
2. **实际内存带宽**：写"纯访存" kernel，grid-stride float4 读，实测 GB/s 即本轮带宽标尺。
3. **cuBLAS baseline**：对每个首次评测的形状，用 cuBLAS/cuBLASLt 实测一次有效算力并缓存到 `baselines/cublas.json`，后续不再重测。
4. 校准要在**同一次运行、同时钟状态下**进行，与主 kernel 使用相同计时方法；时钟波动时用足够长的 kernel 平均。

### 附录 B — 消融基线表（Step 2 兜底与归因）

| 步骤 | 累计优化 | 利用率参考（相对实测峰值，非绝对 TFLOPS） |
|---|---|---|
| 0 | global 直接算（基线） | 极低 |
| 1 | + 分块 tile + 寄存器累加 | 20~40% |
| 2 | + smem 双缓冲 | 40~60% |
| 3 | + TMA 异步加载 | 60~75% |
| 4 | + 多 stage pipeline | 75~85% |
| 5 | + warp 特化 | 85~90% |
| 6 | + swizzle 消 bank conflict | +几个点 |
| 7 | + block swizzle（L2） | +几个点 |

### 附录 C — 术语速查

| 术语 | 含义 |
|---|---|
| AI (Arithmetic Intensity) | 每字节访存对应的 FLOP 数 |
| compute-bound / memory-bound | 受算力 / 带宽约束 |
| occupancy | SM 上活跃 warp 占理论最大 warp 的比例 |
| spill | 寄存器溢出到 local memory |
| swizzle | shared memory 的跨 bank 交错排布 |
| long_scoreboard | warp 因等待访存数据而 stall |

### 附录 D — 本机硬约束（优化时必须遵守）

- 编译目标 `-arch=sm_121`（本机 cc 12.1）。
- 每 SM shared memory 上限 **100 KB**，多 stage 时务必算清 `stage × (A_tile + B_tile) ≤ 100KB`。
- 张量核只能用 **`mma.sync`**：BF16 用 `m16n8k16`，TF32 用 `m16n8k8`；操作数必须经 `ldmatrix`/`ld.shared` 从 smem 搬入寄存器。
- **本机（sm_121）不支持 `tcgen05`/`TMEM`/`STSM`（数据中心 Blackwell 专属），也不支持 FP8 的 `kind::f8f6f4` mma.sync**——因此 FP8/FP4 张量核在本机不可用，且不能复现 DeepGEMM 的 WGMMA/UMMA"操作数直接从 smem 喂张量核"优化。
- 无 Hopper 的 `warpgroup_reg_alloc/dealloc`，寄存器按经典 occupancy 计算。
