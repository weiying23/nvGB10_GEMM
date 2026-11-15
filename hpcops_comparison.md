# GB10 GEMM vs 腾讯 hpc-ops GEMM 对比分析

> 源码分析基础：`https://github.com/Tencent/hpc-ops`（2026-11-15 clone，61 commits，MIT License）
> 本文档把与 hpc-ops 的全部对比（代码分析、编译技术、移植实验、硬件余量分析）整合为一处。

## 0. 概览

hpc-ops 是腾讯混元 AI Infra 团队的 LLM 推理算子库，瞄准 **H20（SM90/Hopper）和 SM100/SM103**，覆盖 Attention/MoE/GEMM/sampling/通信融合。GEMM 家族四个成员：

| GEMM | 精度 | 硬件 | 特殊点 |
|---|---|---|---|
| BF16×FP32 | 两 BF16 分解模拟 FP32 权重 | sm90 | MoE router / 稀疏场景，最高 3.22× 优于 cuBLAS FP32 |
| FP8 GroupGEMM | per-tensor / block-wise scale | sm90 | 专家并行 |
| FP8 Fused MoE | 低延迟 MoE 流水 | sm90 | TP/EP 形状 |
| Gated MLA GEMM | 混合精度 | **sm100** | 用了 tcgen05/TMEM/cluster |

## 1. 架构对比：同构 + 三层差异

两者的 GEMM **架构骨架同构**（TMA + mbarrier full/empty + 多 stage + swizzle + TMA store + 配置族分派），但 hpc-ops 站在我们不具备的三层硬件基础之上：

### 层 1：张量核指令（硬件代际差异，不可移植）

| | 我们（SM121） | hpc-ops（SM90/SM100） |
|---|---|---|
| smem→张量核 | `mma.sync m16n8k16`（同步，操作数必须过寄存器） | **WGMMA**（sm90：异步，操作数直接从 smem 读）；**tcgen05**（sm100：用 TMEM 做累加器） |
| 异步空档 | mma 期间 warp 干等（零余量） | wgmma 发射后 warp 立即返回（有空档可塞 epilogue） |
| 验证 | tcgen05/WGMMA 在 sm_121 编译失败（本项目实测） | — |

### 层 2：warp 组织（硬件支持差异）

| | 我们 | hpc-ops |
|---|---|---|
| 结构 | 8 个同质 warp（thread 0 兼生产） | **1 个 load warp group + N 个 math warp group** |
| 寄存器 | `__launch_bounds__(256, 1)` 统一分配 | `warpgroup_reg_alloc<168>()` / `warpgroup_reg_dealloc<24>()`——producer 只占 24 寄存器，consumer 168 |
| 效果 | warp 特化实测 -–%（证伪） | producer 几乎免费，完全解耦 |

### 层 3：硬件余量（根本差异）

| 余量维度 | H20 | GB10 |
|---|---|---|
| DRAM 带宽 | ≈4000 GB/s（HBM3e，16 倍富余） | **255.6 GB/s**（LPDDR5X，刚够） |
| 张量核异步 | WGMMA 异步（有计算空档） | mma.sync 同步（零空档） |
| warp 特化 | 硬件 reg 重分配 | 无此指令 |
| L2 大小 | 60 MB | **24 MB** |
| TMA multicast | 有（cluster 共享装载） | 无 |
| PDL | 有（kernel 间重叠） | 有但未用（场景不匹配） |

**核心结论**：hpc-ops 的 persistent 能用，不是因为代码更好，而是 SM90+HBM 在四个维度都有余量——L2 失焦造成的额外 DRAM 流量、异步空档不够大、producer 不够快，全被硬件兜住。GB10 四个维度全是零余量。

## 2. 编译技术对比

两者最终都只用 nvcc（无独立编译器/JIT/LLVM/TVM）。区别在 nvcc 上面垫了什么：

| | 我们 | hpc-ops |
|---|---|---|
| 抽象层 | 纯手写 CUDA C++ + inline PTX | CUTLASS 4.x + **CuTe 布局代数** |
| smem swizzle | 手推 XOR 公式 + dump 探针验证 12288 元素 | `GMMA::Layout_K_SW128_Atom` → 编译期类型推导 |
| TMA box | 手调 `{64,128}` + cuTensorMapEncodeTiled | `make_tma_copy(SM90_TMA_LOAD{}, X, layout)` → 自动推导 |
| mma 地址 | 手写 `warp_m*WM + mi*16 + ar8 + ...` | `thr_mma.partition_A(sW)` → 编译期算偏移 |
| epilogue | 手写 `crow = lane/4, ccol = (lane%4)*2` + TMA store | `partition_fragment_C` + STSM 原语 → TMA store |
| tile 几何检查 | `static_assert`（v4 教训后加） | 类型不匹配 → 编译报错（免费） |
| 代码量 | ≈350 行 kernel（手写全部底层） | 557 行（但功能更多，含 split-K/FP8 分解） |

**为什么我们不用 CuTe**：SM121 不是 CuTe 标准目标（主打 sm80/90/100），mma.sync m16n8k16 是老指令，CuTe 抽象反而增加复杂度。且 v8-v11 的三个 race（fence/barrier/双发射）全是 CuTe **不会帮你处理**的硬件怪癖。

## 3. 逐项技术对比

### 两边都做了的（殊途同归）

| 优化 | hpc-ops | 我们 |
|---|---|---|
| TMA + mbarrier | CuTe 封装 | 手写 PTX |
| 多 stage 流水 | kStage 参数化 | NUM_STAGES=2（3/4 实测无效） |
| TMA store epilogue | STSM→smem→TMA store | st.shared→smem→TMA store |
| block swizzle | kBlockSwizzle=4 + FastDivmod | GROUP_M=8 |
| 配置族 + 分派 | LaunchCfg 模板 + entry.cc picker | Cfg 模板 + use_small 一行 if |
| 越界自动填零 | TMA box | TMA box |
| dynamic smem opt-in | cudaFuncSetAttribute | 同（同一个坑也踩过） |
| persistent | grid=min(SM, tiles)，iblock += gridDim.x | 实测证伪（见下） |

### hpc-ops 有我们没有的（6 项，按价值排序）

| # | 技术 | 说明 | 可移植性 |
|---|---|---|---|
| 1 | **WGMMA / tcgen05** | smem→张量核直读 + TMEM 累加器 | ❌ SM121 无此指令 |
| 2 | **warp group 特化 + reg 重分配** | producer 24 regs / consumer 168 regs | ❌ SM121 无 warpgroup_reg_alloc |
| 3 | **PDL**（Programmatic Dependent Launch） | kernel 间重叠，MoE 连续 GEMM 场景 | 可用但场景不匹配 |
| 4 | **FP32 = 两 BF16 分解** | w_high + w_low×scale 模拟 FP32，跑 BF16 速度 | ✅ 可移植（精度策略） |
| 5 | **CuTe 布局代数** | 编译期类型系统推导 smem/TMA/mma 布局 | 部分可移植（但 SM121 非标准目标） |
| 6 | **TMA multicast / cluster** | 一次 TMA 装载服务多个 SM | ❌ SM121 无 cluster multicast |

### 我们有 hpc-ops 没展示的

| 技术 | 说明 |
|---|---|
| **SM121 mbarrier 怪癖** | arrive 隐式 release 不排序 LDSM 读 → 必须显式 fence.acq_rel.cta（v8 踩坑定位） |
| **L2 步长病理** | 2 的幂全局步长 → L2 切片集中×B 流量饱和（三层探针定位机制链） |
| **两端 roofline 消融** | 512³（填充×效率×写回三项叠加）vs 8192³（计算核到墙+写回残余）完整分解 |

## 4. persistent 移植实验（我们做了，三层证伪）

### 实验背景

hpc-ops 的 persistent grid = min(SM 数, tile 数)，每 block 以 stride = gridDim.x 领 tile。我们试图移植此模式到 GB10 实现 epilogue 与下一 tile 计算的重叠。

### 第 1 层：persistent 骨架 + stride-48 分配

```
grid = 48（= SM 数）
block b 处理 tile: b, b+48, b+96, ...（stride-48 递增）
```

| 尺寸 | v12（非 persistent） | persistent（stride-48） |
|---|---|---|
| 2048³ | 77.55 | 72.77（-6.2%） |
| 4096³ | 88.30 | 80.05（-9.3%） |
| 8192³ | 95.64 | **71.85（-24.9%）** |

**实现踩坑**：persistent 跨 tile mbarrier 相位必须**全 stage 追赶**——每 tile 结束时生产者比 empty_bar 少等一次（最后 K 块不发射→不等），非首 tile 的 prologue 不补等所有 stage 的 empty 相位就死锁。

### 第 2 层：swizzle-aware 连续分块（hpc-ops 思想移植）

```
每 block 领一段连续的 swizzle 序 tile：
block 0 → tile 0..21，block 1 → tile 22..43，...
```

| 尺寸 | stride-48 | 连续分块 |
|---|---|---|
| 2048³ | 72.4 | 77.1（改善） |
| 4096³ | 80.2 | **28.2（-68%）** |
| 8192³ | 71.9 | **23.4（-76%）** |

**连续分块比 stride-48 更灾难**——48 个 block 同时从矩阵 48 个不同区域开始，L2 里 48 套不同 B tile 互相踩踏。

### 第 3 层：根因定稿

```
非 persistent（v12）：
  GPU 硬件 block 调度器把 swizzle 序连续的 block 同时分驻 SM
  → 48 个 block 天然时序对齐在同一 GROUP_M 组
  → B tile L2 复用窗口最大化
  ↑ 这是"免费的时序对齐"，persistent 摧毁了它

persistent（任何分配模式）：
  block 间速度不同步 → 第二轮起各 block 的 tile 不在同一 swizzle 组
  → L2 复用窗口失焦 → DRAM 流量翻倍 → 255.6 GB/s 瓶颈爆掉
```

**最终结论**：persistent 与 GROUP_M swizzle 的 L2 共享机制**架构性冲突**——不是 tile 分配模式能解决的。hpc-ops 在 sm90 上能用 persistent 是因为 HBM 带宽（>3TB/s）有充足余量吸收 L2 失焦 + warp group 特化让 block 内完全解耦。**GB10 的 255.6 GB/s + sm121 的 4 调度器没有任何余量**。探针源码保留于 `bench/probe_persist.cu`。

## 5. 性能格局对比

### 我们的 v12 在 GB10 上

| 尺寸 | v12 | cuBLAS | 比值 |
|---|---|---|---|
| 512³ | 26.9 | 38.2 | 70.5% |
| 1024³ | 58.0 | 65.3 | 88.7% |
| 2048³ | 78.6 | 95.1 | 82.6% |
| **4096³** | **89.8** | 83.8 | **107.2%** |
| 8192³ | 94.7 | 97.5 | 97.1% |

### hpc-ops 在 H20 上（README 声称）

| 操作 | 基线 | 声称结果 |
|---|---|---|
| BF16×FP32 GEMM | cuBLAS FP32/TF32 | 最高 **3.22×**（Router GEMM 形状） |
| FP8 GroupGEMM | DeepGEMM | 最高 1.1× prefill / 1.88× decode |
| Fused FP8 MoE | vLLM/SGLang | 最高 1.6× TP / 1.5× EP |

**注意**：hpc-ops 的 3.22× 是 BF16×FP32 对比 cuBLAS FP32（精度分解优势），不是同精度对比。同精度 BF16 GEMM 的差距两者应更接近（都是 TMA+mbarrier 流水）。

## 6. 可移植到我们机器上的技术清单

| 技术 | 预期收益 | 工程量 | 已做？ |
|---|---|---|---|
| FP32 = 两 BF16 分解 | 需 FP32 权重的场景 2-3× | 中（双累加器+scale epilogue） | ❌ 未做 |
| PDL | MoE 连续 kernel 间重叠 | 低（launch attribute） | ❌ 未做（场景不匹配） |
| split-K（带 atomic flag 异步归约） | 小 M×N 大 K | 中 | ❌ 已算术判死（本机带宽不够） |
| persistent | epilogue 重叠 + 尾波消除 | 高 | ❌ 三层证伪 |
| kBlockSwizzle=4（更小组） | 减轻 L2 失焦 | 低（改常量） | 未独立测试（被 persistent 证伪覆盖） |
| CuTe 布局代数 | 减少布局 bug | 高（学习成本） | ❌ 不适用（SM121 非标准目标） |

## 7. 总结

**hpc-ops 和我们的 GEMM 架构骨架相同**（TMA + mbarrier + 多 stage + swizzle + TMA store + 配置族），性能差异来自三层：

1. **硬件代际**（WGMMA/tcgen05/warpgroup reg/multicast）——SM121 不具备，不可移植
2. **硬件余量**（HBM 4TB/s vs LPDDR5X 256GB/s）——决定 persistent 等激进优化能否存活
3. **编译抽象**（CuTe 布局代数 vs 手写 PTX）——影响开发效率不影响性能上限

**在 GB10 的硬件约束下，v12 已接近 sm121 + mma.sync + TMA 路线的性能天花板**（4096³ 反超 cuBLAS 107%，计算核已到探针墙的 95%）。下一个量级跃迁在 FP8（峰值 235.79 = BF16 的 2 倍，指令已验证可用）。

