# GB10（DGX Spark）硬件全景 — 组织架构与实测参数

> 目标：一份文档讲清本机硬件的组织方式（SM/warp 如何构成、cache/memory 层次）与全部关键参数。
> **数据来源标注**：`[实测]` = 本项目微基准测量；`[推导]` = 由实测数字算出；`[官方]` = NVIDIA 规格或驱动报告；`[推断]` = 结构推断（ncu 计数器被权限锁定，无法直接验证微架构细节）。
> 测量环境：驱动 580.173.02 / CUDA 13.0 / nvcc -arch=sm_121。

---

## 1. 芯片级：CPU+GPU 统一内存 SoC

GB10 不是独立显卡，是 **Grace-Blackwell 风格的 SoC**：ARM CPU 与 GPU 共享同一块 LPDDR5X（无独立显存，`cudaDeviceProp::integrated` 语义、统一寻址）。

```
┌─────────────────────────────────────────────────────────────────┐
│                        GB10 SoC                                  │
│                                                                  │
│   ┌──────────────┐  NVLink-C2C  ┌────────────────────────────┐   │
│   │  ARM CPU 核   │◄════════════►│        GPU（本档主角）        │   │
│   │  (Grace 系)   │   [官方]     │                            │   │
│   └──────┬───────┘              │   ┌────┐ ┌────┐   ┌────┐    │   │
│          │                      │   │SM 0│ │SM 1│ … │SM47│ 48个│   │
│          │                      │   └──┬─┘ └──┬─┘   └──┬─┘    │   │
│          │                      │      └──────┼─────────┘      │   │
│          │                      └─────────────┼────────────────┘   │
│          │                                    │ 交叉开关/NoC         │
│   ┌──────┴────────────────────────────────────┴───────┐           │
│   │              LPDDR5X 统一内存 128 GB               │           │
│   │   8533 MHz 有效 × 256-bit 总线 = 273 GB/s 理论 [官方] │           │
│   │   GPU 可见 121.69 GB（其余为系统保留）[驱动报告]       │           │
│   └───────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

**要点**：
- **CPU↔GPU 互连 = NVLink-C2C**（Grace 风格一致性互连，官方规格；本机未单独实测其带宽）。注意：**没有 GPU↔GPU 的 NVLink**（不能多卡扩展）。
- 统一内存 = CPU/GPU 共享物理内存。带宽 273 GB/s 是**整个 SoC 共享的单向峰值**——比数据中心卡（HBM，数 TB/s）低一个数量级，这决定了本机 GEMM 的 roofline 转折点极高（463 FLOP/byte，见 §6）。
- 实测读带宽 **253.5~255.6 GB/s**（理论值的 93%）`[实测, bench/bench_micro.cu]`。

---

## 2. GPU 核心：48 个 SM

```
GPU（cc 12.1，消费级 Blackwell；第五代 Tensor Core；无 GPU 间 NVLink）
│
├── SM 0  ── L1/smem 100KB ── 寄存器堆 64K×32bit
├── SM 1  ── 同上
├── ...
├── SM 47 ── 同上
│
└── 全部 SM 通过交叉开关共享：
    ├── L2 cache：24 MB（分区/切片组成，跨 SM 一致）
    └── 内存控制器 → LPDDR5X
```

### 核心数量

| 项目 | 数量 | 来源与说明 |
|---|---|---|
| SM | **48** | [驱动报告] |
| CUDA core | 外部口径 **6144**（128/SM） | [未验证] 与本机实测不一致，见下 |
| FP32 FMA 数据通路 | **实测 64 FMA/clk/SM = 全 GPU 3072 条** | [实测：16 独立链 FFMA 满载 15.23 TFLOPS @ 实抓 2411 MHz] |
| Tensor core | 官方未公布；按消费级 Blackwell SM 结构**推断 4 个/SM = 192 个** | [推断] |

> ⚠️ **CUDA core 口径分歧**：外部资料（含评价方）称 6144（=48×128），但本机实测 FP32 FFMA 吞吐只有 64 FMA/clk/SM（若为 128 条 lane/SM，2.4 GHz 下应测得 ~29.5 TFLOPS，实测仅 15.23 且时钟确认为 2411 MHz 满载、16 独立链排除了依赖链限制）。可能官方计数含 FP32/INT32 复用管线或其他口径。**本项目的 roofline 一律以实测 64 FMA/clk/SM 为准。**

| 参数 | 值 | 来源 |
|---|---|---|
| SM 数 | **48** | [驱动报告] |
| 计算能力 | 12.1（SM121，消费级 Blackwell） | [驱动报告] |
| L2 cache | **24 MB**（可驻留上限 18 MB） | [驱动报告] |
| 每 SM L1/smem | **100 KB**（每 block optin 上限 99 KB=101,376B，默认 48 KB） | [驱动报告] |
| 每 SM 寄存器 | **65,536 × 32-bit**（=256 KB） | [驱动报告] |
| 每 SM 驻留线程上限 | **1,536**（= 48 warps） | [驱动报告] |
| 每 SM 驻留 block 上限 | 24 | [驱动报告] |
| 每 block 线程上限 | 1,024 | [驱动报告] |
| SM 时钟 | 空闲 208 MHz → 应用时钟 2418 MHz → 张量满载持续 **~2405 MHz**（最大加速 3003 MHz） | [nvidia-smi 轮询实测] |
| 协作/集群 launch、并发 kernel、流优先级 | 均支持 | [驱动报告] |

**为什么 L2 只有 24 MB 很重要**：4096³ 的 A+B 共 64 MB，装不进 L2——这直接迫使本项目发明 block swizzle（GROUP_M=8）来控制 L2 复用窗口（v7 实测 8192³ +43%）。数据中心卡 L2 有 50~126 MB，没有这个问题。

---

## 3. SM 内部结构

ncu 计数器被权限锁定（`ERR_NVGPUCTRPERM`），子分区细节无法直接验证。以下结构图以**实测吞吐为锚**：

```
                     一个 SM（SM121）
┌───────────────────────────────────────────────────────────┐
│                                                           │
│   指令缓存 / 常量缓存                寄存器堆 256 KB       │
│         │                           (65,536 × 32-bit)    │
│         ▼                                  ▲             │
│   ┌─────────────┐   发射     ┌─────────────┴───────────┐ │
│   │ warp 调度器  │ ────────▶ │  4 个子分区（sub-core）   │ │
│   │ (推测 4 个)  │           │  [推断：48 warps/SM ÷ 12] │ │
│   └─────────────┘           │                          │ │
│                             │  每 子 分 区：             │ │
│                             │  · FP32 FMA 阵列          │ │
│                             │    (全 SM 合计 64 FMA/clk  │ │
│                             │     [实测]，张量核为    │ │
│                             │     其 8 倍吞吐)           │ │
│                             │  · 第五代张量核（5th-gen；指令面仅 mma.sync 家族，见 §6）            │ │
│                             │    (指令面仅 mma.sync 家族) │ │
│                             │  · LSU（ld/ldsm/st）       │ │
│                             │  · SFU（超越函数等）        │ │
│                             └──────────────────────────┘ │
│                                                           │
│   ┌─────────────────────────────────────────────────┐    │
│   │  L1 Data / Shared Memory 统一块：100 KB           │    │
│   │  · smem：可编程划分（block optin ≤99 KB）          │    │
│   │  · ldsm 聚合读带宽：全 GPU 130 TB/s [实测]        │    │
│   └─────────────────────────────────────────────────┘    │
│                                                           │
│   异步引擎：                                              │
│   · TMA（cp.async.bulk.tensor，由单线程发射整 tile 搬运） │
│   · cp.async（16B 粒度异步拷贝，Ampere 风格）             │
│   · mbarrier（full/empty 障栅，见 §7 怪癖）               │
└───────────────────────────────────────────────────────────┘
```

### 每 SM 每周期吞吐 `[实测推导，时钟取 2405 MHz]`

| 运算 | FLOP/clk/SM | 全 GPU | 测量方式 |
|---|---|---|---|
| **FP8** `mma.sync.m16n8k32`（e4m3/e5m2） | **2048**（=1024 FMA） | **235.79 TFLOPS** | 8 独立累加器 ILP 压满 [实测] |
| BF16 `mma.sync.m16n8k16` | **1024**（=512 FMA） | **118.37 TFLOPS** | 8 独立累加器 ILP 压满 [实测] |
| TF32 `mma.sync.m16n8k8` | **512** | **58.95 TFLOPS** | 同上 [实测] |
| FP32 FFMA（CUDA core） | **128**（=64 FMA） | **15.23 TFLOPS** | PTX fma 16 独立链 @ 实抓 2411 MHz [实测] |
| FP64 | ~4 | ~0.23 TFLOPS | 官方比值 64:1 [推导，未实测] |
| smem → 寄存器（ldsm） | — | 130 TB/s 聚合 | v5 消融探针 [实测] |

**说明**：
- 上表全部是 **mma.sync 指令路径的微基准结果**，不是官方营销峰值。官方仅公布 FP4 稀疏口径（约 1 PFLOP/s），按 8× 换算 BF16 稠密理论值 ≈ 125 TFLOPS，与实测 118.37 吻合（94.7%）。
- FP32 的 4 独立链测法只得 13.76（依赖链限制），16 独立链才是通路饱和值 15.23。
- **结构结论**：张量核 BF16 路径 = FP32 数据通路的 **8 倍**吞吐（1024 vs 128 FLOP/clk/SM），FP8 再翻倍（**16 倍**，2048）——任何没上 `mma.sync` 的 kernel 最多只能拿到 1/8 算力（本项目 v1→v2 的 3.4× 跳变即源于此）；而 FP8 相对 BF16 有整整 2× 的免费加成空间（若精度允许）。

---

## 4. Warp 执行模型（SIMT）

```
Grid（一维可达 2^31-1 个 block）
 │
 ├── Block #0 ──┬── warp 0  (thread 0-31)   ← 调度/执行的基本单位
 │              ├── warp 1  (thread 32-63)      32 线程锁步（SIMT）
 │              ├── ...
 │              └── warp N                      每 warp 有自己的 PC
 ├── Block #1 ──┬── ...
 │              └── ...
 └── ...

一个 warp 的 32 条 lane 同时执行同一条指令（不同数据）。
if (lane < 4) A(); else B();   →  warp 分歧：两分支串行执行，
                                分歧期间一半 lane 空转。
```

**本项目的 occupancy 实例**（v7/v8 配置）——三个资源同时约束：

| 资源 | 每块需求 | SM 容量 | 限制 |
|---|---|---|---|
| 寄存器 | 198 × 256 线程 = 50,688 | 65,536 | **2 块需 101,376 > 65,536 → 只能 1 block/SM** |
| smem | 72 KB（3-stage） | 100 KB | 同样只能 1 block/SM |
| 线程 | 256（8 warps） | 1,536（48 warps） | 不限制 |

→ **8/48 warps 驻留 = 16.7% occupancy**。这是刻意选择：用 128 个累加器/线程的极高 ILP 换低 occupancy（v7 教训：1 block/SM 下 `__syncthreads` 让整个 SM 停摆 -33%，才有了 v8 的 mbarrier 流水）。

---

## 5. 存储层次（全部实测）

```
            容量                 延迟            带宽
─────────────────────────────────────────────────────────
寄存器      256 KB/SM            ~0              —
   ▲ 由 ldsm/ld 装载
L1/smem     100 KB/SM            —               130 TB/s 聚合 (ldsm)
   ▲ 由 TMA / cp.async 装载
L2          24 MB（全 GPU）      ~143 ns [实测]   8~12 TB/s 读 [实测]
   ▲
DRAM        128 GB (LPDDR5X)     ~400 ns [实测]   253~256 GB/s [实测]
   （统一内存，与 CPU 共享；理论 273 GB/s）
─────────────────────────────────────────────────────────
```

| 层级 | 容量 | 负载延迟 `[实测：单线程指针追逐]` | 读带宽 `[实测]` |
|---|---|---|---|
| 寄存器 | 256 KB/SM（64K×32b） | —（指令操作数） | — |
| L1 / smem | 100 KB/SM | —（约几十周期量级，未单独标定） | ldsm 聚合 **130 TB/s** |
| L2 | 24 MB | **~143 ns**（8MB buffer 追逐） | **8.0~12.0 TB/s**（≤L2 容量 buffer 反复读） |
| DRAM | 128 GB 统一 | **~400 ns**（512MB buffer 追逐） | **253.5~255.6 GB/s** |

**层级间比例（记住这三个数就够）**：
- L2 带宽 ≈ DRAM 的 **40~47 倍** → tile 数据尽量从 L2 复用（swizzle 的意义）
- L2 延迟 ≈ DRAM 的 1/2.8 → 都远慢于计算（一条 mma ≈ 1~2 ns），必须靠多级流水掩盖
- smem 带宽 ≈ L2 的 11~16 倍 → 计算的操作数必须全部驻 smem/寄存器，绝不直接从 L2/DRAM 喂 mma

**带宽与算力的匹配（roofline 转折点）**：

```
118.37 TFLOPS ÷ 255.6 GB/s ≈ 463 FLOP/byte
```

单个 128×256 tile 的算术强度只有 BM×BN/(BM+BN) = 85 FLOP/byte（< 463）——**单靠 block 内复用到不了 compute-bound**，必须靠 L2 层的跨 block 复用（B tile 被 32 个 m-block 重复使用）把有效 AI 抬过 463。这就是大尺寸（≥4096³）才能打满算力、而 512³ 打不满的硬件根源。

---

## 6. 指令集与异步能力（本项目逐一验证过）

### 有（可用）
| 能力 | 说明 | 验证方式 |
|---|---|---|
| `mma.sync.m16n8k16` BF16/FP16 | 张量核主力指令形态；TF32 是 m16n8k8 | v2 起全版本使用 |
| **`mma.sync.m16n8k32` FP8（e4m3/e5m2）** | **实测 235.79 TFLOPS = BF16 的 2 倍**；精度换吞吐的空间 | 编译探针（sm_121 通过）+ 微基准实测 |
| `ldmatrix`（x4 / x2.trans） | smem→寄存器，专为 mma 供操作数 | v2 dump 逐位验证布局 |
| `cp.async`（16B） | Ampere 风格异步拷贝 | v3~v7 使用 |
| TMA `cp.async.bulk.tensor.2d` | 一条指令搬整个 tile，需 `-lcuda` + `__grid_constant__ CUtensorMap`；越界自动补 0 | v8 使用 |
| `mbarrier` + `try_wait.parity` | full/empty 障栅（**但有 §7 怪癖**） | v8 使用 |
| 协作 launch / 集群 launch | persistent kernel 的前提 | 驱动报告 |

### 没有（ptxas 编译探针确认）
| 能力 | 证据 | 影响 |
|---|---|---|
| **tcgen05 / TMEM** | ptxas 报错 `Instruction 'tcgen05.alloc' not supported on .target 'sm_121'` | 不能用数据中心 Blackwell 的第五代张量核编程模型 |
| **WGMMA** | sm_90a 专属指令族，sm_121 编译失败 | 不能用 Hopper 异步张量 warp-group mma |
| FP4（e2m1）mma.sync | **未验证**——所测 PTX 语法在 sm_120（RTX 50 系）上同样编译失败，说明语法未写对，无法据此下结论 | 若存在则还有 2× 空间，待找到正确语法后重测 |

→ **本芯片的 GEMM 指令面 = Ampere 风格 mma.sync 流水**（张量核硬件是第五代，但只暴露 mma.sync 指令模型）。cuBLAS 同样受限：4096³ 用的就是 `cutlass_80_tensorop_bf16_s16816gemm_128x256_32x3`（sm80 kernel，nsys 抓包 [实测]），跑出 87.1 TFLOPS = BF16 峰值的 74%。**若走 FP8 路线，峰值可达 235.79 TFLOPS（本项目尚未做 FP8 GEMM）。**

---

## 7. 硬件行为怪癖（本项目踩坑实测，写代码前必读）

1. **`mbarrier.arrive` 的隐式 release 不排序先前的 LDSM 读**——arrive 作为 LSU 原子可越过未返回的 ldmatrix 数据发射。后果：consumer leader arrive 空 barrier 后，producer 发的 TMA 可能覆写尚未读完的 stage（间歇性数据竞争，4096³ 约 30~50% 概率出错）。**修复：arrive 前显式 `fence.acq_rel.cta`**（v8 实测定位，代价 -4%）。`__syncwarp()` 无效、256 线程各自 arrive 也无效。
2. **`cp.async.mbarrier.arrive` 变体会死锁**——只有 TMA 的 `arrive.expect_tx` + `complete_tx::bytes` 模式可靠。
3. **racecheck 对 mbarrier/TMA 竞争完全失明**（0 hazards 报告但实际有 race）——只能靠消融实验矩阵定位。
4. **ncu 硬件计数器被权限锁定**（`ERR_NVGPUCTRPERM`）——一切微架构分析只能靠自建探针。
5. **满载时钟约 2.4 GHz 持续稳定**（应用时钟 2418），不会降频到 1.2 GHz——微基准测量可放心（本项目多次复测一致）。

---

## 8. 把参数映射到 GEMM（本项目 v0→v8 的硬件解释）

```
DRAM 255.6 GB/s ──┐
                  ├─ roofline 转折点 463 → 必须 tile 化 + L2 复用（v1, v7 swizzle）
118.37 TFLOPS ───┘

L2 24 MB（小） ──────→ 4096³ A+B=64MB 装不下 → swizzle GROUP_M=8 控制 L2 窗口
smem 100 KB ─────────→ 128×256×32 tile 最多 4-stage（96KB），5-stage 不可能
寄存器 64K/SM ───────→ 198×256=50.7K → 1 block/SM → __syncthreads 全 SM 停摆
                        → v8 用 mbarrier full/empty + TMA 消除全部 __syncthreads
张量 1024 FLOP/clk/SM → 64 mma/warp 的 ILP 才能压满（128 累加器摊薄 ldsm 延迟）
L2 延迟 143ns ────────→ TMA 预取 2 级 lookahead（3.2μs）已足够掩盖（4-stage 实测 +1.6% 无效）
```

| 版本轨迹 | 利用的硬件特性 |
|---|---|
| v1 | smem 复用抬高 AI（12.7 → roofline 斜线上） |
| v2 | 张量核 8× 吞吐（1024 vs 128 FLOP/clk/SM） |
| v3 | cp.async 异步引擎：加载与计算重叠 |
| v7 | 低 occupancy 换高 ILP + L2 swizzle（24MB 窗口） |
| v8 | TMA 单线程发射 + mbarrier 流水（1 block/SM 下唯一出路） |

---

## 附：测量方法备忘（可复现）

| 数据 | 方法 | 文件 |
|---|---|---|
| BF16/TF32 峰值 | 8 独立累加器 mma 循环，满载时钟 2405 MHz 下计时 | `bench/bench_micro.cu` |
| FP8 峰值 | 8 独立累加器 `mma.sync.m16n8k32.e4m3` 循环 | /tmp/opencode/peak_fp8.cu |
| FP32 峰值 | PTX `fma.rn.f32` × **16** 独立链（4 链只有 13.76，是依赖链限制非通路极限；nvcc 内建 fmaf 会被优化掉） | /tmp/opencode/ffma16b.cu |
| 指令支持（FP8/FP4/tcgen05/WGMMA） | 编译探针：各写最小 asm 用例，`nvcc -arch=sm_121/120/89/90` 交叉验证语法有效性（**必须看具体报错文本，只数 error 行数曾导致 FP8 误判为不支持**） | /tmp/opencode/isa_probe.cu |
| DRAM/L2 延迟 | 单线程随机置换指针追逐（8MB→L2，512MB→DRAM） | /tmp/opencode/lat.cu |
| L2 带宽 | ≤L2 容量 buffer × 48×32 blocks 反复读 | /tmp/opencode/l2bw.cu |
| DRAM 带宽 | 1GB grid-stride float4 读 | `bench/bench_micro.cu` |
| smem 带宽 | v5 消融探针纯 ldsm（微基准聚合吞吐，非硬件规格值） | `bench/probe_*.cu`（v5 时代） |
| 时钟 | 满载 kernel 运行中 nvidia-smi 轮询（每 0.4s） | — |
