# fp8_v1：v13 骨架 + e4m3 引擎切换 —— 8192³ 166.5 TFLOPS（BF16 的 1.75×）

> **接力链**：BF16 线 v13 后（8192³ 95.0，计算核 113.5 已贴 BF16 墙 118.37），FP8 是唯一未动的天花板级方向（峰值实测 235.79 = 2×）→ 阶段 A 布局探针（mma.m16n8k32.e4m3 对拍 CPU 逐位一致，b16 字节对同构成立）→ 阶段 B 量化 kernel（动态 abs-max/448 + B 写侧转置 [N][K]，A 245 / B 转置 221 GB/s）→ 阶段 C 参照系（GB16 有 cuBLASLt FP8 引擎：8192³ 200.26）→ 本版动作 = v13 骨架（TMA+SWIZZLE_128B+mbarrier+unroll2+4-pass epilogue）原样保留，仅换 mma 指令与 B 侧数据通路 → 实测（协议 v2.1 修订后终值）= 大形状 **1.54–1.80× BF16 v13**，1024³ 反超 cuBLASLt FP8（95.3 vs 86.8），精度与 cuBLASLt **逐位一致**（6 形状 max_abs=0）。

---

## Step 1 — 立项依据（roofline 预判 + 结构推导）

**理论来源**：BK=128 的 e4m3 A tile 与 v13 BK=64 的 bf16 A tile **smem 几何相同**（128 行 × 128B、8 个 16B chunk）——SWIZZLE_128B 公式与 A-LDSM 寻址逐字节照搬；每 K-block 指令数与 v13 完全相同（16 A-ldsm + 16 B-ldsm + 128 mma），但 K-block 覆盖 128 k（2×）→ **同等指令流吞吐翻倍**。

**roofline 预判**（fp8/criterion 2.2，实测兑现）：FP8 搬运字节减半但带宽墙不变（255.6 GB/s）——8192³ AI ≈ 2·8192/(2+2+4) = 2048 > 转折点 922，compute-bound 主导，但达不到 2×。实测 1.75×，与预判吻合。

## Step 2 — 方案（三处替换，骨架零改动）

| 部件 | v13（BF16） | fp8_v1 | 依据 |
|---|---|---|---|
| mma | `m16n8k16.bf16`（4 k16 slice/K块） | `m16n8k32.e4m3`（4 k32 slice/K块） | 阶段 A 逐位验证 |
| B 数据通路 | [K][N] k-major + TMA + x4.**trans** LDSM | 量化 kernel 写侧转置 [N][K] + TMA（内维 K）+ x4 非 trans LDSM（每 x4 出 2 个 n-group 的 {b0,b1}） | 阶段 A 路线 b + 阶段 B 转置 kernel |
| scale | 无 | per-tensor：epilogue 乘 sA·sB 一次 | hpc-ops 图纸简单路线（blockwise 留 v2+） |

其余（block swizzle、mbarrier full/empty、fence.acq_rel.cta、unroll 2、4-pass TMA store epilogue、128×256 BigCfg）与 v13 逐字一致。寄存器 193（v13 BigCfg 192），smem 96KB 不变。

## Step 3 — 关键代码

```cpp
// B-LDSM（n-major 非 trans，阶段 A 验证的 x4 双 n-group 形态）：
//   lane 角色 gl=lane/8：gl=0→(ng_lo,k-lo) gl=1→(ng_lo,k-hi) gl=2→(ng_hi,k-lo) gl=3→(ng_hi,k-hi)
const int ng = npj * 2 + gl / 2;             // 本 lane 覆盖的 n-group
const int row = ng * 8 + lane % 8;           // chunk 内 n 行
const int chunk = ks * 2 + (gl % 2);         // k-lo / k-hi 的 16B chunk
uint32_t addr = __cvta_generic_to_shared(
    b_base + warp_n * B_CHUNK_BYTES + row * 128 + (chunk ^ (row & 7)) * 16);
ldmatrix.x4 {r0,r1,r2,r3} → b[npj*2] = {r0,r1}; b[npj*2+1] = {r2,r3};

// mma（k32 一发）：
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
    {c}, {a[mi][0..3]}, {b[ng][0], b[ng][1]}, {c};
```

## Step 4 — 验证

**精度（6 形状 vs cuBLASLt FP8 同精度同 scale，全部 max_abs_err = 0 逐位一致）**：512³、1024³、2048³、4096³、1000×992×1008（非方形非 2 幂）、512×2048×1024。约束：K%16（TMA uint8 步长对齐，比 BF16 线的 %8 严）。

**性能（协议 v2.1 终值：热身 ≥300ms + 自适应迭代窗 ≥20ms + 5 轮中位 + 同会话交替对照）**：

| 尺寸 | fp8_v1（5 轮中位） | p25–p75 | BF16 v13 | 倍数 | cuBLASLt FP8 | fp8_v1/cuBLASLt |
|---|---|---|---|---|---|---|
| 512³ | 18.7 | 18.7–18.7 | 29.8 | 0.63× | 49.5 | 37.8% |
| 1024³ | **95.3** | 95.3–95.3 | 58.2 | **1.64×** | 86.8 | **109.8%** ✅ |
| 2048³ | 121.2 | 119.5–122.0 | 78.7 | 1.54× | 164.3 | 73.8% |
| 4096³ | 156.3 | 155.2–156.6 | 86.7 | **1.80×** | 176.8 | 88.4% |
| 8192³ | 164.8 | 161.3–169.1 | 95.0 | 1.73× | 202.3 | 81.5% |

单发内核时长（实测）：1024³ 23.7μs / 4096³ 0.90ms / 8192³ 6.6ms（工作总量 1.1 TFLOP）。

**阶段门 D 判定（≥1.3× BF16 v13）**：全主形状通过（1.54–1.80×）→ **FP8 线在 GB10 成立**，采纳为 fp8_v1。峰值利用率 50.9–69.9%（FP8 口径）。

**如实记录**：512³ 仅 18.7（BF16 v13 的 0.63×）——单 BigCfg 的填充病（8 block/48 SM = 17%），与 v12 之前的 BF16 同病，v2 加配置族即可（骨架已模板就绪）。

### 性能数字修订记录（2026-11-18，协议 v2 → v2.1）

初版报告的 3 轮数字（1024³ 95.0 / 4096³ 157.4 / 8192³ 166.5）经 10 轮复测 + 协议升级后修订为终值（95.3 / 156.3 / 164.8）。两个测量教训（已入 fp8/criterion 2.2.1）：

1. **3 轮对 2 幂病理形状不够**：8192³ 初版 3 轮全落在高位（166.5），10 轮中位实为 161.7，协议升级后终值 164.8——裁决轮数按形状分级（2 幂形状 ≥5，大形状热身 ≥300ms）；
2. **报告里曾出现未经测量的数字**："单发 1.24ms" 是从错误假设反推的脑补值（实际 6.6ms），且该假数字曾把一次回归排查带偏一小时——**禁止报告任何未直接测量或未由总窗/迭代数计算的数字**。

## 踩坑与定位链（本轮最有价值的部分）

**核心 bug：B 的 TMA 坐标序随 tmap 方向翻转而未同步交换**。v13 的 B tmap 内维是 N，调用 `(x=block_n+g*64, y=kb*BK)` 正确；fp8 的 B tmap 翻转为内维 K 后，坐标必须变 `(x=kb*BK, y=block_n+g*64)`——照抄调用导致 chunk 1–3 读错位置（K 坐标越界 + OOB 零填充）。**chunk 0 因 x=y=0 恰好掩盖 bug**。

三层定位链（每个都值得复用）：

1. **NOTMA 直写对照**：kernel 内绕过 TMA、按 ldsm 期望布局手工直写 smem → PASS → bug 在 TMA 数据通路而非 fragment/mma 逻辑（一步排除一半代码）；
2. **错位分桶统计**：按 n/64 分桶 → "chunk 0 全对、chunk 1–3 全错" → 把"哪块错"变成"哪个坐标错"；
3. **smem dump**：TMA 写完的原始字节 vs 期望公式逐位对比 → 排除 swizzle 公式（行 0–1 全中），锁定坐标语义。

另一条（阶段 C 遗留）：**对称单位阵检验对转置错误假阳性**（k·N+k = k+n·K ⟺ k=n 对角自洽）——语义验证必须用非对称随机数据。

## 下一步（v2 候选，按性价比）

1. **配置族移植**（小形状）：512³ 填充 17% → 128×64 SmallCfg 预期翻倍以上——v12 的成熟经验，工作量小；
2. **与 cuBLASLt 差距的消融定位**（大形状 82–86%）：候选——stage 数（BK=128 下 2-stage 的 fill/drain 占比比 BK=64 时代更小但 K 块数减半，摊薄结构不同）、带宽（e4m3 半字节后 AI 翻倍，可能更贴带宽墙）、prologue/epilogue 占比。先跑消融探针再动手；
3. blockwise scale v2（主循环逐 K 块乘）——精度增强方向，性能优先级低。
