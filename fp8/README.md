# FP8 GEMM 路线（立项文档）

> **接力链**：BF16 线 v13 后（4096³ 90.0 / 8192³ 95.0，cuBLAS 96–104%），剩余方向排序中 FP8 是
> 唯一天花板级机会——峰值实测 **235.79 TFLOPS = BF16 的 2×**（指令已验证：`mma.sync.m16n8k32.f32.e4m3`）。
> 本文档汇总立项侦察（硬件事实 + hpc-ops 图纸 + 布局推导）与分阶段计划。

---

## 0. 硬件事实（全部实测在案）

| 项 | 值 | 来源 |
|---|---|---|
| FP8 mma 指令 | `mma.sync.aligned.m16n8k32.row.col.f32.e4m3/e5m2` | 编译探针通过 |
| FP8 峰值 | 235.79 TFLOPS（2× BF16 的 118.37） | 微基准（bench_micro） |
| FP4 | 未验证 | — |
| GB10 无 | tcgen05 / TMEM / WGMMA——只能走 mma.sync 路线 | hardware.md |

## 1. hpc-ops 图纸（侦察结论，源码在案）

### 1.1 scale 应用的两种结构（决定 kernel 骨架）

- **per-tensor scale**：整个矩阵一个 scale → GEMM 结果在 **epilogue 乘一次**（乘 A_scale × B_scale），主循环纯 e4m3 mma——结构最简单，精度最低；
- **blockwise scale**（DeepSeek 式，每 128 列一组）：**scale 必须在主循环内逐 K 块应用**——每 K 块把 A 块 scale × B 块 scale 乘进该块的累加贡献（hpc-ops `group_gemm/sm90/kernels.cuh`：`tCS[in] = sAS(...) * wscale` 逐块乘），不能攒到 epilogue（各 K 块 scale 不同，攒了就错）。

**结论**：先做 per-tensor 版（v1），blockwise 版（v2）需主循环内加逐块 scale 乘法（每 K 块 kN 次 FMUL，挤占 ILP，需消融评估）。

### 1.2 量化算子（hpc-ops `src/activation/`，SM121 直接可抄）

- BF16 → e4m3：向量化 8 元素/线程（bf16x2×4 load → float 乘 inv_scale → fp8x4 store），纯带宽型，无 smem 无 barrier；
- e4m3 满量程 **448.0f**（动态量化：abs-max / 448 → scale，防 0 除加 1e-8）；
- **写侧自由**：量化是必经步骤，量化 kernel 可以按 GEMM 想要的布局写输出（见 §2 关键结论）。

### 1.3 精度验证参照（先探针，后定门限）

criterion 规则是"同精度 cuBLAS 对照"。GB10 的 cuBLAS 是否有 FP8 引擎需探针验证（cublasLtMatmul + `CUDA_R_8F_E4M3`）：
- 若有：同精度对照，门限类比 BF16 的 1e-3（预计 FP8 实现差异仍主要是累加顺序）；
- 若无：退一步用 BF16 cuBLAS 对照 + **分离 FP8 固有量化误差**（先量化再反量化跑 BF16 参照，差值归量化，剩余差值归实现）。

## 2. 布局推导（立项时已想清楚的部分，probe 逐位验证）

### 2.1 核心等价：e4m3 tile = b16 字节对（k 相邻两元素装进一个 b16）

m16n8k32.e4m3 的 A/B fragment 与 m16n8k16.bf16 的 fragment **字节结构同构**：
一个 b32 寄存器 = 4 字节 = bf16 两元素（k, k+1）= e4m3 四元素（k, k+1, k+2, k+3，小端序恰好连续）。

**A 侧结论（强）**：把 e4m3 数据按 b16 视图装进 smem 后，v13 的 A-ldsm 寻址**逐字节照搬**——
x4 四矩阵 (rows 0–7 k-lo, rows 8–15 k-lo, rows 0–7 k-hi, rows 8–15 k-hi) 恰好映射 e4m3 fragment 的
(a0=r,k0–15 / a1=r+8,k0–15 / a2=r,k16–31 / a3=r+8,k16–31)。chunk 算术也同构：bf16 k16 slice = 32B = 2 chunk，
e4m3 k32 slice = 32B = 2 chunk——**BK=128 e4m3 时每 stage 的 smem 足迹与 v13 BK=64 完全相同**（A 16KB + B 32KB = 48KB），
SWIZZLE_128B 与现有 ldsm 公式全部复用，K 覆盖翻倍。

**B 侧结论（需 probe 裁决）**：B fragment 每 lane 2 寄存器 = (n = lane/4, k = 4×(lane%4)+0..3) 与 (k+16)。两条布局路线：
- **路线 b（先行）**：全局 B 由量化 kernel **转置写出**（[N][K]，n-major）→ TMA 装载 smem 行 = n、行内 = K 字节 →
  **非 trans ldsm x2** 直接给出正确 fragment（已推导逐位对上）；
- **路线 a（备选）**：保 k-major，量化 kernel 写 k-pair 交错布局（行内 = n，b16 = 同 n 的 k,k+1 两字节）→ trans ldsm。
两条路线都靠"量化 kernel 写侧自由"消除 TMA 无法转置的限制——这是 FP8 管线的结构性优势。

### 2.2 mma 循环结构变化

- BK=128 e4m3 → 每 K 块 4 个 k32 slice（v13 是 4 个 k16 slice）——slice 数、chunk 算术、mbarrier 协议全不变；
- B fragment 每 mma 2 寄存器（vs bf16 每 k16-slice-n8 组 2 寄存器）——寄存器压力近似不变；
- 累加器不变（f32）。

## 3. 分阶段计划（受 fp8/criterion.md 约束——四步循环 + 测量协议 v2 + 阶段门）

| 阶段 | 内容 | 产出 | 状态 |
|---|---|---|---|
| A | fragment 布局 dump 探针 | `fp8/probe_fp8_layout.cu` | ✅ 逐位一致 |
| B | 量化 kernel（per-tensor + 动态 abs-max + B 转置） | `fp8/quantize.cu` | ✅ 245/221 GB/s |
| C | cuBLASLt FP8 支持探针 | `fp8/probe_cublaslt_fp8.cu` / `probe_lt_min.cu` | ✅ 引擎可用（8192³ 200.26） |
| D | fp8 GEMM v1 = v13 骨架 + e4m3 切换 | `fp8/gemm_fp8_v1.cu` + `reports/fp8_v1.md` | ✅ 8192³ 166.5（1.75× BF16，精度逐位一致） |
| E | v2：配置族 + 差距消融 | — | 待立项 |


命名规则：源码与报告都在文件名前缀 `fp8_`，与 BF16 线（v00–v13）并行不互相占用版号。

## 4. 预期与判断点

- **性能预期**：大形状若翻倍到 160–190（受限于实际 AI 与搬运），FP8 线即成立；若只 +30%（带宽吃紧），则记录"FP8 在 GB10 不成立"的实测结论（带宽 255.6 GB/s 是硬墙，A/B 字节减半只省一半搬运时间，K 大时搬运占比小）；
- **精度判断点**：per-tensor 版对([-1,1] 均匀数据) e4m3 相对误差预计 1e-2 量级（e4m3 尾数 3 位）——这是**精度换速度**的固有代价，验证时与参照的差异应在这个量级内；
- **回退成本**：FP8 线独立目录与命名，不影响 BF16 线任何已采纳版本。
