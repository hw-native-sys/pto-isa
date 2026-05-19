# Cube 微指令参考

本节记录 PTO 的 **Cube 微指令表面**：矩阵乘加（MAD）以及面向 cube core（AIC）和其专用缓冲层级（L1 / L0A / L0B / L0C / BT）的数据搬运指令。

!!! note "范围与受众"
    Tile 级的矩阵指令（例如 `pto.tmatmul`，见 [Tile ISA 矩阵与矩阵-向量](../tile/matrix-and-matrix-vector_zh.md)）把这些底层原语隐藏在 tile 形状的接口之后。本节描述的 cube 微指令是编译器后端与手写 cube 内核直接对接的低层表面，把 NZ fractal 布局、L1/L0 缓冲层级、FIXPIPE 回写显式化。

## 架构背景

| 页面 | 用途 |
|------|------|
| [NZ Fractal 布局](./nz-fractal-layout_zh.md) | L1、L0A、L0B、L0C 使用的 fractal NZ 格式，定义 `(k1, m1, m0, k0)` 重新索引与各缓冲变种。 |
| [缓冲层级](./buffer-hierarchy_zh.md) | L1 / L0A / L0B / L0C / BT 内存层级：地址空间、大小、数据流契约。 |
| [FIXPIPE 模型](./fixpipe-model_zh.md) | FIXPIPE 回写通路：L0C 结果如何转换回 ND 并路由到 UB 或 GM。 |

## 矩阵乘加（MAD）指令

MAD 家族在 cube 的 L0A / L0B / L0C 缓冲上计算 `dst = lhs @ rhs`。所有变体共享相同的 `(M, N, K)` 形状参数与一组可选 clauses（`unit_flag`、`disable_gemv`、`sat`/`nosat`、`tf32_mode`、`n_dir`）。

| 指令 | 语义 |
|------|------|
| [pto.mad](./ops/mad/mad_zh.md) | 零初始化：`dst = lhs @ rhs` |
| [pto.mad_acc](./ops/mad/mad-acc_zh.md) | 累加：`dst = dst + lhs @ rhs` |
| [pto.mad_bias](./ops/mad/mad-bias_zh.md) | 偏置初始化：`dst = lhs @ rhs + bias[n]` |
| [pto.mad_mx](./ops/mad/mad-mx_zh.md) | MX（微缩放）零初始化 matmul |
| [pto.mad_mx_acc](./ops/mad/mad-mx-acc_zh.md) | MX 累加 matmul |
| [pto.mad_mx_bias](./ops/mad/mad-mx-bias_zh.md) | MX 偏置初始化 matmul |

## Cube 数据搬运指令

这些指令在 GM、L1、L0A/L0B、L0C 之间搬运 tile，使用与 [标量 DMA Copy](../scalar/dma-copy_zh.md) 同样的内联 `nburst(...)` / `loop(...)` 子句模型。

### GM → L1

- [pto.mte_gm_l1](./ops/data-movement/mte-gm-l1_zh.md)：直接 GM→L1 加载（不做布局变换）
- [pto.mte_gm_l1_frac](./ops/data-movement/mte-gm-l1-frac_zh.md)：GM→L1 并完成 ND→NZ fractal 重排

### L1 ↔ UB

- [pto.mte_l1_ub](./ops/data-movement/mte-l1-ub_zh.md)：L1→UB（cube→vector 数据通路）
- [pto.mte_ub_l1](../scalar/ops/dma-copy/mte-ub-l1_zh.md)：UB→L1（vector→cube 数据通路；位于标量 DMA 节）

### L1 → L0A / L0B（cube 操作数加载）

- [pto.mte_l1_l0a](./ops/data-movement/mte-l1-l0a_zh.md)：把 L1 NZ tile 加载到 L0A（左操作数）
- [pto.mte_l1_l0b](./ops/data-movement/mte-l1-l0b_zh.md)：把 L1 NZ tile 加载到 L0B（右操作数，K-innermost 转置）
- [pto.mte_l1_l0a_mx](./ops/data-movement/mte-l1-l0a-mx_zh.md)：为 L0A 加载 MX scale payload
- [pto.mte_l1_l0b_mx](./ops/data-movement/mte-l1-l0b-mx_zh.md)：为 L0B 加载 MX scale payload

### L1 → BT（偏置）

- [pto.mte_l1_bt](./ops/data-movement/mte-l1-bt_zh.md)：把 bias 向量加载到 BT，供 `pto.mad_bias` / `pto.mad_mx_bias` 消费
- [pto.mte_l1_fb](./ops/data-movement/mte-l1-fb_zh.md)：加载 FIXPIPE 相关 payload（例如反量化参数）

### L0C 回写（FIXPIPE）

- [pto.mte_l0c_l1](./ops/data-movement/mte-l0c-l1_zh.md)：FIXPIPE 回写 L0C → L1
- [pto.mte_l0c_gm](./ops/data-movement/mte-l0c-gm_zh.md)：FIXPIPE 回写 L0C → GM
- [pto.mte_l0c_ub](./ops/data-movement/mte-l0c-ub_zh.md)：FIXPIPE 回写 L0C → UB

## 完整 Cube 流水线

```text
GM (ND)          L1/cbuf (NZ)            L0A/B (NZ)          L0C (NZ)    GM (ND)

A[M,K] --mte_gm_l1_frac/mte_gm_l1--> K1 M1 M0 K0 --mte_l1_l0a-->  K1 M1 M0 K0 -+
                                                             +-MAD-> N1 M1 M0 N0 --> C[M,N]
B[K,N] --mte_gm_l1_frac/mte_gm_l1--> K1 N1 K0 N0 --mte_l1_l0b--> K1 N1 N0 K0 -+
                               ^
                    必要时由 mte_l1_l0b 进行转置
                    不在 GM→L1 阶段做
```

## 相关章节

- [Tile ISA：矩阵与矩阵-向量](../tile/matrix-and-matrix-vector_zh.md) — Tile 级矩阵指令
- [标量 DMA Copy](../scalar/dma-copy_zh.md) — UB 侧分组 DMA 传输
- [流水线同步](../scalar/ops/pipeline-sync/) — Cube / Vector 同步原语
