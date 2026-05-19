# NZ Fractal 布局

Cube 的内部缓冲（`L1` / `cbuf`、`L0A`、`L0B`、`L0C`）都使用 **fractal NZ 布局**，而不是行优先 ND。理解 NZ 布局是编写 cube 数据搬运指令、推理 MAD 操作数组织的前提。

## 定义

给定硬件常数 `C0 = 32 字节`，对元素字节宽度为 `E = sizeof(T)` 的类型：

- 内层 tile 宽度：`K0 = N0 = C0 / E`（例如 `f16` / `bf16` 是 `K0 = 16`；`f32` 是 `K0 = 8`）
- 内层 tile 高度：`M0 = 16`

逻辑 `[M, K]` 张量的 NZ 重新索引：

```text
NZ 索引：(k1, m1, m0, k0)
  其中  k1 = k / K0,  k0 = k % K0
        m1 = m / M0,  m0 = m % M0
物理布局：K1 x M1 x M0 x K0（最后一维连续）
```

对 `[K, N]` 张量做同样的外/内层分解，只是内层宽度轴换成 `N0`。

## 各缓冲的 NZ 布局

| 缓冲 | 逻辑形状 | 物理 NZ 布局 | 备注 |
|------|----------|--------------|------|
| L1（cbuf）- 张量 A | `[M, K]` | `K1 M1 M0 K0` | 行优先 A 被打入 NZ 布局 |
| L1（cbuf）- 张量 B | `[K, N]` | `K1 N1 K0 N0` | 行优先 B 被打入 NZ 布局 |
| L0A（左操作数） | — | `K1 M1 M0 K0` | A5 上为 FRACTAL_NZ / A3 上为 FRACTAL_ZZ：与 L1 cbuf 同 NZ 顺序 |
| L0B（右操作数） | — | `K1 N1 N0 K0` | FRACTAL_ZN：外层行优先，内层列优先（K0 最内） |
| L0C（累加器） | `[M, N]` | `N1 M1 M0 N0` | MMAD 输出（FRACTAL_NZ：外层列优先、内层行优先） |

## 为什么 L0B 必须 K-innermost？

Cube 的归约轴是 `K`。L0B 要求 K 在最内层（`K1 N1 N0 K0`），这样 cube 硬件每个 cycle 都能读到完整的 `K0` 个元素而不跨 stride。

从 L1 的 `K1 N1 K0 N0` 到 L0B 的 `K1 N1 N0 K0` 的内层 box 转置是由 [`pto.mte_l1_l0b`](./ops/data-movement/mte-l1-l0b_zh.md) 这条结构化右侧加载指令在搬运过程中完成的，用户层面看不到额外的转置 pass。从 L1 搬到 L0B 时，每个 512B fractal Z-block 都会被原位置换。

## 数据流：GM → L1 → L0A/B → L0C

```text
+------------------------------------------------------------------------------+
|             GEMM 数据布局：GM -> L1 (NZ) -> L0A/B -> L0C                      |
+------------------------------------------------------------------------------+

步骤 1 — Global Memory（ND，行优先）
-------------------------------------
 张量 A [M, K]                       张量 B [K, N]
 物理：A[m*K + k]                    物理：B[k*N + n]

步骤 2 — GM -> L1（cbuf）：ND→NZ fractal 重排
---------------------------------------------
 L1 中的 A：K1 x M1 x M0 x K0       L1 中的 B：K1 x N1 x K0 x N0
 物理：A_nz[k1][m1][m0][k0]         物理：B_nz[k1][n1][k0][n0]

步骤 3 — L1 -> L0A / L0B
--------------------------
 L0A：cbuf K1 M1 M0 K0 --mte_l1_l0a--> L0A K1 M1 M0 K0  (A5 上为 FRACTAL_NZ)
 L0B：cbuf K1 N1 K0 N0 --mte_l1_l0b--> L0B K1 N1 N0 K0  (FRACTAL_ZN, K0 最内)

步骤 4 — MAD：L0A x L0B -> L0C
-------------------------------
 dst[m, n] = sum k in 0..K-1: lhs[m, k] * rhs[k, n]
 L0C 布局：N1 M1 M0 N0

步骤 5 — L0C 回写（FIXPIPE）
-----------------------------
 FIXPIPE MTE 指令（mte_l0c_l1 / mte_l0c_gm / mte_l0c_ub）把 L0C NZ 结果转换为
 所需的目标布局（通常是 ND），并写入指定的内存空间。
```

## 编写指引

当源 GEMM 操作数本身已经是某种已转置的逻辑布局时，应该在结构化加载层（`pto.mte_l1_l0a` / `pto.mte_l1_l0b`）显式表达这一点，不要寄希望于事后对同一段字节做不同的 NZ 解释。用错误的外/内层分解去操作一块 NZ 缓冲是 verifier 错误，也是最常见的正确性 bug 来源之一。

## 相关章节

- [缓冲层级](./buffer-hierarchy_zh.md)
- [FIXPIPE 模型](./fixpipe-model_zh.md)
- [Cube MAD 指令](./README_zh.md#矩阵乘加mad指令)
- [Cube 数据搬运指令](./README_zh.md#cube-数据搬运指令)
