# 矩阵与矩阵-向量指令集

这一组指令覆盖 PTO tile 路径里的 cube 计算指令。基础型负责生成新的累加器结果，`_acc` 型负责在已有累加器上继续累加，`_bias` 型负责把偏置并入矩阵乘积，`*_mx` 型则在此基础上再引入 MX block-scale 所需的左右 scale tile。

这不是普通逐元素 tile 运算的变体。它的合法性取决于专门的矩阵角色：`Left`、`Right`、`Acc`、`Bias`、`ScaleLeft`、`ScaleRight`，还取决于 A2A3 与 A5 各自的布局和数据类型约束。

## 操作

| 操作 | 作用 | C++ 内建接口 | 说明 |
| --- | --- | --- | --- |
| [pto.tmatmul](./ops/matrix-and-matrix-vector/tmatmul_zh.md) | 生成新累加器结果的矩阵乘 | `TMATMUL(C, A, B)` | 新输出块 |
| [pto.tmatmul_acc](./ops/matrix-and-matrix-vector/tmatmul-acc_zh.md) | 在已有累加器上继续累加的矩阵乘 | `TMATMUL_ACC(C, A, B)` | K 维分块主力形式 |
| [pto.tmatmul_bias](./ops/matrix-and-matrix-vector/tmatmul-bias_zh.md) | 带列偏置的矩阵乘 | `TMATMUL_BIAS(C, A, B, bias)` | Bias tile 为单行 |
| [pto.tmatmul_mx](./ops/matrix-and-matrix-vector/tmatmul-mx_zh.md) | MX block-scale 矩阵乘 | `TMATMUL_MX(C, A, AScale, B, BScale)` | 仅 A5 |
| [pto.tgemv](./ops/matrix-and-matrix-vector/tgemv_zh.md) | 生成新累加器结果的 GEMV | `TGEMV(C, A, B)` | `m = 1` 的 cube 形式 |
| [pto.tgemv_acc](./ops/matrix-and-matrix-vector/tgemv-acc_zh.md) | 在已有累加器上继续累加的 GEMV | `TGEMV_ACC(C, A, B)` | 累加形式 |
| [pto.tgemv_bias](./ops/matrix-and-matrix-vector/tgemv-bias_zh.md) | 带偏置的 GEMV | `TGEMV_BIAS(C, A, B, bias)` | Bias tile 为单行 |
| [pto.tgemv_mx](./ops/matrix-and-matrix-vector/tgemv-mx_zh.md) | MX block-scale GEMV | `TGEMV_MX(C, A, AScale, B, BScale)` | 仅 A5 |

## 这组指令为什么单列出来

PTO 没把矩阵乘积类操作混进普通 tile 算术里，原因很直接：cube 路径有自己的一套角色、布局、合法性和 target 限制。读者通常需要先弄清四件事：

- 左右输入和累加器分别是谁；
- “生成新结果”和“在旧结果上继续累加”有什么区别；
- bias 是怎么并入结果的；
- A2A3 与 A5 的布局规则到底哪里一样、哪里不一样。

把这几个问题拆散写在零散 leaf 页里，查阅成本会很高，因此这组页需要一个正式的家族入口。

## 机制

### TMATMUL

设 `M = a.GetValidRow()`、`K = a.GetValidCol()`、`N = b.GetValidCol()`，则：

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

`pto.tmatmul` 的含义是“生成一个新的输出累加器块”，因此它适合 K 循环的首轮，或者任何不需要保留旧累加器内容的场景。

### TMATMUL_ACC

$$ \mathrm{C1}_{i,j} = \mathrm{C0}_{i,j} + \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

这类指令存在的原因，是块化 GEMM 在 K 维循环中往往需要反复把部分乘积叠加到同一个累加器上。把“新结果”与“继续累加”拆成两条指令，可以把调度和资源语义说清楚。

### TMATMUL_BIAS

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} + \mathrm{Bias}_{0,j} $$

Bias tile 只有一行，因此它按输出列广播。`Bias[0, j]` 会加到结果矩阵的整列 `j` 上。

### TGEMV

GEMV 可以理解为 `m = 1` 的 cube 合同。PTO 没有把它硬塞进普通 matmul 页脚里，而是单列出来，因为它在接口、用法和约束表达上都更接近“矩阵乘一条向量”，而不是任意二维块的对乘。

### MX 变体

`*_mx` 不是“多一个 scale 参数”这么简单。对 MXFP4、MXMP8 这类 block-scale 格式，PTO 把 scale 也建模成 tile：

- 左输入在 `Left`，
- 右输入在 `Right`，
- 左 scale 在 `ScaleLeft`，
- 右 scale 在 `ScaleRight`，
- 输出或累加器在 `Acc`。

也就是说，MX 变体要求左右两侧各自带一块 scale tile，而不是只附带一个笼统的 scale 张量。

## Tile 角色与缓冲区语义

从架构视角看，tile 角色是对目标缓冲区的抽象：

- `Left` 表示左操作数 tile，对应 L0A 路径。
- `Right` 表示右操作数 tile，对应 L0B 路径。
- `Acc` 表示累加器 / 输出 tile。
- `Bias` 表示 `*_bias` 使用的单行偏置 tile。
- `ScaleLeft` 与 `ScaleRight` 表示 MX block-scale 变体使用的左右 scale tile。

这里最容易写错的是 `Right`。虽然 A2A3 和 A5 都有 `Right` 角色，但不能把它们理解成“同一套可移植物理布局”。对右操作数的具体布局要求，A2A3 与 A5 并不相同，必须以各自的 target profile 和 leaf 页约束为准。

## 目标 Profile

本手册里：

- `A2A3` 指 Ascend 910B 与 Ascend 910C；
- `A5` 指 Ascend 950 PR 与 Ascend 950 DT。

| 能力 | CPU 模拟器 | A2A3 | A5 |
| --- | :---: | :---: | :---: |
| `TMATMUL` / `TMATMUL_ACC` / `TMATMUL_BIAS` | 是 | 是 | 是 |
| `TGEMV` / `TGEMV_ACC` / `TGEMV_BIAS` | 是 | 是 | 是 |
| int8 cube 路径 | 否 | 是 | 是 |
| fp16 / bf16 / fp32 cube 路径 | 是 | 是 | 是 |
| fp8 cube 路径 | 否 | 否 | 是 |
| MX block-scale 路径 | 否 | 否 | 是 |

### 通用合法性

- matmul 的形状必须满足 `(M, K) x (K, N) -> (M, N)`。
- GEMV 使用同一套 cube 合同，只是 `m = 1`。
- `Left`、`Right`、`Acc`、`Bias`、`ScaleLeft`、`ScaleRight` 必须与所发射的具体指令匹配。
- 超出合法输出域的内容不会被这组指令自动“修正”为某种可移植结果。

### A2A3 说明

- 基础 cube 路径支持仓内文档已列出的组合，例如 `(int32, int8, int8)`、`(float, half, half)`、`(float, bfloat16_t, bfloat16_t)`。
- 动态 `m`、`k`、`n` 范围受限于 `[1, 4095]`。
- backend 会显式检查 `Left` / `Right` / `Acc` 的角色组合是否合法。

### A5 说明

- 基础 cube 路径允许 `int32` 累加器配 int8 输入对，也允许 `float` 累加器配 fp16、bf16、fp32 和部分 fp8 输入对。
- `Right` 角色在 A5 上有自己的一套布局 / fractal 约束，不能照搬 A2A3 的理解。
- MX 变体是 A5 专属路径，且要求同时提供 `ScaleLeft` 与 `ScaleRight`。

## 性能与吞吐

仓内当前公开的性能证据主要来自 A2A3 costmodel。`TMATMUL`、`TMATMUL_ACC`、`TMATMUL_BIAS`、`TGEMV`、`TGEMV_ACC`、`TGEMV_BIAS` 最终都落到共享的 `mad/mmad` cube 指令模型。

A2A3 的公开模型可以写成：

- 启动开销：`14` cycles；
- repeat 次数：`ceil(M/16) * ceil(N/16) * ceil(K / baskK)`；
- `baskK = 32 / sizeof(left_element_type)`；
- 单个 repeat 的稳态代价：
  - int8、fp16 bucket 为 `1` cycle；
  - fp32 bucket 为 `2` cycles。

因此，仓内已公开的 A2A3 公式是：

```text
cycles = 14 + repeat_count * repeat_cost
```

已有 costmodel 测试样例包括：

- half `40x50 * 50x60`：`62` cycles；
- int8 `6x7 * 7x8`：`15` cycles；
- float `120x110 * 110x50`：`910` cycles。

对 GEMV，因为 `m = 1`，公式仍然成立，只是 `ceil(M/16)` 固定为 `1`。
当前仓库没有公开单列的 A5 latency / throughput 表，因此 A5 在这组页里只能精确写合法性和数据类型边界，不能凭空补周期数字。

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [Tile 指令表面](../instruction-families/tile-families_zh.md)
- [位置意图与合法性](../state-and-types/location-intent-and-legality_zh.md)
- [布局](../state-and-types/layout_zh.md)
