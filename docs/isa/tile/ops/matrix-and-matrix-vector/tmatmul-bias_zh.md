# pto.tmatmul.bias

`pto.tmatmul.bias` 属于[矩阵与矩阵-向量](../../matrix-and-matrix-vector_zh.md)指令集。

## 概述

`TMATMUL_BIAS` 表示"矩阵乘法后立即并入列偏置"。它表达的是矩阵乘积再加一行 bias，而不是另一种不同的乘法。把 bias 作为这条指令的显式输入有两个好处：一是合同清楚，二是文档不必把"先做 matmul、再做逐元素加"误写成完全等价的抽象。

设 `M = aMatrix.GetValidRow()`，`K = aMatrix.GetValidCol()`，`N = bMatrix.GetValidCol()`，对 `0 <= i < M`、`0 <= j < N`：

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} + \mathrm{Bias}_{0,j} $$

Bias tile 只有一行，因此它按输出列广播。

## 机制

`TMATMUL_BIAS` 仍然走 `Left` / `Right` / `Acc` 的 cube 路径，只是在乘积生成后再引入一块 `Bias` tile。这条指令要求 bias 是"单行偏置 tile"，而不是任意 shape 的普通 tile。也正因为如此，它表达的是列偏置，而不是一般意义上的逐元素加法。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

同步形式：

```text
%acc = tmatmul.bias %a, %b, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%c = pto.tmatmul.bias %a, %b, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tmatmul.bias ins(%a, %b, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData,
                                  WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename TileBias,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData,
                                  WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `aMatrix` | Left | 左操作数 tile |
| `bMatrix` | Right | 右操作数 tile |
| `biasData` | Bias | 偏置 tile，必须为单行 |
| `cMatrix` | Acc | 结果累加器 tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `cMatrix` | `Acc` | 先得到矩阵乘积，再把 `bias[0, j]` 加到每个输出列 `j` 上 |

## 副作用

结果写入 `cMatrix` 累加器。

## 约束

- `TMATMUL` 的所有 shape、角色、dtype 和 target-profile 约束在这里同样成立。
- `biasData` 的数据类型必须与结果累加器 `TileRes::DType` 一致。
- `biasData` 必须是单行 Bias tile。
- A2A3 要求 `TileBias::Loc == TileType::Bias` 且 `TileBias::Rows == 1`。
- A5 要求 `TileBias::Loc == TileType::Bias`、`TileBias::Rows == 1` 且 `TileBias::isRowMajor == true`。

## 异常与非法情形

- 用普通 tile 代替 Bias tile。
- bias 不是单行。
- bias dtype 与结果累加器 dtype 不一致。
- 违反 `TMATMUL` 的任一合法性约束。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| bias 支持 | 支持 | 支持 | 支持 |
| 布局要求 | 无额外要求 | 无额外要求 | bias 必须 row-major |

## 性能与吞吐

当前仓内 A2A3 costmodel 对 `TMATMUL_BIAS` 仍复用 `mad/mmad` 的同一套模型，周期口径与 `TMATMUL` 一致：

```text
cycles = 14 + ceil(M/16) * ceil(N/16) * ceil(K / baskK) * repeat_cost
```

其中 `baskK = 32 / sizeof(left_element_type)`；int8、fp16 bucket 的 `repeat_cost = 1`；fp32 bucket 的 `repeat_cost = 2`。当前仓库没有单列的 A5 latency / throughput 表。

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using Bias = Tile<TileType::Bias, float, 1, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  Bias bias;
  C c;
  TMATMUL_BIAS(c, a, b, bias);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using Bias = Tile<TileType::Bias, float, 1, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  Bias bias;
  C c;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(bias, 0x3000);
  TASSIGN(c, 0x4000);
  TMATMUL_BIAS(c, a, b, bias);
}
```

## 相关页面

- [TMATMUL](./tmatmul_zh.md)
- [TMATMUL_ACC](./tmatmul-acc_zh.md)
- [TMATMUL_MX](./tmatmul-mx_zh.md)
- 指令集总览：[矩阵与矩阵-向量](../../matrix-and-matrix-vector_zh.md)
