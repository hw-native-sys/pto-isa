# pto.tmatmul.bias

`pto.tmatmul.bias` 属于[矩阵与矩阵-向量运算](./tile/ops/matrix-and-matrix-vector/tmatmul-bias_zh.md)指令集。

## 概述

带偏置加法的矩阵乘法，在有效域 `0 <= i < M`、`0 <= j < N` 上计算 $\mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} + \mathrm{Bias}_{0,j}$，偏置广播行为由实现定义。

## 机制

设：
- `M = aMatrix.GetValidRow()`
- `K = aMatrix.GetValidCol()`
- `N = bMatrix.GetValidCol()`

对于 `0 <= i < M` 和 `0 <= j < N`：

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} + \mathrm{Bias}_{0,j} $$

偏置广播行为由实现定义。

## 语法

### PTO-AS

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
PTO_INST RecordEvent TMATMUL_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData, WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename TileBias,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `aMatrix` | Left | 左操作数矩阵 Tile |
| `bMatrix` | Right | 右操作数矩阵 Tile |
| `biasData` | Bias | 偏置 Tile（一行，广播到输出） |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `cMatrix` | Acc | 带偏置的 GEMM 结果 Tile |

## 副作用

偏置加法可能触发目标特定的溢出处理或舍入行为。

## 约束

- 所有来自 `TMATMUL` 的约束都适用于 `(cMatrix, aMatrix, bMatrix)` 三元组。
- 偏置约束 (A2A3):
    - `TileBias::DType` 必须匹配 `TileRes::DType`。
    - `TileBias::Loc == TileType::Bias` 且 `TileBias::Rows == 1`。
- 偏置约束 (A5):
    - `TileBias::DType` 必须匹配 `TileRes::DType`。
    - `TileBias::Loc == TileType::Bias`、`TileBias::Rows == 1` 且 `TileBias::isRowMajor`。

## 异常与非法情形

- 当 `TileBias::DType` 与 `TileRes::DType` 不匹配时行为未定义。
- 当 `TileBias::Rows != 1` 时行为未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 偏置加法 | ✓ | ✓ | ✓ |
| 偏置行主序 | ✗ | ✗ | ✓ |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using Bias = Tile<TileType::Bias, half, 1, 16>;
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
  using Bias = Tile<TileType::Bias, half, 1, 16>;
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

### PTO-AS

```text
%acc = tmatmul.bias %a, %b, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

AS Level 2 (DPS)：

```mlir
pto.tmatmul.bias ins(%a, %b, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

![TMATMUL_BIAS tile operation](../figures/isa/TMATMUL_BIAS.svg)

## 相关页面

- 指令集总览：[矩阵与矩阵-向量运算](./tile/ops/matrix-and-matrix-vector/tmatmul-bias_zh.md)
