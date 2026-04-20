# pto.tmatmul

`pto.tmatmul` 属于[矩阵与矩阵-向量运算](./tile/ops/matrix-and-matrix-vector/tmatmul_zh.md)指令集。

## 概述

矩阵乘法 (GEMM)，在有效矩阵乘法域 `0 <= i < M`、`0 <= j < N` 上计算 $\mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j}$，其中 `M = aMatrix.GetValidRow()`、`K = aMatrix.GetValidCol()`、`N = bMatrix.GetValidCol()`，生成累加器/输出 Tile。精确的累加器行为和数据类型提升由目标/实现定义。

## 机制

设：
- `M = aMatrix.GetValidRow()`
- `K = aMatrix.GetValidCol()`
- `N = bMatrix.GetValidCol()`

对于 `0 <= i < M` 和 `0 <= j < N`（有效矩阵乘法域中的输出元素）：

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

精确的累加器行为和数据类型提升由目标/实现定义。

## 语法

### PTO-AS

同步形式：

```text
%acc = tmatmul %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%c = pto.tmatmul %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tmatmul ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `aMatrix` | Left | 左操作数矩阵 Tile |
| `bMatrix` | Right | 右操作数矩阵 Tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `cMatrix` | Acc | GEMM 结果 Tile |

## 副作用

GEMM 累加操作可能触发目标特定的溢出处理或舍入行为。

## 约束

- 实现检查 (A2A3):
    - 支持的 `(CType, AType, BType)` 三元组：
        - `(int32_t, int8_t, int8_t)`
        - `(float, half, half)`
        - `(float, float, float)`
        - `(float, bfloat16_t, bfloat16_t)`
    - 静态形状约束：`TileLeft::Rows == TileRes::Rows`、`TileLeft::Cols == TileRight::Rows`、`TileRight::Cols == TileRes::Cols`。
    - Tile 位置：`TileLeft::Loc == Left`、`TileRight::Loc == Right`、`TileRes::Loc == Acc`。
    - 运行时：`m/k/n`（取自 `aMatrix.GetValidRow()`、`aMatrix.GetValidCol()`、`bMatrix.GetValidCol()`）必须在 `[1, 4095]` 范围内。
- 实现检查 (A5):
    - 累加器类型必须是 `int32_t` 或 `float`。
    - 如果是 `int32_t`：`AType == int8_t` 且 `BType == int8_t`。
    - 如果是 `float`：支持 `half/bfloat16_t/float` 和选定的 fp8 对（目标定义）。
    - 静态形状约束：`TileLeft::Rows == TileRes::Rows`、`TileLeft::Cols == TileRight::Rows`、`TileRight::Cols == TileRes::Cols`。
    - 强制执行分形/布局约束：
        - Left：`Loc == Left`、`!isRowMajor`、`SFractal == RowMajor`
        - Right：`Loc == Right`、`isRowMajor`、`SFractal == ColMajor`
        - Acc：`Loc == Acc`、`!isRowMajor`、`SFractal == RowMajor`
    - 运行时：`m/k/n`（取自 `aMatrix.GetValidRow()`、`aMatrix.GetValidCol()`、`bMatrix.GetValidCol()`）必须在 `[1, 4095]` 范围内。

## 异常与非法情形

- 当 `m/k/n` 超出 `[1, 4095]` 范围时行为未定义。
- 当数据类型组合不符合目标支持的三元组时行为未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| int8_t GEMM | ✓ | ✓ | ✗ |
| float GEMM | ✓ | ✓ | ✓ |
| bfloat16 GEMM | ✓ | ✓ | ✓ |
| 分形布局约束 | ✗ | ✗ | ✓ |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  C c;
  TMATMUL(c, a, b);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  C c;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(c, 0x3000);
  TMATMUL(c, a, b);
}
```

### PTO-AS

```text
%acc = tmatmul %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

AS Level 2 (DPS)：

```mlir
pto.tmatmul ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

![TMATMUL tile operation](../figures/isa/TMATMUL.svg)

## 相关页面

- 指令集总览：[矩阵与矩阵-向量运算](./tile/ops/matrix-and-matrix-vector/tmatmul_zh.md)
