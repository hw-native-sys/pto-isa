# pto.tmatmul.acc

`pto.tmatmul.acc` 属于[矩阵与矩阵-向量运算](./tile/ops/matrix-and-matrix-vector/tmatmul-acc_zh.md)指令集。

## 概述

带累加器输入的矩阵乘法（融合累加），在有效域 `0 <= i < M`、`0 <= j < N` 上计算 $\mathrm{C1}_{i,j} = \mathrm{C0}_{i,j} + \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j}$，其中 `M = aMatrix.GetValidRow()`、`K = aMatrix.GetValidCol()`、`N = bMatrix.GetValidCol()`。

## 机制

设：
- `M = aMatrix.GetValidRow()`
- `K = aMatrix.GetValidCol()`
- `N = bMatrix.GetValidCol()`

对于 `0 <= i < M` 和 `0 <= j < N`：

$$ \mathrm{C1}_{i,j} = \mathrm{C0}_{i,j} + \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

## 语法

### PTO-AS

同步形式：

```text
%acc1 = tmatmul.acc %acc0, %a, %b : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%c_out = pto.tmatmul.acc %c_in, %a, %b : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tmatmul.acc ins(%c_in, %a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c_out : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events);

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `cInMatrix` | Acc | 输入累加器 Tile |
| `aMatrix` | Left | 左操作数矩阵 Tile |
| `bMatrix` | Right | 右操作数矩阵 Tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `cOutMatrix` | Acc | 累加后的 GEMM 结果 Tile |

## 副作用

融合累加操作可能触发目标特定的溢出处理或舍入行为。

## 约束

- 所有来自 `TMATMUL` 的约束都适用于 `(cOutMatrix, aMatrix, bMatrix)` 三元组。
- 实现说明 (A2A3/A5):
    - `TMATMUL_ACC_IMPL` 使用 `aMatrix.GetValidRow()`、`aMatrix.GetValidCol()` 和 `bMatrix.GetValidCol()` 作为 `m/k/n`。
    - `cInMatrix` 在当前实现中不通过显式断言进行验证（目标定义的行为）。

## 异常与非法情形

- 当 `cInMatrix` 与 `cOutMatrix` 类型不匹配时行为未定义。
- 当 `m/k/n` 超出目标允许范围时行为未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 融合累加 | ✓ | ✓ | ✓ |

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
  C c0, c1;
  TMATMUL_ACC(c1, c0, a, b);
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
  C c0, c1;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(c0, 0x3000);
  TASSIGN(c1, 0x4000);
  TMATMUL_ACC(c1, c0, a, b);
}
```

### PTO-AS

```text
%acc1 = tmatmul.acc %acc0, %a, %b : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

AS Level 2 (DPS)：

```mlir
pto.tmatmul.acc ins(%c_in, %a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c_out : !pto.tile_buf<...>)
```

![TMATMUL_ACC tile operation](../figures/isa/TMATMUL_ACC.svg)

## 相关页面

- 指令集总览：[矩阵与矩阵-向量运算](./tile/ops/matrix-and-matrix-vector/tmatmul-acc_zh.md)
