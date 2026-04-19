# pto.tgemv.mx

`pto.tgemv.mx` 属于[矩阵与矩阵-向量运算](./tile/ops/matrix-and-matrix-vector/tgemv-mx_zh.md)指令集。

## 概述

带缩放 Tile 的 GEMV 变体，支持混合精度/量化矩阵向量计算。缩放 tile 参与实现定义的混合精度重建/缩放，输出对应于目标定义的 mx GEMV 语义。

## 机制

概念上（基础 GEMV 路径）：

$$ \mathrm{C}_{0,j} = \sum_{k=0}^{K-1} \mathrm{A}_{0,k} \cdot \mathrm{B}_{k,j} $$

对于 `TGEMV_MX`，缩放 tile 参与实现定义的混合精度重建/缩放。架构约定是输出对应于目标定义的 mx GEMV 语义。

## 语法

### PTO-AS

示意形式：

```text
%acc = tgemv.mx %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%acc = pto.tgemv.mx %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tgemv.mx ins(%a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%acc : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events);

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events);

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &... events);
```

附加重载支持累加/偏置变体和 `AccPhase` 选择。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `aMatrix` | Left | 左操作数向量 Tile |
| `aScaleMatrix` | LeftScale | 左操作数缩放 Tile |
| `bMatrix` | Right | 右操作数矩阵 Tile |
| `bScaleMatrix` | RightScale | 右操作数缩放 Tile |
| `biasData` | Bias | 偏置 Tile（偏置形式可选） |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `cMatrix` / `cOutMatrix` | Acc | MX GEMV 结果 Tile |

## 副作用

MX 混合精度操作可能触发目标特定的缩放、反量化或溢出处理行为。

## 约束

- 使用后端特定的 mx 合法性检查，用于数据类型、tile 位置、分形/布局组合以及缩放格式。
- 缩放 tile 兼容性和累加器提升由目标后端的实现定义。
- 为了可移植性，请根据目标实现约束验证确切的 `(A, B, scaleA, scaleB, C)` 类型元组和 tile 布局。

## 异常与非法情形

- 当缩放 tile 类型或布局不符合目标要求时行为未定义。
- 当数据类型组合不符合目标支持的 mx 规格时行为未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| MX GEMV | ✓ | ✗ | ✓ |
| MX GEMV 累加 | ✓ | ✗ | ✓ |
| MX GEMV 偏置 | ✓ | ✗ | ✓ |

## 示例

### C++ 自动模式

实际使用模式请参见：
- `docs/isa/TMATMUL_MX.md`
- `docs/isa/TGEMV.md`

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using A = TileLeft<float8_e5m2_t, 1, 64>;
  using B = TileRight<float8_e5m2_t, 64, 32>;
  using ScaleA = TileLeftScale<float8_e8m0_t, 1, 2>;
  using ScaleB = TileRightScale<float8_e8m0_t, 2, 32>;
  using C = TileAcc<float, 1, 32>;
  A a;
  B b;
  ScaleA scaleA;
  ScaleB scaleB;
  C c;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(scaleA, GetScaleAddr(a.data()));
  TASSIGN(scaleB, GetScaleAddr(b.data()));
  TASSIGN(c, 0x3000);
  TGEMV_MX(c, a, scaleA, b, scaleB);
}
```

### PTO-AS

```text
%acc = tgemv.mx %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

AS Level 2 (DPS)：

```mlir
pto.tgemv.mx ins(%a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%acc : !pto.tile_buf<...>)
```

![TGEMV_MX tile operation](../figures/isa/TGEMV_MX.svg)

## 相关页面

- 指令集总览：[矩阵与矩阵-向量运算](./tile/ops/matrix-and-matrix-vector/tgemv-mx_zh.md)
