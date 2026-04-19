# pto.tgemv.mx

`pto.tgemv.mx` 属于[矩阵与矩阵-向量](../../matrix-and-matrix-vector_zh.md)指令集。

## 概述

`TGEMV_MX` 是 MX 路径下的矩阵向量乘版本。它和 `TMATMUL_MX` 共享同一套 scale Tile 思路，只是把乘法域收窄到 GEMV 形态（`m = 1`）。从接口上看，这条指令同样支持普通、累加和 bias 三种变体。GEMV 基础乘法域可以写成：

$$ \mathrm{C}_{0,j} = \sum_{k=0}^{K-1} \mathrm{A}_{0,k} \cdot \mathrm{B}_{k,j} $$

`aScaleMatrix` / `bScaleMatrix` 会参与 MX 路径下的重建/缩放；它们的精确作用由目标定义，而不是由这页单独给出一套独立数值规则。

## 机制

`TGEMV_MX` 复用 `TMATMUL_MX` 相同的 scale Tile 机制，只是固定 GEMV 语义使用单行输出：A5 的 `aScaleMatrix` / `bScaleMatrix` 参与 MX 重建/缩放，精确数值语义由目标 backend 定义。CPU 模拟器当前会忽略 `aScaleMatrix` / `bScaleMatrix`，退化为普通 `TGEMV` / `TGEMV_ACC` / `TGEMV_BIAS`。Kirin9030 当前没有 MX 实现路径。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

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
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                              TileRightScale &bScaleMatrix, WaitEvents &... events);

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix,
                              TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events);

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                              TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `aMatrix` | Left | 左操作数 tile |
| `aScaleMatrix` | LeftScale | 左缩放 tile |
| `bMatrix` | Right | 右操作数 tile |
| `bScaleMatrix` | RightScale | 右缩放 tile |
| `biasData` | Bias | 偏置 tile（仅 bias 变体） |
| `cMatrix` | Acc | 结果累加器 tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `cMatrix` | `Acc` | MX 路径下的重建/缩放结果，输出为一行 |

## 副作用

结果写入 `cMatrix` 累加器。

## 约束

- `TGEMV_MX` 复用与 `TMATMUL_MX` 相同的 `CheckMadMxValid(...)` 约束：结果累加器必须是 `float`；输入必须是受支持的 fp4 或 fp8 组合；Left / Right / Acc 的位置与 fractal 方向必须合法。
- 运行时固定 GEMV 语义使用单行输出：`k = aMatrix.GetValidCol()`、`n = bMatrix.GetValidCol()`，均必须落在 `[1, 4095]`。
- Bias 变体要求 `biasData` 元素类型为 `float` 且为单行 `TileType::Bias`。
- CPU 模拟器当前会忽略 `aScaleMatrix` / `bScaleMatrix`，退化为普通 `TGEMV` / `TGEMV_ACC` / `TGEMV_BIAS`。
- Kirin9030 当前没有 MX 实现路径。

## 异常与非法情形

- 违反 `CheckMadMxValid(...)` 约束。
- 在不支持 MX 语义的 target 上使用。
- k 或 n 超出 `[1, 4095]` 范围。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| MX 语义 | 退化（忽略 scale） | 不支持 | 支持 |
| fp4/fp8 量化 | 不支持 | 不支持 | 支持 |

## 示例

### C++ 普通模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
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
  TGEMV_MX(c, a, scaleA, b, scaleB);
}
```

### C++ 累加模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_acc() {
  using A = TileLeft<float8_e5m2_t, 1, 64>;
  using B = TileRight<float8_e5m2_t, 64, 32>;
  using ScaleA = TileLeftScale<float8_e8m0_t, 1, 2>;
  using ScaleB = TileRightScale<float8_e8m0_t, 2, 32>;
  using C = TileAcc<float, 1, 32>;

  A a;
  B b;
  ScaleA scaleA;
  ScaleB scaleB;
  C acc;
  TGEMV_MX(acc, acc, a, scaleA, b, scaleB);
}
```

### C++ Bias 模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_bias() {
  using A = TileLeft<float8_e5m2_t, 1, 64>;
  using B = TileRight<float8_e5m2_t, 64, 32>;
  using ScaleA = TileLeftScale<float8_e8m0_t, 1, 2>;
  using ScaleB = TileRightScale<float8_e8m0_t, 2, 32>;
  using Bias = Tile<TileType::Bias, float, 1, 32>;
  using C = TileAcc<float, 1, 32>;

  A a;
  B b;
  ScaleA scaleA;
  ScaleB scaleB;
  Bias bias;
  C c;
  TGEMV_MX(c, a, scaleA, b, scaleB, bias);
}
```

## 相关页面

- [TMATMUL_MX](./tmatmul-mx_zh.md)
- [TGEMV](./tgemv_zh.md)
- 指令集总览：[矩阵与矩阵-向量](../../matrix-and-matrix-vector_zh.md)
