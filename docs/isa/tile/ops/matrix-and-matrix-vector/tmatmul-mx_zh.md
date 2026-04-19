# pto.tmatmul.mx

`pto.tmatmul.mx` 属于[矩阵与矩阵-向量](../../matrix-and-matrix-vector_zh.md)指令集。

## 概述

`TMATMUL_MX` 是带双 scale Tile 的矩阵乘法扩展，用来表达 MX 路径下的混合精度/量化 GEMM。它和普通 `TMATMUL` 共享 Left / Right / Acc 的整体结构，但额外携带 `aScaleMatrix` 和 `bScaleMatrix`。这条指令存在的意义，是把"矩阵乘法本体"和"MX 重建/缩放参数"同时放进一条架构可见指令里，而不是靠外部约定隐式拼接。

设 `M = aMatrix.GetValidRow()`，`K = aMatrix.GetValidCol()`，`N = bMatrix.GetValidCol()`，基础乘法域仍然是：

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

区别在于，`aScaleMatrix` 和 `bScaleMatrix` 会参与 MX 路径下的重建/缩放。它们如何参与，不是通用 PTO 规则，而是目标定义的 MX 语义。

## 机制

当前仓库里，只有 A5 backend 真正实现了 MX 语义。CPU 模拟器会接受 `TMATMUL_MX` 接口，但当前实现会忽略 `aScaleMatrix` / `bScaleMatrix`，直接退化为普通 `TMATMUL` / `TMATMUL_ACC` / `TMATMUL_BIAS`。Kirin9030 当前明确不支持 `TMATMUL_MX`，对应实现直接 `static_assert` 失败。因此，如果要验证真正的 MX 语义，应以 A5 为准；CPU 只能用来跑接口形态或近似流程，不适合作为 MX 数值语义的最终依据。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

示意形式：

```text
%c = tmatmul.mx %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%c_out = tmatmul.mx.acc %c_in, %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%c = tmatmul.mx.bias %a, %a_scale, %b, %b_scale, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%c = pto.tmatmul.mx %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%c_out = pto.tmatmul.mx.acc %c_in, %a, %a_scale, %b, %b_scale : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%c = pto.tmatmul.mx.bias %a, %a_scale, %b, %b_scale, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tmatmul.mx ins(%a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
pto.tmatmul.mx.acc ins(%c_in, %a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c_out : !pto.tile_buf<...>)
pto.tmatmul.mx.bias ins(%a, %a_scale, %b, %b_scale, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                                TileRightScale &bScaleMatrix, WaitEvents &... events);

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix,
                                TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events);

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                                TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &... events);
```

`AccPhase` 的模板重载与普通 `TMATMUL` 一样，主要用于目标实现侧的 unit-flag 选择。

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
| `cMatrix` | `Acc` | MX 路径下的重建/缩放结果 |

## 副作用

结果写入 `cMatrix` 累加器。

## 约束

- A5 的 `CheckMadMxValid(...)` 要求：结果累加器类型必须是 `float`；输入必须是受支持的 fp4 或 fp8 组合；`TileLeft::Cols` 必须是 `64` 的倍数；若走 fp4 路径，`TileLeft::Cols` 还必须是偶数；Left / Right / Acc 的位置与 fractal 方向必须符合 cube 路径要求。
- 支持的 fp4 组合：`float4_e1m2x2_t / float4_e1m2x2_t`、`float4_e1m2x2_t / float4_e2m1x2_t`、`float4_e2m1x2_t / float4_e2m1x2_t`、`float4_e2m1x2_t / float4_e1m2x2_t`。
- 支持的 fp8 组合：`float8_e4m3_t / float8_e4m3_t`、`float8_e4m3_t / float8_e5m2_t`、`float8_e5m2_t / float8_e4m3_t`、`float8_e5m2_t / float8_e5m2_t`。
- Bias 变体还要求 `biasData` 的元素类型必须是 `float` 且为单行 `TileType::Bias`。
- A5 的 `m/k/n` 均必须落在 `[1, 4095]`。
- CPU 模拟器会接受接口但忽略 scale，退化为普通 `TMATMUL` / `TMATMUL_ACC` / `TMATMUL_BIAS`。
- Kirin9030 明确不支持 `TMATMUL_MX`，对应实现直接 `static_assert` 失败。

## 异常与非法情形

- 违反 `CheckMadMxValid(...)` 约束。
- 使用不支持的 fp4/fp8 组合。
- `TileLeft::Cols` 不是 `64` 的倍数（fp4 路径下还须为偶数）。
- 在不支持 MX 语义的 target（Kirin9030）上使用。
- m、k 或 n 超出 `[1, 4095]` 范围。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| MX 语义 | 退化（忽略 scale） | 不支持 | 支持 |
| fp4 量化 | 不支持 | 不支持 | 支持 |
| fp8 量化 | 不支持 | 不支持 | 支持 |
| MX bias | 不支持 | 不支持 | 支持 |
| Kirin9030 | 不支持 | 不支持 | 不支持 |

## 示例

### C++ 自动模式（普通）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using A = TileLeft<float8_e5m2_t, 16, 64>;
  using B = TileRight<float8_e5m2_t, 64, 32>;
  using ScaleA = TileLeftScale<float8_e8m0_t, 16, 2>;
  using ScaleB = TileRightScale<float8_e8m0_t, 2, 32>;
  using Bias = Tile<TileType::Bias, float, 1, 32>;
  using C = TileAcc<float, 16, 32>;
  A a;
  B b;
  ScaleA scaleA;
  ScaleB scaleB;
  Bias bias;
  C c;
  TMATMUL_MX(c, a, scaleA, b, scaleB, bias);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using A = TileLeft<float8_e5m2_t, 16, 64>;
  using B = TileRight<float8_e5m2_t, 64, 32>;
  using ScaleA = TileLeftScale<float8_e8m0_t, 16, 2>;
  using ScaleB = TileRightScale<float8_e8m0_t, 2, 32>;
  using Bias = Tile<TileType::Bias, float, 1, 32>;
  using C = TileAcc<float, 16, 32>;
  A a;
  B b;
  ScaleA scaleA;
  ScaleB scaleB;
  Bias bias;
  C c;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(scaleA, GetScaleAddr(a.data()));
  TASSIGN(scaleB, GetScaleAddr(b.data()));
  TASSIGN(bias, 0x3000);
  TASSIGN(c, 0x4000);
  TMATMUL_MX(c, a, scaleA, b, scaleB, bias);
}
```

## 相关页面

- [TGEMV_MX](./tgemv-mx_zh.md)
- [TMATMUL](./tmatmul_zh.md)
- [TMATMUL_ACC](./tmatmul-acc_zh.md)
- [TMATMUL_BIAS](./tmatmul-bias_zh.md)
- 指令集总览：[矩阵与矩阵-向量](../../matrix-and-matrix-vector_zh.md)
