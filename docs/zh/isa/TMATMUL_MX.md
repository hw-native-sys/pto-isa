# TMATMUL_MX

## 产品支持情况<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->


## 指令示意图

![TMATMUL_MX tile operation](../../figures/isa/TMATMUL_MX.svg)

## 简介

带额外缩放Tile的矩阵乘法 (GEMM)，用于支持目标上的混合精度/量化矩阵乘法。

该指令目前仅在 Ascend 950PR/Ascend 950DT 和 Ascend 960 上实现（参见
`include/pto/npu/a5/TMatmul.hpp` 和 `include/pto/npu/a6/TMatmul.hpp`）。
带 `cInMatrix` 的累加重载为原地累加：`cOutMatrix` 和 `cInMatrix` 必须指向同一个累加器 Tile，运算结果写回该 Tile。

## 数学语义

设：

- `M = aMatrix.GetValidRow()`
- `K = aMatrix.GetValidCol()`
- `N = bMatrix.GetValidCol()`

概念上，结果对应于有效矩阵乘法域（`0 <= i < M`，`0 <= j < N`）上的矩阵乘法，缩放tile `aScaleMatrix` / `bScaleMatrix` 配置实现定义的混合精度行为：

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

累加重载对应：

$$ \mathrm{C}_{i,j} \gets \mathrm{C}_{i,j} + \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

`aScaleMatrix` / `bScaleMatrix` 的确切作用（以及任何反量化/量化语义）由目标定义。

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events);

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events);

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix, TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &... events);
```

带 `cInMatrix` 的累加重载要求 `cOutMatrix.data() == cInMatrix.data()`；底层指令只接收一个 C 矩阵地址，累加输入和输出必须使用同一个 Tile 或同一地址别名。

## 约束

- **实现检查 (Ascend 950PR/Ascend 950DT)**:
    - `m/k/n` 取自 `aMatrix.GetValidRow()`、`aMatrix.GetValidCol()`、`bMatrix.GetValidCol()`。
    - 静态合法性检查通过 `CheckMadMxValid<...>()`（类型、形状、分形和缩放tile合法性）。
    - 支持的 `(C, A, B)` 三元组（`C` 始终为 `float`；缩放tile为 `float8_e8m0_t`）：
        - FP8：`(float, float8_e4m3_t, float8_e4m3_t)`、`(float, float8_e4m3_t, float8_e5m2_t)`、`(float, float8_e5m2_t, float8_e4m3_t)`、`(float, float8_e5m2_t, float8_e5m2_t)`。
        - FP4：`(float, float4_e1m2x2_t, float4_e1m2x2_t)`、`(float, float4_e1m2x2_t, float4_e2m1x2_t)`、`(float, float4_e2m1x2_t, float4_e2m1x2_t)`、`(float, float4_e2m1x2_t, float4_e1m2x2_t)`。
- **偏置形式**:
    - `TileBias::DType` 必须是 `float` 且 `TileBias::Loc == TileType::Bias`，`TileBias::Rows == 1`（Ascend 950PR/Ascend 950DT通过 `static_assert` 检查）。
- **累加形式**:
    - `cOutMatrix.data() == cInMatrix.data()`；累加输入和输出必须是同一个 Tile 或同一地址别名。

## 示例

### 自动（Auto）

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

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

constexpr int kTileNum = 4;

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
  for (int i = 0; i < kTileNum; ++i) {
    if (i == 0) {
      TMATMUL_MX(c, a, scaleA, b, scaleB, bias);
    } else {
      TMATMUL_MX(c, c, a, scaleA, b, scaleB);
    }
  }
}
```
