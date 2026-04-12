<!-- Generated from `docs/isa/tile/ops/matrix-and-matrix-vector/tgemv-bias_zh.md` -->

# TGEMV_BIAS

## 指令示意图

![TGEMV_BIAS tile operation](../figures/isa/TGEMV_BIAS.svg)

## 简介

带偏置加法的 GEMV。

## 另请参见

- 基础 GEMV 指令：`docs/isa/TGEMV.md`。
- 累加变体：`docs/isa/TGEMV_ACC.md`。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData, WaitEvents &... events);

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename TileBias,
          typename... WaitEvents>
PTO_INST RecordEvent TGEMV_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData, WaitEvents &... events);
```

## 数学语义

设：

- `M = 1`
- `K = bMatrix.GetValidRow()`
- `N = bMatrix.GetValidCol()`

对于 `0 <= j < N`（将偏置项加入矩阵乘积）：

$$ \mathrm{C}_{0,j} = \mathrm{Bias}_{0,j} + \sum_{k=0}^{K-1} \mathrm{A}_{0,k} \cdot \mathrm{B}_{k,j} $$

**注意：** 精确的累加器行为和数据类型提升由目标/实现定义。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%acc = tgemv.bias %a, %b, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%c = pto.tgemv.bias %a, %b, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tgemv.bias ins(%a, %b, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```

## 约束

### 通用形状与位置约束

- 静态形状约束：
    - `TileLeft::Rows == TileRes::Rows`
    - `TileLeft::Cols == TileRight::Rows`
    - `TileRight::Cols == TileRes::Cols`
- Tile 位置约束：
    - `TileLeft::Loc == Left`
    - `TileRight::Loc == Right`
    - `TileRes::Loc == Acc`
- 运行时有效尺寸约束：
    - `m` 必须为 `1`
    - `k` 和 `n`（取自 `bMatrix.GetValidRow()` 与 `bMatrix.GetValidCol()`）必须位于 `[1, 4095]`

### 数据类型约束

- **实现检查 (A2A3)**:
    - 支持的 `(CType, AType, BType)` 三元组：
        - `(int32_t, int8_t, int8_t)`
        - `(float, half, half)`
        - `(float, float, float)`
        - `(float, bfloat16_t, bfloat16_t)`
- **实现检查 (A5)**:
    - 累加器类型必须是 `int32_t` 或 `float`。
    - 如果为 `int32_t`：`AType == int8_t` 且 `BType == int8_t`。
    - 如果为 `float`：支持 `half`、`bfloat16_t`、`float` 以及选定的 fp8 组合（目标定义）。
    - 会强制执行以下分形/布局约束：
        - Left：`Loc == Left`、`!isRowMajor`、`SFractal == RowMajor`
        - Right：`Loc == Right`、`isRowMajor`、`SFractal == ColMajor`
        - Acc：`Loc == Acc`、`!isRowMajor`、`SFractal == RowMajor`

### 偏置专属约束

- 偏置 tile 的数据类型必须与 `TileRes::DType` 完全一致。
- 偏置 tile 必须配置为单行。
- 偏置 tile 的位置必须为 `TileType::Bias`。
- **A5 附加说明**：
    - 除上述 GEMV 约定外，底层 A5 matmul 实现不会再单独补充一组显式的 `m/k/n` 运行时断言。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using A = TileLeft<half, 1, 16>;
  using B = TileRight<half, 16, 16>;
  using Bias = Tile<TileType::Bias, half, 1, 16>;
  using C = TileAcc<float, 1, 16>;
  A a;
  B b;
  Bias bias;
  C c;
  TGEMV_BIAS(c, a, b, bias);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using A = TileLeft<half, 1, 16>;
  using B = TileRight<half, 16, 16>;
  using Bias = Tile<TileType::Bias, half, 1, 16>;
  using C = TileAcc<float, 1, 16>;
  A a;
  B b;
  Bias bias;
  C c;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(bias, 0x3000);
  TASSIGN(c, 0x4000);
  TGEMV_BIAS(c, a, b, bias);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%c = pto.tgemv.bias %a, %b, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%c = pto.tgemv.bias %a, %b, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%acc = tgemv.bias %a, %b, %bias : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tgemv.bias ins(%a, %b, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%c : !pto.tile_buf<...>)
```
