# TDIVS

## 指令示意图

![TDIVS tile operation](../figures/isa/TDIVS.svg)

## 简介

与标量的逐元素除法（Tile/标量 或 标量/Tile）。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

- Tile/scalar:

  $$ \mathrm{dst}_{i,j} = \frac{\mathrm{src}_{i,j}}{\mathrm{scalar}} $$

- Scalar/tile:

  $$ \mathrm{dst}_{i,j} = \frac{\mathrm{scalar}}{\mathrm{src}_{i,j}} $$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

Tile/scalar form:

```text
%dst = tdivs %src, %scalar : !pto.tile<...>, f32
```

Scalar/tile form:

```text
%dst = tdivs %scalar, %src : f32, !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tdivs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst = pto.tdivs %scalar, %src : (dtype, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
pto.tdivs ins(%scalar, %src : dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileData& dst, TileData& src0, typename TileData::DType scalar, WaitEvents&... events);

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileData& dst, typename TileData::DType scalar, TileData& src0, WaitEvents&... events);
```

## 约束

- **实现检查 (A2A3)** (both overloads):
  - `TileData::DType` must be one of: `int32_t`, `int`, `int16_t`, `half`, `float16_t`, `float`, `float32_t`.
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
  - Runtime: `src0.GetValidRow() == dst.GetValidRow()` and `src0.GetValidCol() == dst.GetValidCol()`.
  - Tile 布局 must be row-major (`TileData::isRowMajor`).
- **实现检查 (A5)** (both overloads):
  - `TileData::DType` must be one of: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, `float`.
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
  - Runtime: `src0.GetValidRow() == dst.GetValidRow()` and `src0.GetValidCol() == dst.GetValidCol()`.
  - Tile 布局 must be row-major (`TileData::isRowMajor`).
- **有效区域**:
  - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.
- **Division-by-zero**:
  - Behavior is target-defined; on A5 the tile/scalar form maps to multiply-by-reciprocal and uses `1/0 -> +inf` for `scalar == 0`.

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TDIVS(dst, src, 2.0f);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TDIVS(dst, 2.0f, src);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tdivs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tdivs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = pto.tdivs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

