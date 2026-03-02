# TGATHER

## 指令示意图

![TGATHER tile operation](../figures/isa/TGATHER.svg)

## 简介

使用索引 Tile 或编译时掩码模式来收集/选择元素。

## 数学语义

基于索引的 gather（概念性定义）：

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}\!\left[\mathrm{indices}_{i,j}\right] $$

Exact index interpretation and bounds behavior are implementation-defined.

Mask-pattern gather is an implementation-defined selection/reduction controlled by `pto::MaskPattern`.

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

Index-based gather:

```text
%dst = tgather %src0, %indices : !pto.tile<...> -> !pto.tile<...>
```

基于掩码模式的 gather：

```text
%dst = tgather %src {maskPattern = #pto.mask_pattern<P0101>} : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%dst = pto.tgather %src {maskPattern = #pto.mask_pattern<P0101>}: !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tgather ins(%src, %indices : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<P0101>} : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp` and `include/pto/common/type.hpp`:

```cpp
template <typename TileDataD, typename TileDataS0, typename TileDataS1, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(TileDataD& dst, TileDataS0& src0, TileDataS1& src1, WaitEvents&... events);

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(DstTileData& dst, SrcTileData& src, WaitEvents&... events);
```

## 约束

- **Index-based gather: implementation checks (A2A3)**:
  - `sizeof(DstTileData::DType)` must be must be `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `float`.
  - `sizeof(Src1TileData::DType)` must be must be `int32_t`, `uint32_t`.
  - `DstTileData::DType` must be the same type as `Src0TileData::DType`.
  - `src1.GetValidCol() == Src1TileData::Cols` and `dst.GetValidCol() == DstTileData::Cols`.
- **Index-based gather: implementation checks (A5)**:
  - `sizeof(DstTileData::DType)` must be must be `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `float`.
  - `sizeof(Src1TileData::DType)` must be must be `int16_t`, `uint16_t`, `int32_t`, `uint32_t`.
  - `DstTileData::DType` must be the same type as `Src0TileData::DType`.
  - `src1.GetValidCol() == Src1TileData::Cols` and `dst.GetValidCol() == DstTileData::Cols`.
- **基于掩码模式的 gather： implementation checks (A2A3)**:
  - Source element size must be `2` or `4` bytes.
  - `SrcTileData::DType`/`DstTileData::DType` must be `int16_t` or `uint16_t` or `int32_t` or `uint32_t`
    or `half` or `bfloat16_t` or `float`.
  - `dst` and `src` must both be `TileType::Vec` and row-major.
  - `sizeof(dst element) == sizeof(src element)` and `dst.GetValidCol() == DstTileData::Cols` (continuous dst storage).
- **基于掩码模式的 gather： implementation checks (A5)**:
  - Source element size must be `1` or `2` or `4` bytes.
  - `dst` and `src` must both be `TileType::Vec` and row-major.
  - `SrcTileData::DType`/`DstTileData::DType` must be `int8_t` or `uint8_t` or `int16_t` or `uint16_t` or `int32_t` or `uint32_t`
    or `half` or `bfloat16_t` or `float` or `float8_e4m3_t`or `float8_e5m2_t` or `hifloat8_t`.
  - Supported dtypes are restricted to a target-defined set (checked via `static_assert` in the implementation), and `sizeof(dst element) == sizeof(src element)`, `dst.GetValidCol() == DstTileData::Cols` (continuous dst storage).
- **边界 / 有效性**:
  - Index bounds are not validated by explicit runtime assertions; out-of-range indices are target-defined.

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, int32_t, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src0;
  IdxT idx;
  DstT dst;
  TGATHER(dst, src0, idx);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TGATHER<DstT, SrcT, MaskPattern::P0101>(dst, src);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tgather ins(%src, %indices : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

