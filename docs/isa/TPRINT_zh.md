# TPRINT

## 指令示意图

![TPRINT tile operation](../figures/isa/TPRINT.svg)

## 简介

调试/打印 Tile 中的元素（实现定义）。

Print the contents of a Tile or GlobalTensor for debugging purposes directly from device code.

The `TPRINT` instruction outputs the logical view of data stored in a Tile or GlobalTensor. It supports common data types (e.g., `float`, `half`, `int8`, `uint32`) and multiple memory layouts (`ND`, `DN`, `NZ` for GlobalTensor; vector tiles for on-chip buffers).

> **Important**:
> - This instruction is **for development and debugging ONLY**.
> - It incurs **significant runtime overhead** and **must not be used in production kernels**.
> - Output may be **truncated** if it exceeds the internal print buffer.
> - **Requires CCE compilation option `-D_DEBUG --cce-enable-print`** (see [Behavior](#behavior)).

## 数学语义

除非另有说明, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## 汇编语法

PTO-AS 形式：参见 [PTO-AS Specification](../assembly/PTO-AS.md).

```text
tprint %src : !pto.tile<...> | !pto.global<...>
```

### AS Level 1 (SSA)

```text
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
```

### AS Level 2 (DPS)

```text
pto.tprint ins(%src : !pto.tile_buf<...> | !pto.partition_tensor_view<MxNxdtype>)
```

### AS Level 1（SSA）

```text
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
```

### AS Level 2（DPS）

```text
pto.tprint ins(%src : !pto.tile_buf<...> | !pto.partition_tensor_view<MxNxdtype>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`:
```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPRINT(TileData &src, WaitEvents &... events);
```

### Supported Types for T
- **Tile**: Must be a vector tile (`TileType::Vec`) with supported element type.
- **GlobalTensor**: Must use layout `ND`, `DN`, or `NZ`, and have a supported element type.

## 约束

- **Supported element type**:
    - Floating-point: `float`, `half`
    - Signed integers: `int8_t`, `int16_t`, `int32_t`
    - Unsigned integers: `uint8_t`, `uint16_t`, `uint32_t`
- **For Tiles**: `TileData::Loc == TileType::Vec` (only vector tiles are printable).
- **For GlobalTensor**: Layout must be one of `Layout::ND`, `Layout::DN`, or `Layout::NZ`.

## 示例

### Print a Tile

```cpp
#include <pto/pto-inst.hpp>

PTO_INTERNAL void DebugTile(__gm__ float *src) {
  using ValidSrcShape = TileShape2D<float, 16, 16>;
  using NDSrcShape = BaseShape2D<float, 32, 32>;
  using GlobalDataSrc = GlobalTensor<float, ValidSrcShape, NDSrcShape>;
  GlobalDataSrc srcGlobal(src);

  using srcTileData = Tile<TileType::Vec, float, 16, 16>;
  srcTileData srcTile;
  TASSIGN(srcTile, 0x0);

  TLOAD(srcTile, srcGlobal);
  TPRINT(srcTile);
}
```

### Print a GlobalTensor

```cpp
#include <pto/pto-inst.hpp>

PTO_INTERNAL void DebugGlobalTensor(__gm__ float *src) {
  using ValidSrcShape = TileShape2D<float, 16, 16>;
  using NDSrcShape = BaseShape2D<float, 32, 32>;
  using GlobalDataSrc = GlobalTensor<float, ValidSrcShape, NDSrcShape>;
  GlobalDataSrc srcGlobal(src);

  TPRINT(srcGlobal);
}
```
