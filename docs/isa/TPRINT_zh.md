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

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

```text
tprint %src : !pto.tile<...> | !pto.global<...>
```

### IR Level 1（SSA）

```text
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
```

### IR Level 2（DPS）

```text
pto.tprint ins(%src : !pto.tile_buf<...> | !pto.partition_tensor_view<MxNxdtype>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：
```cpp
template <typename T, typename... WaitEvents>
PTO_INST RecordEvent TPRINT(T &src, WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TPRINT, src);
  return {};
}
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

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
```

### PTO 汇编形式

```text
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
# IR Level 2 (DPS)
pto.tprint ins(%src : !pto.tile_buf<...> | !pto.partition_tensor_view<MxNxdtype>)
```

