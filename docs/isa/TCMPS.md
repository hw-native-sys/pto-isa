# TCMPS


## Tile Operation Diagram

![TCMPS tile operation](../figures/isa/TCMPS.svg)

## Introduction

Compare a tile against a **scalar** or another tile and write per-element comparison results. The Tile-form behavior is
backend-specific as described below.

Two overloads are provided:

- **Scalar form**: compare every element of `src0` against a scalar value.
- **Tile form**: A2/A3 compares every element of `src0` against the scalar from the first element of `src1`.
  A5 and CPU_SIM compare corresponding `src0[i,j]` and `src1[i,j]` elements instead.

## Math Interpretation

**Scalar form** — for each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \left(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{scalar}\right) $$

**Tile form on A2/A3** — for each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \left(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{src1}_{0,0}\right) $$

**Tile form on A5 and CPU_SIM** — for each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \left(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{src1}_{i,j}\right) $$

The encoding/type of `dst` is implementation-defined (a bit-packed mask tile where each bit represents one comparison result).

## Assembly Syntax

Synchronous form:

```text
%dst = tcmps %src, %scalar {cmpMode = #pto.cmp<EQ>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tcmps %src, %scalar {cmpMode = #pto<cmp xx>} : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tcmps ins(%src, %scalar{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/type.hpp`:

**Scalar form** — compare tile against a scalar:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents,
          std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TCMPS(TileDataDst& dst, TileDataSrc& src0,
                           typename TileDataSrc::DType src1, CmpMode mode,
                           WaitEvents&... events);
```

**Tile form** — compare tile against another tile (backend behavior described above):

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
          typename... WaitEvents,
          std::enable_if_t<is_tile_data_v<TileDataSrc1> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TCMPS(TileDataDst& dst, TileDataSrc0& src0,
                           TileDataSrc1& src1, CmpMode mode,
                           WaitEvents&... events);
```

## Constraints

- **Implementation checks (A2A3)**:
    - `TileData::DType` must be one of: `int32_t`, `float`, `half`, `uint16_t`, `int16_t`.
    - Tile layout must be row-major (`TileData::isRowMajor`).
    - For `int32_t` input, only `CmpMode::EQ` is supported; other comparison modes fall back to `EQ`.
- **Implementation checks (A5)**:
    - `TileData::DType` must be one of: `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `float`, `int16_t`, `uint16_t`, `half`, `uint8_t`, `int8_t`, `bfloat16_t`.
    - Tile layout must be row-major (`TileData::isRowMajor`).
- **Common constraints**:
    - Tile location must be vector (`TileData::Loc == TileType::Vec`) for both `src` and `dst`.
    - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
    - Data type: `src0` and `src1` must have the same data type.
    - On A5 and CPU_SIM, Tile-form `src0` and `src1` are compared element by element; A2/A3 broadcasts `src1[0,0]`.
- **Backend shape checks**:
    - A2/A3 and A5 assert that `src0` and `dst` have the same valid row count. They do not explicitly check the valid column counts.
    - CPU_SIM does not assert valid-shape equality. It evaluates the valid region of `src0`, bounds packed writes by the valid shape of `dst`, and expects Tile-form `src1` to provide the corresponding readable elements.
- **Valid region**:
    - The op uses `src0.GetValidRow()` / `src0.GetValidCol()` as the iteration domain.
- **Comparison modes**:
    - Supports `CmpMode::EQ`, `CmpMode::NE`, `CmpMode::LT`, `CmpMode::GT`, `CmpMode::LE`, `CmpMode::GE`.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src;
  DstT dst(16, 2);
  TCMPS(dst, src, 0.0f, CmpMode::GT);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src;
  DstT dst(16, 2);
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TCMPS(dst, src, 0.0f, CmpMode::GT);
}
```

### Tile Form (compare against another tile)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_tile() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src0, src1;
  DstT dst(16, 2);
  // A2/A3: broadcast src1[0,0]. A5 and CPU_SIM: compare corresponding elements.
  TCMPS(dst, src0, src1, CmpMode::GE);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tcmps %src, %scalar {cmpMode = #pto<cmp xx>} : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcmps %src, %scalar {cmpMode = #pto<cmp xx>} : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tcmps %src, %scalar {cmpMode = #pto.cmp<EQ>} : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcmps ins(%src, %scalar{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```
