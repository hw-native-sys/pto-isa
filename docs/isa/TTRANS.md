# TTRANS


## Tile Operation Diagram

![TTRANS tile operation](../figures/isa/TTRANS.svg)

## Introduction

Transpose with an implementation-defined temporary tile.

## Math Interpretation

For a 2D tile, over the effective transpose domain:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{j,i} $$

Exact shape/layout and the transpose domain depend on the target (see Constraints).

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

Synchronous form:

```text
%dst = ttrans %src : !pto.tile<...> -> !pto.tile<...>
```
Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### IR Level 1 (SSA)

```text
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TTRANS(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp, WaitEvents&... events);
```

## Constraints

- **Implementation checks (A2A3)**:
  - `sizeof(TileDataSrc::DType) == sizeof(TileDataDst::DType)`.
  - Source layout must be row-major (`TileDataSrc::isRowMajor`).
  - Element size must be `1`, `2`, or `4` bytes.
  - Supported element types are restricted per element width:
    - 4 bytes: `uint32_t`, `int32_t`, `float`
    - 2 bytes: `uint16_t`, `int16_t`, `half`, `bfloat16_t`
    - 1 byte: `uint8_t`, `int8_t`
  - The transpose size is taken from `src.GetValidRow()` / `src.GetValidCol()`.
- **Implementation checks (A5)**:
  - `sizeof(TileDataSrc::DType) == sizeof(TileDataDst::DType)`.
  - 32-byte alignment constraints are enforced on the major dimension of both input and output (row-major checks `Cols * sizeof(T) % 32 == 0`, col-major checks `Rows * sizeof(T) % 32 == 0`).
  - Supported element types are restricted per element width:
    - 4 bytes: `uint32_t`, `int32_t`, `float`
    - 2 bytes: `uint16_t`, `int16_t`, `half`, `bfloat16_t`
    - 1 byte: `uint8_t`, `int8_t`
  - The implementation operates over the static tile shape (`TileDataSrc::Rows/Cols`) and does not consult `GetValidRow/GetValidCol`.
- **Temporary tile**:
  - The C++ API requires `tmp`, but some implementations may not use it.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TTRANS(dst, src, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TTRANS(dst, src, tmp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = ttrans %src : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

