# TCI


## Tile Operation Diagram

![TCI tile operation](../figures/isa/TCI.svg)

## Introduction

Generate a contiguous integer sequence into a destination tile.

## Math Interpretation

For a linearized index `k` over the valid elements:

- Ascending:

  $$ \mathrm{dst}_{k} = S + k $$

- Descending:

  $$ \mathrm{dst}_{k} = S - k $$

The linearization order depends on the tile layout (implementation-defined).

## Assembly Syntax

Synchronous form:

```text
%dst = tci %S {descending = false} : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData &dst, T start, WaitEvents &... events);

template <typename TileData, typename TileDataTmp, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData &dst, T start, TileDataTmp &tmp, WaitEvents &... events);
```

## Constraints

- **Implementation checks (A2A3/A5)**:
    - `TileData::DType` must be exactly the same type as the scalar template parameter `T`.
    - `dst/scalar` element types must be identical, and must be one of: `int32_t`, `uint32_t`, `int16_t`, `uint16_t`.
    - `TileData::Rows == 1` (this is the condition enforced by the implementation; the sequence is generated along the column direction).
- **Valid region**:
    - The implementation uses `dst.GetValidCol()` as the sequence length and does not consult `dst.GetValidRow()`.
- **Temporary tile**:
    - **A2A3**: The C++ API provides an overload with an explicit `tmp` tile for the vectorized implementation path. The no-`tmp` overload uses a scalar loop. `TileDataTmp::DType` must be a 4-byte type (`float`, `int32_t`, or `uint32_t`). The implementation casts `tmp` to `float *`; size the tile by bytes, independent of the declared `TileDataTmp::DType`.
    - **b32 element types** (`int32_t`, `uint32_t`): minimum tmp size = 768 bytes (192 float elements).
      The vectorized path uses two float sub-buffers within `tmp`: `tmp0` at offset 0 and `tmp1` at offset +128 floats. `tmp0` holds up to 64 float elements (256 bytes) for the initial fractional sequence, and `tmp1` holds up to 64 float elements (256 bytes) for the accumulated result. The highest accessed byte is offset 128 × 4 + 64 × 4 = **768 bytes** (192 float elements).
    - **b16 element types** (`int16_t`, `uint16_t`): minimum tmp size = 1792 bytes (448 float elements).
      The vectorized path uses four sub-buffers within `tmp`: `tmp0/tmp1` (float) at offsets 0 and +128, and `tmp2/tmp3` (half) at offsets +256 and +384 (in float-index units). `tmp0/tmp1` each hold up to 64 floats (256 bytes) for the fractional sequence generation. `tmp2` holds up to 16 half elements (32 bytes) for the float-to-half conversion. `tmp3` holds up to 128 half elements (256 bytes) for the final half-precision accumulation. The highest accessed byte is offset 384 × 4 + 128 × 2 = **1792 bytes** (448 float elements).
    - A convenient shape-independent allocation is 2048 bytes (2 KiB), e.g., `Tile<TileType::Vec, float, 1, 512>`.
    - **A5**: The `tmp` tile is accepted and ignored. A5 hardware uses the `vci` vector instruction directly without requiring a scratch buffer.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, int32_t, 1, 16>;
  TileT dst;
  TCI<TileT, int32_t, /*descending=*/0>(dst, /*S=*/0);
}
```

### Auto (with tmp)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto_tmp() {
  using TileT = Tile<TileType::Vec, int32_t, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 1, 512>;
  TileT dst;
  TmpT tmp;
  TCI<TileT, TmpT, int32_t, /*descending=*/0>(dst, /*S=*/0, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, int32_t, 1, 16>;
  TileT dst;
  TASSIGN(dst, 0x1000);
  TCI<TileT, int32_t, /*descending=*/1>(dst, /*S=*/100);
}
```

### Manual (with tmp)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual_tmp() {
  using TileT = Tile<TileType::Vec, int32_t, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 1, 512>;
  TileT dst;
  TmpT tmp;
  TASSIGN(dst, 0x1000);
  TASSIGN(tmp, 0x2000);
  TCI<TileT, TmpT, int32_t, /*descending=*/1>(dst, /*S=*/100, tmp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tci %S {descending = false} : !pto.tile<...>
# AS Level 2 (DPS)
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```

