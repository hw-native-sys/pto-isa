# TSTORE


## Tile Operation Diagram

![TSTORE tile operation](../figures/isa/TSTORE.svg)

## Introduction

Store data from a Tile into a GlobalTensor (GM), optionally using atomic write or quantization parameters.

## Math Interpretation

Notation depends on the `GlobalTensor` shape/stride and the `Tile` layout. Conceptually (2D view, with a base offset):

$$ \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{src}_{i,j} $$

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

Synchronous form:

```text
tstore %t1, %sv_out[%c0, %c0]
```

### IR Level 1 (SSA)

```text
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### IR Level 2 (DPS)

```text
pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events);

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, uint64_t preQuantScalar, WaitEvents&... events);

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events);
```

## Constraints

- **Implementation checks (A2A3)**:
  - Source tile location must be one of: `TileType::Vec`, `TileType::Mat`, `TileType::Acc`.
  - Runtime: all `dst.GetShape(dim)` values and `src.GetValidRow()/GetValidCol()` must be `> 0`.
  - For `TileType::Vec` / `TileType::Mat`:
    - `TileData::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
    - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
    - Layouts must match ND/DN/NZ (or a special case where `TileData::Rows == 1` or `TileData::Cols == 1`).
    - For `int64_t/uint64_t`, only ND->ND or DN->DN are supported.
  - For `TileType::Acc` (including quantized/atomic variants):
    - Destination layout must be ND or NZ.
    - Source dtype must be `int32_t` or `float`.
    - When not using quantization, destination dtype must be `__gm__ int32_t/float/half/bfloat16_t`.
    - Static shape constraints: `1 <= TileData::Cols <= 4095`; if ND then `1 <= TileData::Rows <= 8192`; if NZ then `1 <= TileData::Rows <= 65535` and `TileData::Cols % 16 == 0`.
    - Runtime: `1 <= src.GetValidCol() <= 4095`.
- **Implementation checks (A5)**:
  - Source tile location must be `TileType::Vec` or `TileType::Acc` (no `Mat` store on this target).
  - For `TileType::Vec`:
    - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
    - `TileData::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`, `float8_e4m3_t`, `float8_e5m2_t`, `hifloat8_t`, `float4_e1m2x2_t`, `float4_e2m1x2_t`.
    - Layouts must match ND/DN/NZ (or a special case where `TileData::Rows == 1` or `TileData::Cols == 1`).
    - Additional alignment constraints are enforced (e.g., for ND the row-major width in bytes must be a multiple of 32; for DN the column-major height in bytes must be a multiple of 32, with special-case exceptions).
  - For `TileType::Acc`:
    - Destination layout must be ND or NZ; source dtype must be `int32_t` or `float`.
    - When not using quantization, destination dtype must be `__gm__ int32_t/float/half/bfloat16_t`.
    - Static shape constraints match A2A3 for rows/cols; `AtomicAdd` additionally restricts destination dtype to supported atomic types.
- **Valid region**:
  - The implementation uses `src.GetValidRow()` / `src.GetValidCol()` as the transfer size.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_auto(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TSTORE(gout, t);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_manual(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TASSIGN(t, 0x1000);
  TSTORE<TileT, GTensor, AtomicType::AtomicAdd>(gout, t);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### PTO Assembly Form

```text
tstore %t1, %sv_out[%c0, %c0]
# IR Level 2 (DPS)
pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

