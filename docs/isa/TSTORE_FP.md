# TSTORE_FP


## Tile Operation Diagram

![TSTORE_FP tile operation](../figures/isa/TSTORE_FP.svg)

## Introduction

Store an accumulator tile into global memory using a scaling (`fp`) tile for vector quantization parameters.

`TSTORE_FP` is the fp-quantization overload of `TSTORE` (see `docs/isa/TSTORE.md`).

## Math Interpretation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. Conceptually (2D view, with a base offset), for `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{Convert}\!\left(\mathrm{src}_{i,j};\ \mathrm{fp}\right) $$

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

Synchronous form:

```text
tstore.fp %src, %fp, %sv_out[%c0, %c0]
```

### AS Level 1 (SSA)

```text
pto.tstore.fp %src, %fp, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### AS Level 2 (DPS)

```text
pto.tstore.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData &dst, TileData &src, FpTileData &fp, WaitEvents &... events);
```

## Constraints

- **Implementation checks (A2A3)**:
    - The fp store path is implemented via `TSTORE_IMPL(dst, src, fp)` and uses the same accumulator-to-GM legality checks as quantized accumulator stores:
    - Destination layout must be ND or NZ.
    - Source dtype must be `int32_t` or `float`.
    - Static shape constraints: `1 <= TileData::Cols <= 4095`; if ND then `1 <= TileData::Rows <= 8192`; if NZ then `1 <= TileData::Rows <= 65535` and `TileData::Cols % 16 == 0`.
    - Runtime: `1 <= src.GetValidCol() <= 4095`.
    - No explicit `static_assert` is enforced on `FpTileData` (the implementation uses `fp` to set FPC state).
- **Implementation checks (A5)**:
    - Implemented via `TSTORE_IMPL(dst, src, fp)` and validated by `CheckStaticAcc<..., true>()` for the accumulator path (ND/NZ only, `int32_t/float` source dtype, rows/cols ranges).
    - No explicit `static_assert` is enforced on `FpTileData` (the implementation uses `fp` to set FPC state).

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto(__gm__ int8_t* out) {
  using AccT = TileAcc<float, 16, 16>;
  using FpT = Tile<TileType::Scaling, uint64_t, 1, 16, BLayout::RowMajor, 1, DYNAMIC, SLayout::NoneBox>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<int8_t, 16, 16, Layout::ND>;
  using GT = GlobalTensor<int8_t, GShape, GStride, Layout::ND>;

  GT gout(out);
  AccT acc;
  FpT fp(16);
  TSTORE_FP(gout, acc, fp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual(__gm__ int8_t* out) {
  using AccT = TileAcc<float, 16, 16>;
  using FpT = Tile<TileType::Scaling, uint64_t, 1, 16, BLayout::RowMajor, 1, DYNAMIC, SLayout::NoneBox>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<int8_t, 16, 16, Layout::ND>;
  using GT = GlobalTensor<int8_t, GShape, GStride, Layout::ND>;

  GT gout(out);
  AccT acc;
  FpT fp(16);
  TASSIGN(acc, 0x1000);
  TASSIGN(fp,  0x2000);
  TSTORE_FP(gout, acc, fp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.tstore.fp %src, %fp, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tstore.fp %src, %fp, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### PTO Assembly Form

```text
tstore.fp %src, %fp, %sv_out[%c0, %c0]
# AS Level 2 (DPS)
pto.tstore.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

