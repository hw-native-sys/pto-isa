# pto.tstore_fp

`pto.tstore_fp` is part of the [Memory And Data Movement](../../memory-and-data-movement.md) instruction set.

## Summary

Store an accumulator tile into global memory through the fix-pipe path.

## Mechanism

`TSTORE_FP` is the fix-pipe overload of [`pto.tstore`](./tstore.md). The `_fp` suffix means **fix pipe**, not floating point. The auxiliary `fp` tile is the sideband tile consumed by the backend `set_fpc(...)` path before the store executes.

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. Conceptually (2D view, with a base offset), for `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{Convert}\!\left(\mathrm{src}_{i,j};\ \mathrm{fp}\right) $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

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

### IR Level 1 (SSA)

```text
pto.tstore.fp %src, %fp, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### IR Level 2 (DPS)

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

## Inputs

- `src` is the source accumulator tile.
- `fp` is the auxiliary fix-pipe tile consumed by the backend FPC path.
- `dst` is the destination GlobalTensor.
- `reluPreMode` (optional): specifies ReLU pre-processing mode.

## Expected Outputs

`src` is written to `dst` through the fix-pipe path configured by the auxiliary `fp` tile.

## Side Effects

This operation writes to global memory and programs fix-pipe sideband state for the transfer.

## Constraints

- Addressing, layout, and transfer shape MUST satisfy the legality rules of the selected target profile.

- Programs must not assume hidden ordering stronger than the documented PTO synchronization rules.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- **Implementation checks (A2A3)**:
    - The fp store path is implemented via `TSTORE_IMPL(dst, src, fp)` and uses the same accumulator-to-GM legality checks as quantized accumulator stores:
    - Destination layout must be ND or NZ.
    - Source dtype must be `int32_t` or `float`.
    - Static shape constraints: `1 <= TileData::Cols <= 4095`; if ND then `1 <= TileData::Rows <= 8192`; if NZ then `1 <= TileData::Rows <= 65535` and `TileData::Cols % 16 == 0`.
    - Runtime: `1 <= src.GetValidCol() <= 4095`.
    - No explicit `static_assert` is enforced on `FpTileData`; the implementation uses `fp` to set FPC state.

- **Implementation checks (A5)**:
    - Implemented via `TSTORE_IMPL(dst, src, fp)` and validated by `CheckStaticAcc<..., true>()` for the accumulator path (ND/NZ only, `int32_t/float` source dtype, rows/cols ranges).
    - No explicit `static_assert` is enforced on `FpTileData`; the implementation uses `fp` to set FPC state.

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

## Related Ops / Instruction Set Links

- Instruction set overview: [Memory And Data Movement](../../memory-and-data-movement.md)
- Previous op in instruction set: [pto.tstore](./tstore.md)
- Next op in instruction set: [pto.mgather](./mgather.md)
