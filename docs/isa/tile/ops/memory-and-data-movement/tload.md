# pto.tload

`pto.tload` is part of the [Memory And Data Movement](../../memory-and-data-movement.md) instruction set.

## Summary

Load data from global memory into a tile. The transfer is rectangular, spanning `dst.GetValidRow()` by `dst.GetValidCol()` elements.

## Mechanism

`pto.tload` initiates a DMA transfer from the source GlobalTensor to the destination tile buffer. The transfer reads a rectangular region from the GlobalTensor and writes it into the tile's on-chip storage.

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. The transfer size is `R × C` elements. The element mapping depends on the GlobalTensor layout:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{r_0 + i,\; c_0 + j} $$

where `(r_0, c_0)` is the base offset within the GlobalTensor. The exact address computation also depends on the GlobalTensor stride.

The operation is asynchronous. A `RecordEvent` token is returned; use `TSYNC` or `set_flag`/`wait_flag` before reading the tile data.

## Syntax

### PTO Assembly Form

```text
%t0 = tload %sv[%c0, %c0] : (!pto.memref<...>, index, index) -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
!pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

### AS Level 2 (DPS)

```text
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>)
          outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `dst` | Destination tile. The transfer shape is `dst.GetValidRow()` × `dst.GetValidCol()`. |
| `src` | Source GlobalTensor. Must be addressable from the local NPU. |
| `events...` | Optional `RecordEvent` tokens to wait on before issuing the operation |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | `RecordEvent` | Token signaling completion of the load. Must be waited on before the tile data is consumed. |

After the load completes, `dst` contains the loaded data with element layout determined by the tile layout and GlobalTensor stride.

## Side Effects

Reads from global memory and writes to the tile buffer. Does not implicitly fence unrelated tile traffic.

## Constraints

!!! warning "Constraints"
    - **Valid region**: The transfer size is `dst.GetValidRow()` × `dst.GetValidCol()`.
    - **Element size match**: `sizeof(tile.dtype) == sizeof(gtensor.dtype)`.
    - **Layout compatibility**: Source (GlobalTensor) layout and destination (tile) layout must be a supported combination. See the layout compatibility table in [Memory And Data Movement](../../memory-and-data-movement.md).
    - **Shape positivity**: `src.GetShape(dim) > 0` and `dst.GetValidRow() > 0` and `dst.GetValidCol() > 0` at runtime.

## Layout Compatibility

| TileType | ND→ND | DN→DN | NZ→NZ | ND→NZ | DN→ZN |
|----------|:-----:|:-----:|:-----:|:-----:|:-----:|
| `TileType::Vec` | Yes | Yes | Yes | No | No |
| `TileType::Mat` | Yes | Yes | Yes | Yes | Yes |
| `TileType::Acc` | Yes | No | Yes | No | No |

Additional constraints (A5):
- `Vec` with `ND→NZ` or `DN→ZN`: requires `GlobalData::staticShape[0..2] == 1` and `TileData::SFractalSize == 512`.
- `Vec` with `int64_t/uint64_t`: only `ND→ND` or `DN→DN` supported.

## Performance

### A2/A3 Cycle Count

`pto.tload` is a **MTE2** DMA from GM (or remote core memory) into the destination tile. Cost is dominated by DMA throughput and the chosen layout-conversion path; the vector and cube pipes are not on the critical path.

**Cycle model**:

```
total ≈ startup + R × C × bytes_per_elem / mte2_throughput + drain
```

For shapes that exceed a single MTE2 transaction, the transfer is decomposed into multiple sub-transactions that the hardware issues back-to-back. NPU benchmarks (Flash Attention) show MTE2 utilisation reaching ~55%, which is the dominant bottleneck on the AIC side.

### Layout-Conversion Path Impact

| Source → Dest | Path | Notes |
|---|---|---|
| ND → ND | Direct DMA | Fastest; no on-the-fly transpose |
| DN → DN | Direct DMA | Same as ND→ND |
| NZ → NZ | Direct DMA | Native fractal layout, no conversion |
| ND → NZ (`Mat`) | Layout-converting DMA | Hardware transposes into fractal layout en-route; latency-tolerant |
| DN → ZN (`Mat`) | Layout-converting DMA | Symmetric to ND→NZ |

### Bandwidth Asymmetry

A5 has asymmetric bandwidth across pipes; for `tload`, the **DDR → L1 → UB** path is typically the slowest segment for `Vec` and the **DDR → L1 → L0A/L0B** path bounds GEMM input feeders. Pair `tload` with [`tprefetch`](./tprefetch.md) and double-buffering to hide MTE2 latency behind compute.

> Note: cycle numbers are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    **A2/A3**:
    - `TileData::DType` must be one of: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
    - Destination tile location must be `TileType::Vec` or `TileType::Mat`.
    - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
    - `Vec` loads: layouts must match (ND→ND, DN→DN, NZ→NZ).
    - `Mat` loads: supports all combinations including ND→NZ and DN→ZN.
    - For ND→NZ or DN→ZN: `GlobalData::staticShape[0..2] == 1` and `TileData::SFractalSize == 512`.
    - `int64_t/uint64_t`: only ND→ND or DN→DN.

    **A5**:
    - `sizeof(TileData::DType)` must be 1, 2, 4, or 8 bytes, and must match `sizeof(GlobalData::DType)`.
    - `Vec` loads: row-major ND→ND, col-major DN→DN, or row-major NZ→NZ only.
    - `Mat` loads: constrained by `TLoadCubeCheck` (specific ND/DN/NZ conversions and L1-size limits).
    - `Mat` loads also handle `mx` format loads including `MX_A_ZZ/MX_A_ND/MX_A_DN` to ZZ for scalarA and `MX_B_NN/MX_B_ND/MX_B_DN` to NN for scalarB.
    - For `MX_A_ZZ/MX_B_NN`: `GlobalData::staticShape[3] == 16` and `GlobalData::staticShape[4] == 2`.
    - For `MX_A_ND/MX_ADN/MX_B_ND/MX_B_DN`: `GlobalData::staticShape[0] == 1` and `GlobalData::staticShape[1] == 1` and `GlobalData::staticShape[4] == 2`.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T>
void example(__gm__ T* in) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gin(in);
  TileT t;
  RecordEvent e = TLOAD(t, gin);
  TSYNC(e);
}
```

## See Also

- Instruction set overview: [Memory And Data Movement](../../memory-and-data-movement.md)
- Next op in instruction set: [pto.tprefetch](./tprefetch.md)
