# Memory And Data Movement Instruction Set

Memory operations transfer data between global memory (GM) and tile buffers. These are the **only** tile operations that cross between tile-visible state and GM-visible state.

## Operations

| Operation | Description | Direction | C++ Intrinsic |
|-----------|-------------|-----------|----------------|
| [pto.tload](./ops/memory-and-data-movement/tload.md) | Load from GM into tile | GM → local tile buffer | `TLOAD(dst, gtensor)` |
| [pto.tprefetch](./ops/memory-and-data-movement/tprefetch.md) | Prefetch from GM into tile (non-blocking) | GM → local tile buffer | `TPREFETCH(dst, gtensor)` |
| [pto.tstore](./ops/memory-and-data-movement/tstore.md) | Store from tile to GM | local tile buffer → GM | `TSTORE(gtensor, src)` |
| [pto.tstore_fp](./ops/memory-and-data-movement/tstore.md) | Store through the fix-pipe path | Tile → local tile buffer → GM | `TSTORE_FP(gtensor, src, fp)` |
| [pto.mgather](./ops/memory-and-data-movement/mgather.md) | Gather scattered elements from GM | GM → local tile buffer | `MGATHER(dst, gtensor, indices)` |
| [pto.mscatter](./ops/memory-and-data-movement/mscatter.md) | Scatter tile elements to GM | local tile buffer → GM | `MSCATTER(gtensor, indices, src)` |

## Mechanism

### Contiguous Transfer (TLOAD, TSTORE)

Data is transferred in a rectangular region determined by the tile's valid region:

```
TLOAD:  dst[i,j] = src[ r0 + i, c0 + j ]   (i ∈ [0, dst.Rv), j ∈ [0, dst.Cv))
TSTORE: dst[ r0 + i, c0 + j ] = src[i,j]
```

Transfer size: `dst.GetValidRow() × dst.GetValidCol()` elements.

### Prefetch (TPREFETCH)

`TPREFETCH` initiates a non-blocking DMA transfer from GM to the tile buffer. It does not stall the pipeline. A subsequent operation that reads the tile buffer must wait for the transfer to complete via `TSYNC` or `set_flag`/`wait_flag`.

### Gather/Scatter (MGATHER, MSCATTER)

An index tile specifies which GM elements to transfer:

$$ \mathrm{dst}_i = \mathrm{src}_{\mathrm{index}_i} $$

### Fix-Pipe Variants (TSTORE_FP)

`TSTORE_FP` is a **fix-pipe** variant, not a “floating-point” variant. The `_fp` suffix names the backend path that programs fix-pipe state before storing.

## Layout Compatibility

| TileType | ND→ND | DN→DN | NZ→NZ | ND→NZ | DN→ZN | Notes |
|----------|:-----:|:-----:|:-----:|:-----:|:-----:|-------|
| `TileType::Vec` | Yes | Yes | Yes | No | No | |
| `TileType::Mat` | Yes | Yes | Yes | Yes | Yes | |
| `TileType::Acc` | Yes | No | Yes | No | No | Atomic store only |

Additional constraints on A5:
- `TileType::Vec` with `ND→NZ` or `DN→ZN`: requires `GlobalData::staticShape[0..2] == 1` and `TileData::SFractalSize == 512`.
- `TileType::Vec` with `int64_t/uint64_t`: only `ND→ND` or `DN→DN` supported.

## Type Support by Target Profile

| Element Type | CPU Simulator | A2/A3 | A5 |
|------------|:-------------:|:------:|:--:|
| f32 (float) | Yes | Yes | Yes |
| f16 (half) | Yes | Yes | Yes |
| bf16 (bfloat16_t) | Yes | Yes | Yes |
| i8 / u8 | Yes | Yes | Yes |
| i16 / u16 | Yes | Yes | Yes |
| i32 / u32 | Yes | Yes | Yes |
| i64 / u64 | Yes | Yes | Yes |
| f8e4m3 / f8e5m2 | No | No | Yes |
| hifloat8_t / float4_e* | No | No | Yes |

## Ordering

Memory operations are subject to PTO's producer-consumer ordering rules. Programs MUST use explicit synchronization (`TSYNC`, `set_flag`/`wait_flag`) to ensure data is ready before use.

See [Producer Consumer Ordering](../memory-model/producer-consumer-ordering.md) for the full ordering model.

## Constraints

- Source and destination element types MUST have the same size: `sizeof(tile.dtype) == sizeof(gtensor.dtype)`.
- Transfer size is determined by the destination tile's valid region for `TLOAD`, or source tile's valid region for `TSTORE`.
- Layout compatibility between GM layout and tile layout is profile-dependent (see layout compatibility table above).
- Gather/scatter index tiles must have compatible shapes.
- `TSTORE` with `TileType::Acc` supports `AtomicType`: `AtomicNone`, `AtomicAdd`, `AtomicMax`, `AtomicMin` (A5 only).
- `TSTORE_FP` is only legal for `TileType::Acc` on A2A3 and A5 and uses the fix-pipe sideband state carried by the auxiliary `fp` tile argument.

## Cases That Are Not Allowed

- Transferring to or from an uninitialized tile register.
- Using a GlobalTensor with strides incompatible with the transfer pattern.
- Accessing GM addresses outside the tensor's declared shape.
- Using `TSTORE_FP` with a non-Acc tile type.
- Using atomic store variants on CPU simulator.

## C++ Intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Basic load
template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData& dst, GlobalData& src, WaitEvents&... events);

// Atomic store
template <typename TileData, typename GlobalData,
          AtomicType atomicType = AtomicType::AtomicNone, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events);

// FP store (quantized, A2/A3+)
template <typename TileData, typename GlobalData, typename FpTileData,
          AtomicType atomicType = AtomicType::AtomicNone, typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData& dst, TileData& src, FpTileData& fp,
                               WaitEvents&... events);

// Prefetch
template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData& dst, GlobalData& src);

// Gather/Scatter
template <typename TileData, typename GlobalData, typename IndexData>
PTO_INST RecordEvent MGATHER(TileData& dst, GlobalData& src, IndexData& indices);

template <typename TileData, typename GlobalData, typename IndexData>
PTO_INST RecordEvent MSCATTER(GlobalData& dst, IndexData& indices, TileData& src);
```

## See Also

- [Memory model](../memory-model/consistency-baseline.md) — GM ordering and consistency
- [Producer consumer ordering](../memory-model/producer-consumer-ordering.md) — Sync rules
- [Tile instruction set](../instruction-families/tile-families.md) — Instruction set overview
- [Tile instruction set](../instruction-families/tile-families.md) — Instruction Set description
