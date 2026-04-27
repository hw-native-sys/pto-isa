# pto.tstore

`pto.tstore` is part of the [Memory And Data Movement](../../memory-and-data-movement.md) instruction set.

## Summary

`pto.tstore` initiates a DMA transfer from a source tile to global memory. It writes a rectangular region from the source tile into a GlobalTensor. Two storage-path variants are provided:

| Variant | Suffix | Description | Tile Types | Use Case |
|---------|---------|-------------|-----------|----------|
| Standard store | *(none)* | Direct tile-to-GM transfer | `Vec`, `Mat`, `Acc` | General tile output |
| Fix-pipe store | `_fp` | Store through fix-pipe quantization path | `Acc` | Quantized accumulation output |

## Mechanism

`TSTORE` initiates a DMA transfer from the source tile buffer to the destination GlobalTensor. The transfer reads a rectangular region from the source tile and writes it to global memory.

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. The transfer size is `R × C` elements. The element mapping depends on the GlobalTensor layout:

$$ \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{src}_{i,j} $$

### Fix-Pipe Variant (`TSTORE_FP`)

The `_fp` suffix means **fix pipe** — it routes the accumulator tile through the hardware fix-pipe quantization pipeline before writing to GM. This is the production path for quantized neural network inference where accumulation results must be converted (e.g., float32 → int8) before storage.

The auxiliary `fp` tile is the **sideband configuration tile** consumed by the backend `set_fpc(...)` path. It does not participate in the arithmetic — it programs the hardware quantization control registers.

$$ \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{Quantize}\!\left(\mathrm{src}_{i,j};\ \mathrm{fp}\right) $$

## Variants

### Variant 1: Standard Store

```cpp
// Basic store
template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &... events);

// Pre-quantization scalar (Acc tiles only)
template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, uint64_t preQuantScalar, WaitEvents &... events);
```

### Variant 2: Fix-Pipe Store (`TSTORE_FP`)

```cpp
// Fix-pipe quantized store — the _fp suffix means fix pipe, NOT floating point
template <typename TileData, typename GlobalData, typename FpTileData,
          AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData &dst, TileData &src, FpTileData &fp, WaitEvents &... events);
```

The `TSTORE_FP` overload is only legal for `TileType::Acc` tiles. It is the production path for quantized output — the `fp` tile carries quantization parameters (scale, zero-point) consumed by the fix-pipe.

## Syntax

### PTO Assembly Form

Standard store:

```text
tstore %t1, %sv_out[%c0, %c0]
```

Fix-pipe store:

```text
tstore.fp %t1, %fp, %sv_out[%c0, %c0]
```

### AS Level 1 (SSA)

```mlir
// Standard
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()

// Fix-pipe
pto.tstore.fp %src, %fp, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### AS Level 2 (DPS)

```mlir
// Standard
pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)

// Fix-pipe
pto.tstore.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `dst` | GlobalTensor | Destination in GM. Transfer shape is `src.GetValidRow()` × `src.GetValidCol()`. |
| `src` | Tile | Source tile. For standard: `Vec`, `Mat`, or `Acc`. For fix-pipe: `Acc` only. |
| `fp` | Tile (fix-pipe only) | Fix-pipe configuration tile. On A2/A3: `TileType::Scaling`. Programs quantization via `set_fpc(...)`. |
| `atomicType` | enum | Optional atomic mode. Default: `AtomicNone`. |
| `preQuantScalar` | uint64_t | Optional scalar for pre-quantization (Acc tiles only). |
| `reluPreMode` | enum | Optional ReLU pre-processing mode (fix-pipe variant only). |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | token | Signals completion of the DMA transfer. |

After the store completes, the data is written to `dst`. With atomic modes, values are accumulated. With `TSTORE_FP`, the transfer uses the fix-pipe sideband state programmed by the `fp` tile.

## Side Effects

- **Standard store**: Writes to global memory. With atomic modes, concurrent access may produce different accumulation ordering on different targets: on A2/A3, the DMA engine serializes concurrent atomic stores and guarantees all increments are applied, though the exact per-element interleaving is hardware-dependent; on A5, the atomic path also guarantees all increments are applied but may use different internal buffering; on the CPU simulator, atomic accumulation is emulated and the exact ordering of concurrent updates is not guaranteed to match hardware.
- **Fix-pipe store**: Programs fix-pipe sideband state (`set_fpc`) before the DMA transfer executes. Writes to global memory through the quantized path.

## Constraints

- **Valid region**: Transfer size is `src.GetValidRow()` × `src.GetValidCol()`.
- **Element size match**: `sizeof(tile.dtype) == sizeof(gtensor.dtype)`.
- **Layout compatibility**: Tile layout and GM layout must be a supported combination. See target-specific restrictions below.
- **Atomic modes**: Only supported on `TileType::Acc`. Supported modes: `AtomicNone`, `AtomicAdd`, `AtomicMax`, `AtomicMin` (A5 only).
- **Fix-pipe**: Only `TileType::Acc` is supported as the source. The `fp` tile must be `TileType::Scaling`. The fix-pipe path does not support arbitrary `ReluPreMode` on all backends — see target restrictions.

## Target-Profile Restrictions

### A2/A3

**Standard store:**

| Source Tile Type | Requirements |
|-----------------|-------------|
| `Vec` / `Mat` | `sizeof(TileData::DType)` must match `sizeof(GlobalData::DType)`. Supported dtypes: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`. |
| `Acc` (non-quantized) | Destination dtype must be `__gm__ int32_t / float / half / bfloat16_t`. |
| `Acc` (atomic) | AtomicAdd on `int32_t` or `float`. |
| `int64_t/uint64_t` | Only ND→ND or DN→DN layout. |

**Accumulator shape constraints (A2/A3):**
- `1 <= TileData::Cols <= 4095`
- If ND layout: `1 <= TileData::Rows <= 8192`
- If NZ layout: `1 <= TileData::Rows <= 65535` and `TileData::Cols % 16 == 0`

**Fix-pipe store (TSTORE_FP on A2/A3):**

| Requirement | Value |
|------------|-------|
| Destination layout | ND or NZ only |
| Source dtype | `int32_t` or `float` |
| Static row constraint | `1 <= TileData::Cols <= 4095`; ND: `Rows <= 8192`; NZ: `Rows <= 65535`, `Cols % 16 == 0` |
| Runtime col constraint | `1 <= src.GetValidCol() <= 4095` |
| FpTileData | No explicit `static_assert`; used via `set_fpc(...)` internally |

### A5

**Standard store:**

| Source Tile Type | Notes |
|-----------------|-------|
| `Vec` | `sizeof(TileData::DType)` must match `sizeof(GlobalData::DType)`. Additional dtypes on A5: `float8_e4m3_t`, `float8_e5m2_t`, `hifloat8_t`, `float4_e1m2x2_t`, `float4_e2m1x2_t`. |
| `Acc` | Destination layout must be ND or NZ. Source dtype must be `int32_t` or `float`. Additional alignment: ND row-major width in bytes must be a multiple of 32. |
| `Acc` (atomic) | `AtomicAdd`, `AtomicMax`, `AtomicMin` on `int32_t`. |

**Fix-pipe store (TSTORE_FP on A5):**

| Requirement | Value |
|------------|-------|
| Destination layout | ND or NZ |
| Source dtype | `int32_t` or `float` |
| FpTileData | Used via `CheckStaticAcc<..., true>()` validation |

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier.
- Programs must not rely on behavior outside the documented legal domain.
- Calling `TSTORE_FP` on a non-accumulator tile is rejected by the backend.

## Common Patterns

### Pattern 1: Basic Vector Tile Store

```cpp
template <typename T>
void storeResult(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  // ... compute into t ...
  TSTORE(gout, t);
}
```

### Pattern 2: Atomic Accumulation

```cpp
void atomicStore(GlobalTensor<int32_t>& gout, TileAcc<int32_t, 64, 64>& acc) {
  // Atomically add accumulator to GM location
  TSTORE(gout, acc, AtomicType::AtomicAdd);
}
```

### Pattern 3: Fix-Pipe Quantized Store (Production Inference)

```cpp
void quantizedStore(__gm__ int8_t* out) {
  using AccT = TileAcc<float, 16, 16>;
  using FpT = Tile<TileType::Scaling, uint64_t, 1, 16,
                   BLayout::RowMajor, 1, DYNAMIC, SLayout::NoneBox>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<int8_t, 16, 16, Layout::ND>;
  using GT = GlobalTensor<int8_t, GShape, GStride, Layout::ND>;

  GT gout(out);
  AccT acc;
  FpT fp(16);  // 16 scale factors (one per output channel)

  // ... compute into acc ...
  // Apply fix-pipe quantization: float32 acc → int8 output via fp scales
  TSTORE_FP(gout, acc, fp);
}
```

### Pattern 4: Manual Mode with TASSIGN

```cpp
void manualStore(__gm__ float* out) {
  using TileT = Tile<TileType::Vec, float, 32, 32>;
  using GShape = Shape<1, 1, 1, 32, 32>;
  using GStride = BaseShape2D<float, 32, 32, Layout::ND>;
  using GTensor = GlobalTensor<float, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TASSIGN(t, 0x1000);
  // ... compute into t ...
  TSTORE(gout, t);
}
```

## See Also

- Instruction set overview: [Memory And Data Movement](../../memory-and-data-movement.md)
- Previous op in instruction set: [pto.tprefetch](./tprefetch.md)
- [pto.tload](./tload.md) — The inverse operation (GM → tile)
- [Assembly Spelling And Operands](../../../syntax-and-operands/assembly-model.md)
