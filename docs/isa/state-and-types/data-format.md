# Data Format Reference

The **physical data format** defines how tiles, vectors, and scalars are represented in memory and in hardware registers. It covers memory spaces, element packing, address alignment, VLane architecture, and the relationship between the PTO logical view and the underlying storage.

## Memory Spaces

PTO distinguishes global memory from local tile-buffer storage. The ISA-level concept is the **tile buffer**; the hardware names behind that concept depend on `TileType`.

| Memory Space | Location | Access Unit | Bandwidth | Access Pattern |
|-------------|----------|-------------|-----------|----------------|
| **GM** (Global Memory) | Off-chip device DRAM | Byte-granular | Low | Global backing store |
| **Local tile buffers** | On-chip local storage | Role-specific | High | Direct tile/vector access through the selected tile role |

The hardware mapping of local tile buffers is:

| PTO tile-buffer role | Hardware-local buffer |
|----------------------|----------------------|
| `TileType::Vec` | Unified Buffer (UB) |
| `TileType::Left` | L0A |
| `TileType::Right` | L0B |
| `TileType::Acc` | L0C |
| `TileType::ScaleLeft` | L0A scale buffer |
| `TileType::ScaleRight` | L0B scale buffer |

`Tile Register File` terminology in the current manual should be read as the ISA abstraction over these local tile buffers, not as a second user-visible storage class separate from the buffers themselves.

## Tile Buffer Format

A tile occupies a contiguous region in one local tile buffer. Its logical shape `(Rows, Cols)` is independent of its physical storage format.

### In-Memory Format

In a local tile buffer, elements are stored in their `BLayout` order — either `RowMajor` or `ColMajor`. Each element occupies `sizeof(DType)` bytes. For `TileType::Vec`, that local tile buffer is the hardware Unified Buffer.

For `BLayout = RowMajor`, shape `(R, C)`:

$$ \text{addr}(r, c) = (r \times C + c) \times \mathrm{sizeof(DType)} $$

For `BLayout = ColMajor`, shape `(R, C)`:

$$ \text{addr}(r, c) = (c \times R + r) \times \mathrm{sizeof(DType)} $$

### Tile-Register View

The tile-register view is the ISA abstraction presented to authors. It names typed local tile buffers and hides the fact that different `TileType` values are backed by different hardware-local buffers. Tile data is moved in and out via explicit `TLOAD`/`TSTORE`/`TMOV*`-family operations rather than by scalar byte addressing.

### Address Alignment

| Access Type | Required Alignment |
|-------------|-------------------|
| GM read/write | Element-size aligned (2 bytes for f16/i16, 4 bytes for f32) |
| Vector tile buffer DMA transfer | 32-byte block aligned (DMA engine unit) |
| Local tile-buffer access | Element-size aligned, plus any role-specific backend constraints |

The DMA engine operates on 32-byte blocks (`BLOCK_BYTE_SIZE = 32`). Misaligned GM addresses produce target-specific behavior:
- **A2/A3 and A5**: The DMA engine requires natural alignment for best performance; unaligned addresses may cause DMA errors or performance degradation. Software should align GM addresses to at least 32-byte boundaries.
- **CPU simulator**: Unaligned addresses are accepted and handled by the host CPU's memory access instructions.

## Element Type Encoding

### Standard Types

| Type | C++ Type | SSA Name | Size (bytes) | Register Width |
|------|----------|----------|:------------:|:-------------:|
| IEEE FP16 | `half` | `f16` | 2 | 128 lanes |
| Brain FP16 | `bfloat16_t` | `bf16` | 2 | 128 lanes |
| IEEE FP32 | `float` | `f32` | 4 | 64 lanes |
| Signed int8 | `int8_t` | `i8` | 1 | 256 lanes |
| Unsigned int8 | `uint8_t` | `u8` | 1 | 256 lanes |
| Signed int16 | `int16_t` | `i16` | 2 | 128 lanes |
| Unsigned int16 | `uint16_t` | `u16` | 2 | 128 lanes |
| Signed int32 | `int32_t` | `i32` | 4 | 64 lanes |
| Unsigned int32 | `uint32_t` | `u32` | 4 | 64 lanes |

### A5-Only Types

| Type | C++ Type | SSA Name | Size (bytes) | Notes |
|------|----------|----------|:------------:|-------|
| FP8 E4M3 | `float8_e4m3_t` | `f8e4m3` | 1 | 256 lanes |
| FP8 E5M2 | `float8_e5m2_t` | `f8e5m2` | 1 | 256 lanes |
| HI Float8 | `hifloat8_t` | `hifloat8` | 1 | 256 lanes |
| Float4 E1M2x2 | `float4_e1m2x2_t` | `float4_e1m2x2` | 1 | 256 lanes (packed 2×2) |
| Float4 E2M1x2 | `float4_e2m1x2_t` | `float4_e2m1x2` | 1 | 256 lanes (packed 2×2) |

## Vector Register Format (VLane Architecture)

On A5 (Ascend 950 PR / DT), the vector register is organized as **8 VLanes** of 32 bytes each. A VLane is the atomic unit for group reduction operations. This architecture is architecturally visible in PTO.

```
vreg (256 bytes total):
┌─────────┬─────────┬─────────┬─────┬─────────┬─────────┐
│ VLane 0 │ VLane 1 │ VLane 2 │ ... │ VLane 6 │ VLane 7 │
│   32B   │   32B   │   32B   │     │   32B   │   32B   │
└─────────┴─────────┴─────────┴─────┴─────────┴─────────┘
```

Vector registers hold `N` elements of type `DType` packed contiguously with no padding. The register width is always 256 bytes (2048 bits):

| Element Type | Lane Count N | Bytes/Lane | Total |
|-------------|:-----------:|:----------:|:-----:|
| `f32` | 64 | 4 | 256 B |
| `f16` / `bf16` / `i16` / `u16` | 128 | 2 | 256 B |
| `i8` / `u8` / FP8 / HI-FP8 | 256 | 1 | 256 B |
| `float4_*` (packed) | 256 (effective) | 1 | 256 B |

### Group Reduction and VLanes

Group reduction operations (`vcgadd`, `vcgmax`, `vcgmin`) reduce within each VLane independently. The reduction produces one result per VLane (one value per 32-byte lane), which is then broadcast or stored:

```c
// Per-VLane group reduction: each VLane independently reduces its K elements
int K = N / 8;  // elements per VLane (e.g., 8 for f32, 16 for f16)
for (int g = 0; g < 8; g++) {
    T sum = 0;
    for (int i = 0; i < K; i++)
        sum += src[g*K + i];
    dst[g*K] = sum;           // write result to first position of each VLane
    for (int i = 1; i < K; i++)
        dst[g*K + i] = 0;    // zero-fill remaining positions
}
```

This is architecturally visible: the result is not a single scalar but one value per VLane.

## Pad Value Encoding

The `Pad` parameter in `Tile<DType, ..., Pad>` specifies the value of out-of-valid-region elements. Declared in `include/pto/common/constants.hpp`.

### Standard Pad Values

| Pad Value | Meaning | `float` Encoding | `half`/`bf16` Encoding | `i8`/`u8` Encoding |
|-----------|---------|-------------------|-------------------------|---------------------|
| `Zero` | Initialize to zero | `0x00000000` | `0x0000` | `0x00` |
| `Null` | Undefined; must not be read | `0x00000000` | `0x0000` | `0x00` |
| `Min` | Fill with type minimum | `0xff800000` (≈ −0) | `0xfc00` | `0xff` |
| `Max` | Fill with type maximum | `0x7f800000` (+Inf) | `0x7c00` | `0x7f` |

### Custom Pad Values (A5)

The `PadValueCustom(value)` helper allows compile-time-specified float patterns as pad values. This is useful for operations that need a specific fill value (e.g., `-1.0f` for softmax):

```cpp
// Custom pad value: all out-of-valid-region elements become -1.0f
using TilePadNeg1 = Tile<TileType::Vec, float, 16, 16, RowMajor, NoneBox, None, PadValueCustom(-1.0f)>;
```

Custom pad values encode the float bit pattern in the upper bits of the 64-bit `PadValue` enum. They are processed by `PadValueMap` and applied via `GetPadValue()` at load time.

## MX Block-Scale Formats

MX block-scale matmul forms use extra scale tiles in addition to the left and right payload tiles. In the current codebase:

- `TileLeft` corresponds to L0A
- `TileRight` corresponds to L0B
- `TileLeftScale` corresponds to the L0A-side scale buffer
- `TileRightScale` corresponds to the L0B-side scale buffer

The A5 `TMATMUL_MX` / `TGEMV_MX` code paths explicitly require both scale tiles, and the supported combinations include MX FP4 and MX FP8 families. These are block-scale formats, not plain elementwise FP formats.

## Fractal Layout Encoding

The `TileLayoutCustom` enum in `include/pto/common/constants.hpp` encodes the concrete layout used at runtime:

| `TileLayoutCustom` | BLayout | SLayout | Fractal | Block Size | Typical Use |
|--------------------|---------|---------|---------|:---------:|-------------|
| `ND` | RowMajor | NoneBox | — | — | Standard tile; most ops |
| `DN` | ColMajor | NoneBox | — | — | Fortran-order tile |
| `NZ` | ColMajor | RowMajor | NZ | 512 B | Left/L0A-side matmul operand on A5 |
| `ZN` | RowMajor | ColMajor | ZN | 512 B | Symmetric NZ variant |
| `ZZ` | RowMajor | RowMajor | ZZ | 512 B | CUBE-specific pattern |

The `BLOCK_BYTE_SIZE = 32` constant and `FRACTAL_NZ_ROW = 16` and `CUBE_BLOCK_SIZE = 512` give the fractal block dimensions used in address generation.

## Constants Reference

| Constant | Value | Units | Use |
|----------|-------|-------|-----|
| `BLOCK_BYTE_SIZE` | 32 | bytes | DMA block transfer unit |
| `FIXP_BURST_UNIT_LEN` | 64 | half-words | DMA burst length |
| `FRACTAL_NZ_ROW` | 16 | elements | Fractal row dimension for NZ/ZN |
| `CUBE_BLOCK_SIZE` | 512 | bytes | CUBE fractal block |
| `C0_SIZE_BYTE` | 32 | bytes | Cube C0 dimension (in bytes) |
| `MX_COL_LEN` | 2 | elements | MX block-scale column block |
| `MX_ROW_LEN` | 16 | elements | MX block-scale row block |
| `MX_BLOCK_SIZE` | 32 | elements | MX block-scale block |
| `TMP_UB_SIZE` | 8 × 1024 | bytes | Temporary UB buffer size |
| `TMP_UB_OFFSET` | 184 × 1024 | bytes | Temporary UB offset |
| `MASK_LEN` | 64 | bits | Predicate mask width |
| `BLOCK_LEN` | 16 | elements | Standard block length |
| `VLane_COUNT` | 8 | lanes | VLanes per vector register (A5) |

## See Also

- [Type System](./type-system.md) — Element type inventory, NaN/Inf rules, conversion rules
- [Layout Reference](./layout.md) — BLayout, SLayout, Fractal, TileType–Layout compatibility
- [Tiles and Valid Regions](../programming-model/tiles-and-valid-regions.md) — Valid-region semantics and programming model
- [Memory Model](../memory-model/consistency-baseline.md) — GM, UB, TRF hierarchy and ordering guarantees
