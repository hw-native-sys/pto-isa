# THISTOGRAM

> **Implementation status**: THISTOGRAM is provided as a C++ intrinsic on the A5 and Kirin9030 back ends and in CPU simulation (`__CPU_SIM`), and is registered in the virtual-ISA indexes (`PTOISA`, `isa/README`, `manifest.yaml`, the `appendix-d` family matrix, and the mkdocs nav). It is available on A5 / Kirin9030 / CPU-sim only (not on A2/A3); it has no public bytecode encoding yet.

## Introduction

Computes a histogram (per byte-value 0–255 occurrence count) over a selected **byte** of each source-tile element, with an optional cascaded filter that restricts the count to elements whose already-processed upper bytes equal supplied index values. It is the **byte-bucket counting primitive of radix sort**: the first pass histograms the most-significant byte (MSB); each subsequent pass histograms the next lower byte but counts only elements whose upper bytes match the previous pass's bucket index, yielding the current radix digit's distribution within a given prefix bucket.

A single call produces, for each valid source row, a set of 256 `uint32` bin counts.

## Math Interpretation

Let the source `src` have valid shape $R \times C$ with `uint16_t` or `uint32_t` elements. Let $B_k(x)$ denote byte $k$ of element $x$:

$$
B_0 = \text{bits } 7\text{–}0\ (\text{LSB}),\quad B_1 = \text{bits } 15\text{–}8,\quad B_2 = \text{bits } 23\text{–}16,\quad B_3 = \text{bits } 31\text{–}24\ (\text{MSB})
$$

The template parameter `byte` selects the byte $k\in\{0,1,2,3\}$ being histogrammed. For each source row $r\in[0,R)$ and each bin value $b\in[0,256)$:

$$
\mathrm{dst}_{r,b} = \bigl|\{\,j\in[0,C)\ \big|\ B_k(\mathrm{src}_{r,j})=b\ \wedge\ F_{k}(r,j)\ \}\bigr|
$$

where the cascaded filter $F_k$ is defined MSB-first (process $k=3$, then $k=2,1,0$):

| Source dtype | `byte` $k$ | Filter $F_k(r,j)$ | `idx` meaning |
|--------------|-----------|-------------------|---------------|
| `uint16` | `BYTE_1` (MSB) | always true (first pass, no filter) | unused |
| `uint16` | `BYTE_0` (LSB) | $B_1(\mathrm{src}_{r,j})=\mathrm{idx}_{r}$ | 1 match byte per row (upper byte) |
| `uint32` | `BYTE_3` (MSB) | always true (first pass, no filter) | unused |
| `uint32` | `BYTE_2` | $B_3=\mathrm{idx}_{r,0}$ | 1 filter-byte row |
| `uint32` | `BYTE_1` | $B_3=\mathrm{idx}_{r,0}\ \wedge\ B_2=\mathrm{idx}_{r,1}$ | 2 filter-byte rows |
| `uint32` | `BYTE_0` (LSB) | $B_3=\mathrm{idx}_{r,0}\wedge B_2=\mathrm{idx}_{r,1}\wedge B_1=\mathrm{idx}_{r,2}$ | 3 filter-byte rows |

- `dst` always has 256 `uint32` bins per row (one per byte value 0–255).
- A `uint16` source supports only `BYTE_0` / `BYTE_1` (only two bytes are extractable).
- Unless otherwise specified, semantics are defined over the valid region; the exact in-memory interleaved layout of the bin counts (N0/N1 dual banks, even/odd split) is implementation-defined — the logical result is 256 counts per row.

> `src` and `idx` are ISA-visible tile operands (not compiler scratch).

## C++ Intrinsics

Declared in `include/pto/common/pto_instr.hpp`, available under A5 / Kirin9030 / CPU simulation (`PTO_NPU_ARCH_A5 || PTO_NPU_ARCH_KIRIN9030 || __CPU_SIM`). The `HistByte` enum is defined in `include/pto/common/type.hpp`.

```cpp
enum class HistByte : uint8_t {
    BYTE_0 = 0, // LSB (bits 7-0)
    BYTE_1 = 1, // bits 15-8
    BYTE_2 = 2, // bits 23-16
    BYTE_3 = 3  // MSB (bits 31-24)
};

template <HistByte byte, typename TileDataDst, typename TileDataSrc, typename TileDataIdx, typename... WaitEvents>
PTO_INST RecordEvent THISTOGRAM(TileDataDst &dst, TileDataSrc &src, TileDataIdx &idx, WaitEvents &...events);
```

| Parameter | Direction | Meaning |
|-----------|-----------|---------|
| `byte` | template | The byte to histogram (`HistByte::BYTE_0`…`BYTE_3`) |
| `dst` | output | Histogram result tile, `uint32_t`, row-major, 256 bins per row |
| `src` | input | Source data tile, `uint16_t` or `uint32_t`, row-major |
| `idx` | input | Cascaded filter-index tile, `uint8_t`, shape varies with `byte` and source dtype (see below) |
| `events...` | input | Wait events (`WaitEvents`); an implicit `TSYNC` precedes the op |

## Tile Sizes & Data Types

For a source valid shape $R \times C$:

| Tile | dtype | Valid shape | Layout | Notes |
|------|-------|-------------|--------|-------|
| `dst` | `uint32_t` | $R \times 256$ | RowMajor | 256 bin counts per row |
| `src` | `uint16_t` or `uint32_t` | $R \times C$ | RowMajor | Data being histogrammed |
| `idx` (`uint16` src) | `uint8_t` | $R \times 1$ | ColMajor (DN) | 1 match byte per row (upper byte) |
| `idx` (`uint32` src) | `uint8_t` | $(3-k) \times C$ | RowMajor | 1 filter byte broadcast per row; 0 rows when $k=3$ (unused) |

> `idx` physical rows must be aligned to 32-byte blocks (`PTO_CEIL(rows · sizeof(uint8_t), 32)`); in `uint16` mode `idx` must use the DN layout (`BLayout::ColMajor` + `SLayout::NoneBox`) with exactly one column.

## Supported Input Dtypes

| Source dtype | Destination dtype | idx dtype | Allowed `byte` | Notes |
|--------------|-------------------|-----------|----------------|-------|
| `U16` (`uint16_t`) | `U32` | `U8` | `BYTE_0`, `BYTE_1` | Only low/high byte extractable |
| `U32` (`uint32_t`) | `U32` | `U8` | `BYTE_0`…`BYTE_3` | All four bytes, paired with 0–3 idx rows |

> `dst` must be `uint32_t`, `idx` must be `uint8_t`, and `src` is limited to `uint16_t` / `uint32_t`; other combinations are rejected by an in-implementation `static_assert`.

## Implementation Notes

THISTOGRAM runs on the vector pipeline (`PIPE_V`):

1. **Byte extraction**: source elements are deinterleaved into per-byte vectors — `uint16` via `DINTLV_B8` (splits MSB/LSB), `uint32` via `DINTLV_B16` + `vdintlv` (splits into 4 bytes).
2. **Cascaded filter**: `vcmp_eq` builds a predicate of "already-processed upper byte == idx", AND-ed across bytes (the MSB first pass has no filter).
3. **Byte histogram**: the hardware `chistv2` op counts the selected byte under the filter predicate, accumulating internally across the N0/N1 dual banks with an even/odd split; each row finally stores 256 `uint32` bins (`INTLV_B32` interleaved store). The dual-bank / even-odd split is an implementation detail — the logical result is 256 counts per row.

## Constraints

| Constraint | Applies to | Reason |
|------------|------------|--------|
| `dst` is `uint32_t` and row-major | all targets | 256-bin count width and store layout |
| `src` ∈ {`uint16_t`, `uint32_t`} and row-major | all targets | byte-extraction path |
| `idx` is `uint8_t` | all targets | filter-byte width |
| `uint16` src: `idx` is DN (ColMajor + NoneBox) with 1 column | A5 / Kirin9030 / CPU | one match byte broadcast per row |
| `uint32` src: `idx` row-major, rows $=3-k$, cols $=$ source cols | A5 / Kirin9030 / CPU | index rows needed for cascaded filtering |
| `uint16` src allows only `BYTE_0` / `BYTE_1` | all targets | `uint16` has only 2 bytes |
| `dst` has 256 bins per row | all targets | byte value space 0–255 |

## Examples

```cpp
// uint16 source: histogram the high byte (MSB) of each element (radix-sort pass 1).
THISTOGRAM<HistByte::BYTE_1>(dstTile, srcTile, idxTile);

// uint16 source: histogram the low byte (LSB), counting only elements whose high byte == idx (pass 2).
THISTOGRAM<HistByte::BYTE_0>(dstTile, srcTile, idxTile);
```

Typical tile declarations (`uint16` mode, source valid shape $R\times C$):

```cpp
using TileDataSrc = Tile<TileType::Vec, uint16_t, R, alignedC,        BLayout::RowMajor>;
using TileDataDst = Tile<TileType::Vec, uint32_t, R, 256,             BLayout::RowMajor>;
using TileDataIdx = Tile<TileType::Vec, uint8_t,  alignedIdxBytes, 1, BLayout::ColMajor>;
```

See `tests/npu/a5/src/st/testcase/thistogram/` (A5), `tests/npu/kirin9030/src/st/testcase/thistogram/` (Kirin9030), `tests/npu/kirinX90/src/st/testcase/thistogram/` (KirinX90), and `tests/cpu/st/testcase/thistogram/` (CPU reference) for complete ST examples.
