# TROWSUM


## Tile Operation Diagram

![TROWSUM tile operation](../figures/isa/TROWSUM.svg)

## Introduction

Reduce each row by summing across columns.

## Math Interpretation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R`:

$$ \mathrm{dst}_{i,0} = \sum_{j=0}^{C-1} \mathrm{src}_{i,j} $$

## Assembly Syntax

Synchronous form:

```text
%dst = trowsum %src : !pto.tile<...> -> !pto.tile<...>
```
Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### AS Level 1 (SSA)

```text
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## Constraints

### General constraints / checks

The following constraints describe the NPU backends. CPU_SIM checks and compatibility exceptions are listed below.

- `dst` and `src` must both be `TileType::Vec`.
- `src` must use standard ND layout: row-major and non-fractal (`BLayout::RowMajor`, `SLayout::NoneBox`).
- `dst` must use one of the following non-fractal layouts:
    - ND layout (`BLayout::RowMajor`, `SLayout::NoneBox`), or
    - DN layout with exactly one column (`BLayout::ColMajor`, `SLayout::NoneBox`, `Cols == 1`).
- `dst` and `src` must use the same element type.
- Runtime valid-region checks:
    - `src.GetValidRow() != 0`
    - `src.GetValidCol() != 0`
    - `src.GetValidRow() == dst.GetValidRow()`
- The intrinsic signature requires an explicit `tmp` operand.

### NPU implementation checks

- Supported element types (A2A3): `half`, `float`, `int32_t`, `int16_t`.
- Supported element types (A5): `half`, `float`, `int32_t`, `int64_t`, `uint64_t`, `int16_t`.
- The implementation accepts both ND output and DN output with `Cols == 1`; it is not limited to DN output.
- Runtime checks follow the shared row-reduce check path:
    - `src.GetValidRow() != 0`
    - `src.GetValidCol() != 0`
    - `src.GetValidRow() == dst.GetValidRow()`

### CPU_SIM implementation checks

The target comes from the calling thread's memory model, not from host hardware. See
[Selecting the simulated architecture](../coding/cpu_sim.md#selecting-the-simulated-architecture).

- **A5**: Input and output types must match and be one of `half`, `float`, `int16_t`, `int32_t`,
  `int64_t`, or `uint64_t`. Native BF16 and mixed input/output types are not supported.
  The Vec, ND/DN layout, non-empty input, and equal valid-row constraints above are checked at runtime.
  Source and destination physical row counts may differ.
- **A2A3 compatibility path**: Supported source/output pairs are `half`/`half`, `half`/`float`,
  `bfloat16_t`/`bfloat16_t`, `bfloat16_t`/`float`, `float`/`float`, `int16_t`/`int16_t`,
  `int16_t`/`int32_t`, and `int32_t`/`int32_t`. Physical row counts must match, but the
  A5-specific layout and valid-region assertions are not applied. These compatibility allowances
  do not extend the A2A3 NPU contract.
- Type pairs supported by neither CPU path are rejected at compile time; pairs unsupported by the selected
  architecture are rejected at runtime. Architecture-specific assertions abort on failure even with `NDEBUG`.
- Callers must keep dynamic valid shapes within the physical Tiles and provide destination storage and a valid
  region covering all output rows and column zero. The implementation does not check `dst.GetValidCol()`
  or provide general dynamic-shape bounds validation. Both paths write only column zero of each processed row,
  leaving other destination elements and valid-shape metadata unchanged.
- **Numerical differences**: A5 floating-point reduction uses grouped binary trees, converting back to the
  element type after every addition; integers wrap at the output width without saturation. A2A3 retains
  the legacy accumulation loop: `half`/`bfloat16_t` outputs accumulate in `float` and then convert;
  other outputs accumulate in their own type, without A5's unsigned modular-overflow handling.
  A2A3 row-major vectorization hints may reorder floating-point additions even without `-ffast-math`;
  strict left-to-right order and bit-identical results across compilers are not guaranteed.
- Full bit-exact hardware equivalence is not guaranteed: A5 NaN payloads, subnormal/flush-to-zero behavior,
  and other floating-point details are not established as equivalent; the A2A3 path is for compatibility.
  Do not enable reassociation options such as `-ffast-math` when relying on the A5 reduction order.
  See [TROWSUM implementation notes](../coding/cpu_sim.md#trowsum-implementation-notes) for group order,
  rounding, and the BF16 placeholder caveat.

## Temporary Space

### A2A3

`tmp` **is used** as scratch storage for row-wise reduction.

- For **integer** types (`int32_t`, `int16_t`): `tmp` is used as a per-row accumulator buffer (1 block). For each row, `tmp` is initialized to 0, then blocks of `src` are accumulated via `vadd`. The final sum is read from `tmp` in scalar mode.
  - `tmp` size: at least 1 row and `BLOCK_BYTE_SIZE / sizeof(T)` columns (8 for `int32_t`, 16 for `int16_t`).
- For **floating-point** types (`float`, `half`): `tmp` is used for binary-tree reduction via `vcadd`/`vcgadd`. The required size depends on the number of repeat blocks per row.
  - A safe default: set `tmp` to the same shape as `src`.

### A5

`tmp` is accepted by the interface but **not used** by the A5 implementation. The A5 backend uses vector register-based reduction (`vcadd` instruction) and does not require scratch tile storage. `tmp` is retained in the C++ intrinsic signature solely for API compatibility with A2A3.


### CPU_SIM

Both architecture paths accept `tmp` for API compatibility but do not access its storage. Portable kernels must
still provide any scratch storage required by their NPU backend.

## Examples

These snippets illustrate invocation and storage binding, not complete runnable programs. Initialize source
values before the reduction. For CPU_SIM, compile with `__CPU_SIM`; the Auto example also needs `__PTO_AUTO__`.
To exercise the A5 path, select A5 at application startup as described in the CPU_SIM guide; these functions do
not choose the architecture themselves and otherwise use the configured default (initially A2A3).

### Auto

```cpp
#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWSUM(dst, src, tmp);
}
```

### Manual

```cpp
#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TROWSUM(dst, src, tmp);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = trowsum %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
