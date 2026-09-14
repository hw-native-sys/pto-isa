# PTO ISA Conventions

This page defines shared conventions used by the per-instruction ISA reference pages in `docs/isa/` and the corresponding C++ intrinsics in `include/pto/common/pto_instr.hpp`.

## Notation

- **Tile**: A fixed-size on-chip tile object (e.g., `pto::Tile<...>`). Many instructions operate on tiles and use the tile’s valid region (`GetValidRow()`, `GetValidCol()`).
- **GM (global memory)**: Off-chip memory accessed via `pto::GlobalTensor<...>`.
- **Scalar / immediate**: A host-side scalar value or an encoded immediate used by `*S` / `*C` variants.

For the detailed C++ programming model behind these terms, see:

- Tiles: `docs/coding/Tile.md`
- GlobalTensor: `docs/coding/GlobalTensor.md`
- Scalars and enums: `docs/coding/Scalar.md`

## Shapes and layouts

- **Row-major vs. column-major**: Unless stated otherwise, CPU simulator kernels assume row-major tiles. Instructions that support multiple layouts will state supported layouts explicitly.
- **Valid region**: The runtime compute region of a tile, expressed as `(valid_row, valid_col)` and queried via `GetValidRow()` / `GetValidCol()`.

### Physical shape and valid shape

`Rows` / `Cols` describe the physical storage shape; `GetValidRow()` / `GetValidCol()` describe the active region. Changing the valid shape does not change physical strides. For a non-fractal tile, element `(i, j)` has element offset `i * RowStride + j * ColStride`: RowMajor strides are `(Cols, 1)` and ColMajor strides are `(1, Rows)`.

For example, an A5 `int64_t` / `uint64_t` RowMajor output with physical shape `[64,4]` and valid shape `[64,1]` still has a row stride of 4 elements (32 bytes). Row-reduction results occupy offsets `0, 4, 8, ...`; using valid column count 1 as the physical stride misplaces results between rows. A compact ColMajor `[64,1]` output instead uses consecutive element offsets.

The 32-byte alignment requirement for 64-bit Vec tiles applies to physical dimensions: RowMajor requires `Cols % 4 == 0`, and ColMajor requires `Rows % 4 == 0`. Valid dimensions may be smaller, subject to instruction constraints. Allocate the complete physical tile; padding outside the valid region is distinct from adjacent independent allocations. See each instruction's padding-write semantics, particularly [TSCATTER](TSCATTER.md).

### Valid Region Semantics

For instruction pages, when we say “for each element `(i, j)` in the valid region”, we mean:

- `valid_row = dst.GetValidRow()` and `valid_col = dst.GetValidCol()` unless the instruction explicitly defines a different domain (e.g., some ops may use the source tile’s valid region).
- The math interpretation defines `dst[i, j]` only for indices where `0 <= i < valid_row` and `0 <= j < valid_col`.
- Elements outside the valid region are **unspecified** unless the instruction explicitly states otherwise (do not assume they are zeroed or preserved).

For multi-operand instructions (e.g., `src0`, `src1`), the docs assume the input tiles are compatible with the iteration domain unless the constraints section states stricter requirements.

## Types

- The instruction page lists supported data types (e.g., `fp16`, `fp32`, `int8`, `int16`, `int32`, `uint8`, `uint16`, `uint32`). CPU simulator support may be a subset and is documented in `include/README.md`.

## Events and synchronization

- Instructions may require ordering between memory and vector pipelines. When examples show events (e.g., `set_flag(...)` / `wait_flag(...)`), they indicate the required ordering constraints on the target backend.
- event synchronization is used for explicit synchronization when needed by a sequence of instructions.

See `docs/coding/Event.md` for the event model used by PTO Tile Lib.
