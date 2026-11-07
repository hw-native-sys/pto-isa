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

### Public enumerations

The following public enums in `include/pto/common/type.hpp` parameterise instruction behaviour and appear in intrinsic signatures. They are part of the public API and are documented here so instruction pages can refer to them by name without re-defining the value set.

- **`VecStoreMode`** (`type.hpp:222`) — selects the vector-store address layout. `ND` (0) = n-dimensional (contiguous row-major), `NZ` (1) = fractal NZ layout. Used by `TSTORE`/`TSTORE_FP` overloads that emit a vector-path store.
- **`TCopyMode`** (`type.hpp:265`) — selects the copy semantic for tile copies. `SHALLOW_COPY` (0) = share the underlying buffer (metadata-only), `DEEP_COPY` (1) = allocate and copy the buffer.
- **`TileLayoutCustom`** (`type.hpp:370`) — explicit layout tag for custom-layout tiles. Values: `ND`, `DN`, `NZ`, `ZN`, `ZZ`, `NONE`. `NONE` leaves the layout to be inferred from the tile constructor; the other values pin the layout explicitly and bypass inference.
- **`SqrtAlgorithm` / `RsqrtAlgorithm` / `FmodAlgorithm`** (`type.hpp:409` / `:411` / `:419`) — algorithm selector for the corresponding math intrinsics. `DEFAULT` = the backend's default precision path, `HIGH_PRECISION` = the higher-precision path (may trade throughput for accuracy). The same `DEFAULT` / `HIGH_PRECISION` pattern is shared by the sibling math-algorithm enums `PowAlgorithm`, `DivAlgorithm`, `RecipAlgorithm`, `ExpAlgorithm`, `LogAlgorithm`, `FmodSAlgorithm`, `RemAlgorithm`, and `RemSAlgorithm` (`type.hpp:405`–`425`); see the intrinsic signatures for which selector each instruction accepts.

## Events and synchronization

- Instructions may require ordering between memory and vector pipelines. When examples show events (e.g., `set_flag(...)` / `wait_flag(...)`), they indicate the required ordering constraints on the target backend.
- event synchronization is used for explicit synchronization when needed by a sequence of instructions.

See `docs/coding/Event.md` for the event model used by PTO Tile Lib.

## Architecture-specific convention index

Some conventions are too long to inline here and live in dedicated pages. Each entry below pairs the convention with the proven testcase that exercises it, so a developer can jump from the rule to working code in one hop.

| Convention | Document | Proven testcase / source |
|-----------|----------|--------------------------|
| A5 mix cross-core flag pairing (Cube↔Vec, GM/UB paths, shadow event id) | [a5-cross-core-sync-conventions.md](a5-cross-core-sync-conventions.md) | `tests/npu/a5/src/st/testcase/tload_mix`, `tests/npu/a5/src/st/testcase/tmov_acc2vec`, `tests/npu/a5/src/st/testcase/tinsert_acc2vec`; impl `include/pto/npu/a5/custom/TSync_Custom.hpp` |
| A5 on-chip buffer reuse barriers (L0A/L0B/L0C/UB pipe ownership, loop-reuse barrier position) | [a5-on-chip-storage-reuse.md](a5-on-chip-storage-reuse.md) | `tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp:286-287` (M→MTE2 barrier), `:195-196` (M→FIX barrier) |
| A5 UB multi-tile spacing and default base address | [a5-on-chip-storage-reuse.md](a5-on-chip-storage-reuse.md#ub-multi-tile-layout--offset-spacing-and-default-base) | `tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp:234-242` (TASSIGN block, `0x0` / `0x10000` / `0x20000` spacing); check set and default base in [TASSIGN.md](TASSIGN.md) |
| Error message convention (`Fix:` prefix) | [conventions.md](#error-messages) § Error messages below | `include/pto/npu/kirin9030/TGather.hpp:32` (reference example) |
| Public API naming convention (`T` prefix) | [conventions.md](#public-api-naming) § Public API naming below | `include/pto/common/pto_instr.hpp` |

## Error messages

A `static_assert` / `PTO_ASSERT` message that fires on a misuse of a PTO intrinsic must let the developer fix the call site **without reading further docs**. The repository convention is to prefix every such message with `Fix: ` and to state the four elements: the intrinsic, the violated constraint, the allowed set, and the remediation.

### Format

```
Fix: <INSTRUCTION> <what was wrong>; <allowed set / remediation>.
```

- **`Fix: ` prefix** — the message is actionable; the developer looks for `Fix:` in the build log to find what to change.
- **`<INSTRUCTION>`** — the intrinsic name in upper-case (e.g. `TGATHER`, `TMATMUL`), so `grep` finds it.
- **`<what was wrong>`** — the violated constraint in one clause (e.g. "Dst data type must be ...").
- **`<allowed set / remediation>`** — the accepted values or the corrective action.

### Reference example

`include/pto/npu/kirin9030/TGather.hpp:32`:

```cpp
static_assert(..., "Fix: TGATHER Dst data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/float.");
```

### Current state

The `Fix:` prefix is applied across `include/pto/npu/{a5,a2a3,kirin9030,kirin9050,kirinX90}` but not yet in `common` or `cpu`. When adding a new `static_assert` against a PTO intrinsic, use the `Fix:` prefix and the four-element form above. When porting an intrinsic to a new backend, carry the `Fix:` messages over from the reference backend rather than rewording them.

## Public API naming

Public PTO intrinsics follow a `T` + operation-name convention. The `T` prefix marks a Tile-level primitive (as opposed to a runtime / utility symbol). The operation name is upper-case and matches the ISA page name.

### Convention

- Default: `T` + upper-case operation name — e.g. `TADD`, `TMATMUL`, `TSTORE`, `TLOAD`, `TMOV`, `TCOLEXPANDADD`.
- Suffix variants append a short qualifier: `TMOV_FP` (fixed-point move-out), `TMATMUL_MX` (MX-format matmul), `TADDS` (saturated add), `TSTORE_FP` (fixpipe store).

### Documented exceptions (no `T` prefix)

The following public intrinsics are **intentionally** not `T`-prefixed because they are control-plane / configuration primitives, not Tile data transforms. They are listed here so that the absence of the `T` prefix is recognised as deliberate rather than a naming drift:

| Name | Why no `T` prefix |
|------|-------------------|
| `SYNCALL` | Cross-core control-plane barrier, not a per-Tile operation. |
| `SETFMATRIX` | Cube configuration (fractal matrix setup), not a Tile data transform. |
| `SET_QUANT_SCALAR` / `SET_QUANT_VECTOR` | Quantization scale binding for MX paths (scalar / vector form), configuration primitives. |
| `SET_IMG2COL_*` | Img2col configuration, not a Tile data transform. |
| `MGATHER` | GM→L1/UB gather; historically named without `T` (the data-movement sibling `TGATHER` exists for the UB→UB form). |
| `MSCATTER` | GM scatter counterpart of `MGATHER`; same historical naming. |

When introducing a new public intrinsic, default to the `T` prefix; only omit it if the intrinsic is a control-plane / configuration primitive in the same category as the exceptions above, and add an entry to this table.
