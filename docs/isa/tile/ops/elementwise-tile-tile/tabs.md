# pto.tabs

`pto.tabs` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

Elementwise absolute value of a tile.

## Mechanism

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \left|\mathrm{src}_{i,j}\right| $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tabs %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tabs %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tabs ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TABS(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `%src` | Source tile | Source tile; read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `%dst` | Destination tile | Destination tile receiving the result |
| `WaitEvents...` | Optional synchronisation | `RecordEvent` tokens to wait on before issuing the operation |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.tile<...>` | Destination tile; all `(i, j)` in its valid region contain `|src[i,j]|` after the operation |

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - **Valid region**:
        - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Performance

### A2/A3 Throughput

`TABS` compiles to CCE vector instructions via the `TUnaryOp.hpp` performance model:

| Metric | Value |
|--------|-------|
| Startup latency | 13 |
| Completion latency | 26 (FP transcendental) |
| Per-repeat throughput | 1 |
| Pipeline interval | 18 |

---

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - **Implementation checks (CPU sim)**:
        - `TileData::DType` must be one of: `int32_t`, `int`, `int16_t`, `int8_t`, `half`, `bfloat16_t`, `float`.
        - The implementation iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.

    - **Implementation checks (Costmodel)**:
        - `TileData::DType` must be one of: `int32_t`、`int16_t`、`int8_t`、`uint8_t`、`half`、`float`.

    - **Implementation checks (NPU)**:
        - For A3, `TileData::DType` must be one of: `float` or `half`;
        - For A5, `TileData::DType` must be one of: `float`, `half`, `int32_t`, `int16_t`, or `int8_t`;
        - Tile location must be vector (`TileData::Loc == TileType::Vec`);
        - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`;
        - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`;
        - Tile layout must be row-major (`TileData::isRowMajor`).

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TABS(dst, src);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TABS(dst, src);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tabs %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tabs %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tabs %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tabs ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.tadd](./tadd.md)
- Next op in instruction set: [pto.tand](./tand.md)
