# pto.tci

`pto.tci` is part of the [Irregular And Complex](../../irregular-and-complex.md) instruction set.

## Summary

Generate a contiguous integer sequence into a destination tile.

## Mechanism

Generate a contiguous integer sequence into a destination tile. It belongs to the tile instructions and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

For a linearized index `k` over the valid elements:

- Ascending:

  $$ \mathrm{dst}_{k} = S + k $$

- Descending:

  $$ \mathrm{dst}_{k} = S - k $$

The linearization order depends on the tile layout. On A2/A3 and A5, the linearization order follows row-major order: elements are visited left-to-right within each row, then top-to-bottom across rows.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tci %S {descending = false} : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData &dst, T start, WaitEvents &... events);
```

## Inputs

- `start` is the starting integer value for the sequence.
- `descending` (template parameter): if true, generates descending sequence.
- `dst` names the destination tile. The operation iterates over dst's valid region.

## Expected Outputs

`dst` holds a contiguous integer sequence starting from `start`.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    - **Valid region**:
        - The implementation uses `dst.GetValidCol()` as the sequence length and does not consult `dst.GetValidRow()`.

## Performance

### A2/A3 Cycle Count

`pto.tci` lowers to a vector-pipe sequence generator. The cost is linear in `R × ⌈C / vlen⌉` PIPE_V issues plus a small startup. The scalar `start` is materialised once and incremented in-pipe.

**Cycle model**: `total ≈ startup + R × ⌈C / vlen⌉ × (per_issue + interval)`.

### Instruction Sequence by Shape (FP32 / int32)

| Valid Shape | Instruction Sequence | Estimated Cycles |
|-------------|----------------------|------------------|
| 1×16  | `vci` → PIPE_V | ~O(8) |
| 1×64  | `vci`*1 → PIPE_V | ~O(16) |
| 16×16 | `vci`*16 → PIPE_V | ~O(64) |
| R×C   | `vci`*R → PIPE_V  | ~O(R × ⌈C/vlen⌉) |

Descending mode (`descending = true`) has the same cost as the ascending path.

> Note: cycle numbers below are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - **Implementation checks (A2A3/A5)**:
        - `TileData::DType` must be exactly the same type as the scalar template parameter `T`.
        - `dst/scalar` element types must be identical, and must be one of: `int32_t`, `uint32_t`, `int16_t`, `uint16_t`.
        - `TileData::Cols != 1` (this is the condition enforced by the implementation).

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, int32_t, 1, 16>;
  TileT dst;
  TCI<TileT, int32_t, /*descending=*/0>(dst, /*S=*/0);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, int32_t, 1, 16>;
  TileT dst;
  TASSIGN(dst, 0x1000);
  TCI<TileT, int32_t, /*descending=*/1>(dst, /*S=*/100);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tci %S {descending = false} : !pto.tile<...>
# AS Level 2 (DPS)
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```

## See Also

- Instruction set overview: [Irregular And Complex](../../irregular-and-complex.md)
- Previous op in instruction set: [pto.tscatter](./tscatter.md)
- Next op in instruction set: [pto.ttri](./ttri.md)

