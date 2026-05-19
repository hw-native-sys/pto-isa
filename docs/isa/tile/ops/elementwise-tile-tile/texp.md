# pto.texp

`pto.texp` is part of the [Elementwise Tile Tile](../../elementwise-tile-tile.md) instruction set.

## Summary

Elementwise exponential.

## Mechanism

Elementwise exponential.

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \exp(\mathrm{src}_{i,j}) $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = texp %src : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.texp %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.texp ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <auto PrecisionType = ExpAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TEXP(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

`PrecisionType` has the following values available:

- `ExpAlgorithm::DEFAULT`: Normal algorithm, faster but with lower precision.
- `ExpAlgorithm::HIGH_PRECISION`: High precision algorithm, but slower.

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `%src` | Source tile | Source tile; read at `(i, j)` for each `(i, j)` in `dst` valid region |
| `%dst` | Destination tile | Destination tile receiving the result |
| `WaitEvents...` | Optional synchronisation | `RecordEvent` tokens to wait on before issuing the operation |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst` | `!pto.tile<...>` | Destination tile; all `(i, j)` in its valid region contain `exp(src[i,j])` after the operation |

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

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - **Implementation checks (NPU)**:
        - `TileData::DType` must be one of: `float` or `half`;
        - Tile location must be vector (`TileData::Loc == TileType::Vec`);
        - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`;
        - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`;
        - Tile layout must be row-major (`TileData::isRowMajor`).

    - **High precision algorithm**:
        - Only available on A5. `PrecisionType` is ignored on A3.

## Performance

### A2/A3 Throughput

`TEXP` is a transcendental tile operation compiled to CCE SFU instructions via the `TUnaryOp.hpp` performance model:

| Metric | Value (f32) | Value (f16) |
|--------|-------------|-------------|
| Startup latency | 13 (`A2A3_STARTUP_REDUCE`) | 13 |
| Completion latency | 26 (`A2A3_COMPL_FP32_EXP`) | 28 (`A2A3_COMPL_FP16_EXP`) |
| Per-repeat throughput | 2 | 4 |
| Pipeline interval | 18 | 18 |

**Example**: 16×64 f32 tile:

```
R = 16 × 64 / 8 = 128
total ≈ 13 + 26 + 256 + (128-1) × 18 = 2571 cycles
```

**Note**: `TEXP` is significantly more expensive than `TADD`/`TMUL` due to SFU pipeline. For numerically stable softmax kernels, prefer the vector-level `vexpdif` fused operation instead.

---

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TEXP(dst, src);
  TEXP<ExpAlgorithm::HIGH_PRECISION>(dst, src);  // A5 only
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
  TEXP(dst, src);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.texp %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.texp %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = texp %src : !pto.tile<...>
# AS Level 2 (DPS)
pto.texp ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Elementwise Tile Tile](../../elementwise-tile-tile.md)
- Previous op in instruction set: [pto.tsqrt](./tsqrt.md)
- Next op in instruction set: [pto.tnot](./tnot.md)
