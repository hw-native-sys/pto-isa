# pto.tmatmul

`pto.tmatmul` is part of the [Matrix And Matrix Vector](../../matrix-and-matrix-vector.md) instruction set.

## Summary

Matrix multiply (GEMM) on tile operands, producing a result in an accumulator tile.

## Mechanism

Let:
- `M = aMatrix.GetValidRow()`
- `K = aMatrix.GetValidCol()`
- `N = bMatrix.GetValidCol()`

For `0 <= i < M` and `0 <= j < N`:

$$ \mathrm{C}_{i,j} = \sum_{k=0}^{K-1} \mathrm{A}_{i,k} \cdot \mathrm{B}_{k,j} $$

The operation consumes three tiles with specific roles: `aMatrix` (left input, `TileType::Left`), `bMatrix` (right input, `TileType::Right`), and `cMatrix` (accumulator, `TileType::Acc`). The accumulator tile may start with existing values (accumulation semantics) or may be zero-initialized first.

Accumulator behavior and datatype promotion are concrete per target. On A2/A3: accumulation uses the accumulator tile's native datatype (int32_t or float), with zero-initialization performed implicitly before the first phase; subsequent phases accumulate in-place. On A5: accumulation is always in the accumulator tile's native type, and multi-phase accumulation follows a fixed sequence with no implicit zero-initialization between phases. On CPU simulator: accumulation follows A5 semantics by default but may be configurable.

## Syntax

### PTO Assembly Form

```text
%acc = tmatmul %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%c = pto.tmatmul %a, %b : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tmatmul ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>)
          outs(%c : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
// Basic matmul
template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix,
                            WaitEvents &... events);

// Matmul with accumulator phase
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix,
                            WaitEvents &... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `cMatrix` | Destination accumulator tile. Must be `TileType::Acc`. |
| `aMatrix` | Left input tile. Must be `TileType::Left`. |
| `bMatrix` | Right input tile. Must be `TileType::Right`. |
| `Phase` | Optional accumulator phase for multi-pass matmul. |
| `events...` | Optional `RecordEvent` tokens to wait on before issuing. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | `RecordEvent` | Token signaling completion |

The accumulator tile holds `dst[i,j]` = sum over `k` of `a[i,k] * b[k,j]` after the operation.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated tile traffic.

## Constraints

- **Shape constraints**:
  - `aMatrix.GetValidRow()` = `cMatrix.GetValidRow()` = `M`
  - `aMatrix.GetValidCol()` = `bMatrix.GetValidRow()` = `K`
  - `bMatrix.GetValidCol()` = `cMatrix.GetValidCol()` = `N`
  - Runtime `m/k/n` must be in `[1, 4095]`
- **Tile roles**: `aMatrix.Loc == Left`, `bMatrix.Loc == Right`, `cMatrix.Loc == Acc`.
- Programs must not assume implicit broadcasting, reshaping, or valid-region repair.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier.
- Programs must not rely on behavior outside the documented legal domain.

## Target-Profile Restrictions

**A2/A3**:
- Supported `(CType, AType, BType)` triples:
  - `(int32_t, int8_t, int8_t)`
  - `(float, half, half)`
  - `(float, float, float)`
  - `(float, bfloat16_t, bfloat16_t)`
- Static shape: `TileLeft::Rows == TileRes::Rows`, `TileLeft::Cols == TileRight::Rows`, `TileRight::Cols == TileRes::Cols`.

**A5**:
- Accumulator type: `int32_t` or `float`.
- If `int32_t`: `AType == int8_t` and `BType == int8_t`.
- If `float`: supports `half/bfloat16_t/float` and selected fp8 pairs.
- Fractal/layout constraints:
  - Left: `Loc == Left`, `!isRowMajor`, `SFractal == RowMajor`
  - Right: `Loc == Right`, `isRowMajor`, `SFractal == ColMajor`
  - Acc: `Loc == Acc`, `!isRowMajor`, `SFractal == RowMajor`

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using A = TileLeft<half, 16, 16>;
  using B = TileRight<half, 16, 16>;
  using C = TileAcc<float, 16, 16>;
  A a;
  B b;
  C c;
  TMATMUL(c, a, b);
}
```

## See Also

- Instruction set overview: [Matrix And Matrix Vector](../../matrix-and-matrix-vector.md)
- Previous op in instruction set: [pto.tmatmul_mx](./tmatmul-mx.md)
- Next op in instruction set: [pto.tmatmul_acc](./tmatmul-acc.md)
