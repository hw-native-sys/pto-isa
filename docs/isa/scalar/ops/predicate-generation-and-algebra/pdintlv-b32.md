# pto.pdintlv_b32

`pto.pdintlv_b32` is part of the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) instruction set.

## Summary

Deinterleave two `b32`-granularity predicate sources and materialize the lower and higher result halves as two predicate outputs.

## Mechanism

`pto.pdintlv_b32` is the 32-bit-element-granularity variant of the predicate-deinterleave family ([`pto.pdintlv_b8`](./pdintlv-b8.md) / [`pto.pdintlv_b16`](./pdintlv-b16.md) / `pto.pdintlv_b32`). It takes two `!pto.mask<b32>` sources and emits the two deinterleaved halves under the same `b32` granularity. The hardware view of the predicate-register image is preserved bit-for-bit; only how the bits are grouped into 32-bit element slots changes.

## Syntax

### AS Level 1 (SSA)

```mlir
%low, %high = pto.pdintlv_b32 %src0, %src1
    : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src0` | `!pto.mask<b32>` | First predicate source |
| `%src1` | `!pto.mask<b32>` | Second predicate source |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%low` | `!pto.mask<b32>` | Lower deinterleaved half produced from `%src0` / `%src1` |
| `%high` | `!pto.mask<b32>` | Upper deinterleaved half produced from `%src0` / `%src1` |

## Side Effects

None. `pto.pdintlv_b32` is a pure predicate transform; it does not read or write UB, GM, or any architectural state beyond producing its two SSA results.

## Constraints

!!! warning "Constraints"
    - All operands and results MUST use `!pto.mask<b32>`. Mixing predicate granularities is illegal; use `pto.pbitcast` first if a producer emits a different granularity.
    - The two outputs form an ordered pair (`%low`, `%high`) and that pairing MUST be preserved.

## Examples

```mlir
%lo, %hi = pto.pdintlv_b32 %m0, %m1
    : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Other variants: [pto.pdintlv_b8](./pdintlv-b8.md), [pto.pdintlv_b16](./pdintlv-b16.md)
- Inverse: [pto.pintlv_b32](./pintlv-b32.md)
- Mask granularity reinterpret: [pto.pbitcast](../../../vector/ops/conversion-ops/pbitcast.md)
