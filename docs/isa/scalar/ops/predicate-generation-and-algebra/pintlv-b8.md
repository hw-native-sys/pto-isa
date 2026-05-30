# pto.pintlv_b8

`pto.pintlv_b8` is part of the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) instruction set.

## Summary

Interleave two `b8`-granularity predicate sources and materialize the lower and higher result halves as two predicate outputs.

## Mechanism

`pto.pintlv_b8` is the 8-bit-element-granularity variant of the predicate-interleave family (`pto.pintlv_b8` / [`pto.pintlv_b16`](./pintlv-b16.md) / [`pto.pintlv_b32`](./pintlv-b32.md)). It takes two `!pto.mask<b8>` sources and emits the two interleaved halves under the same `b8` granularity. The hardware view of the predicate-register image is preserved bit-for-bit; only how the bits are grouped into 8-bit element slots changes.

## Syntax

### AS Level 1 (SSA)

```mlir
%low, %high = pto.pintlv_b8 %src0, %src1
    : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src0` | `!pto.mask<b8>` | First predicate source |
| `%src1` | `!pto.mask<b8>` | Second predicate source |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%low` | `!pto.mask<b8>` | Lower interleaved half produced from `%src0` / `%src1` |
| `%high` | `!pto.mask<b8>` | Upper interleaved half produced from `%src0` / `%src1` |

## Side Effects

None. `pto.pintlv_b8` is a pure predicate transform; it does not read or write UB, GM, or any architectural state beyond producing its two SSA results.

## Constraints

!!! warning "Constraints"
    - All operands and results MUST use `!pto.mask<b8>`. Mixing predicate granularities is illegal; use `pto.pbitcast` first if a producer emits a different granularity.
    - The two outputs form an ordered pair (`%low`, `%high`) and that pairing MUST be preserved.

## Examples

```mlir
%lo, %hi = pto.pintlv_b8 %m0, %m1
    : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Other variants: [pto.pintlv_b16](./pintlv-b16.md), [pto.pintlv_b32](./pintlv-b32.md)
- Inverse: [pto.pdintlv_b8](./pdintlv-b8.md)
- Mask granularity reinterpret: [pto.pbitcast](../../../vector/ops/conversion-ops/pbitcast.md)
