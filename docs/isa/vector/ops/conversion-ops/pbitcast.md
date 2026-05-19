# pto.pbitcast

`pto.pbitcast` is part of the [Conversion Ops](../../conversion-ops.md) instruction set.

## Summary

`pto.pbitcast` performs a bitwise reinterpretation of a `!pto.mask<...>` value without changing the underlying predicate-register image. It makes mask-family reinterpretation explicit in VPTO IR for cases where a producer and consumer expect different granularity views (`b8`, `b16`, `b32`) of the same hardware predicate state.

## Mechanism

The op is a pure type cast at the mask-register level. No predicate bits are materialized, normalized, or recomputed — VPTO only updates which mask granularity the surrounding IR uses to interpret the same predicate bits. This decouples the mask producer's natural granularity from the consumer's required granularity without inserting an extra hardware operation.

## Syntax

```mlir
%result = pto.pbitcast %input : !pto.mask<G0> -> !pto.mask<G1>
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%input` | `!pto.mask<G0>` | Source predicate register value. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.mask<G1>` | Same predicate bits, reinterpreted under granularity `G1`. |

## Side Effects

`pto.pbitcast` has no architectural side effects beyond producing its SSA result. It does not materialize new mask bits or rewrite hardware predicate state.

## Constraints

!!! warning "Constraints"
    - Both source and result must be `!pto.mask<...>` types.
    - `pto.pbitcast` does not materialize or normalize predicate contents; it only changes which mask granularity the surrounding VPTO IR uses to interpret the same predicate bits.
    - Use only when the consumer requires a different mask granularity (`b8` / `b16` / `b32`) but the underlying predicate-register image is intended to be reused as-is. If the consumer needs a recomputed predicate, lower or materialize the mask through the appropriate predicate-generation op instead of `pto.pbitcast`.

## Examples

### Reinterpret a b16 predicate as b32 before a consumer

```mlir
%m16 = pto.pintlv_b16 %lhs, %rhs
    : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>
%m32 = pto.pbitcast %m16#0 : !pto.mask<b16> -> !pto.mask<b32>
%result = pto.vsel %a, %b, %m32
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Conversion Ops](../../conversion-ops.md)
- Vector-side bitcast: [pto.vbitcast](./vbitcast.md)
- Predicate generation and algebra: [Predicate Generation and Algebra](../../../scalar/ops/predicate-generation-and-algebra/)
