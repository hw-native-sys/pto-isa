# pto.mte_gm_l1

`pto.mte_gm_l1` is part of [Cube Data Movement Ops](../../README.md#cube-data-movement-ops).

## Summary

Structured GM→L1 (cube CBUF) copy. Copies grouped byte ranges from `%src` in GM to `%dst` in L1 without performing any layout transform — the source bytes are written to L1 verbatim.

Use [`pto.mte_gm_l1_frac`](./mte-gm-l1-frac.md) when the source is row-major ND data that needs ND→NZ fractal repack before it can serve as a cube operand.

## Mechanism

Like the scalar [`pto.mte_gm_ub`](../../../scalar/ops/dma-copy/copy-gm-to-ubuf.md), this op uses the grouped `nburst(...) [loop(...)]*` model. For each `nburst` row, source and destination advance by `src_stride` / `dst_stride`. Optional outer `loop(...)` groups wrap the inner transfer.

## Syntax

```mlir
pto.mte_gm_l1 %src, %dst, %len_burst
  nburst(%count, %src_stride, %dst_stride)
  [loop(%count_i, %src_stride_i, %dst_stride_i)]*
  : !pto.ptr<T, gm>, !pto.ptr<T, l1>, i64, i64, i64, i64
```

## Inputs

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | GM source base pointer |
| `%dst` | ptr | L1 destination base pointer (`!pto.ptr<T, l1>`) |
| `%len_burst` | i64 | Bytes copied per burst row |
| `nburst(%count, %src_stride, %dst_stride)` | i64 triple | Innermost burst count and byte strides between row starts |
| `loop(%count_i, %src_stride_i, %dst_stride_i)` | i64 triple | Optional outer repetition; byte advances between enclosed patterns |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | Writes data into the L1 destination region. |

## Side Effects

Reads GM-visible storage; writes L1-visible storage. Engages the AIC MTE2 pipe.

## Constraints

!!! warning "Constraints"
    - `nburst(...)` is required.
    - Each `loop(...)` group must provide all three operands.
    - All strides are bytes. For a contiguous 16-element f16 vector, use `%len_burst = 32`.

## Examples

```mlir
pto.mte_gm_l1 %bias_gm, %l1_bias, %c32_i64
  nburst(%c4_i64, %c64_i64, %c32_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, i64, i64, i64, i64
```

## Related Ops

- ND→NZ repack: [pto.mte_gm_l1_frac](./mte-gm-l1-frac.md)
- L1 → UB: [pto.mte_l1_ub](./mte-l1-ub.md)
- L1 → cube operand tiles: [pto.mte_l1_l0a](./mte-l1-l0a.md), [pto.mte_l1_l0b](./mte-l1-l0b.md)
