# pto.vbitsort

`pto.vbitsort` is part of the [SFU And DSA Instructions](../../sfu-and-dsa-ops.md) instruction set.

## Summary

Sort 32 score/index pairs in descending score order and materialize the sorted records into UB.

## Mechanism

`pto.vbitsort` is a UB-to-UB accelerator operation. It reads score values from `%src`, uses `%indices` as the companion index stream, sorts the pairs by score in descending order, and writes packed output records to `%dest`.

## Syntax

### PTO Assembly Form

```mlir
pto.vbitsort %dest, %src, %indices, %repeat_times : (!pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<i32, ub>, index) -> ()
```

### AS Level 1 (SSA)

```mlir
pto.vbitsort %dest, %src, %indices, %repeat_times : (!pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<i32, ub>, index) -> ()
```

## C++ Intrinsic

The installed Bisheng public intrinsic exposes both score-only and score-plus-index overloads; this page corresponds to the score-plus-index `VBS32` form.

```cpp
__ubuf__ float *dst;
__ubuf__ float *scores;
__ubuf__ unsigned int *indices;
uint8_t repeat = 1;
vbitsort(dst, scores, indices, repeat);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%dest` | `!pto.ptr<T, ub>` | UB destination buffer for sorted output records |
| `%src` | `!pto.ptr<T, ub>` | UB source buffer containing score values |
| `%indices` | `!pto.ptr<i32, ub>` | UB source buffer containing companion indices |
| `%repeat_times` | `index` | Number of adjacent 32-element groups to process |

## Expected Outputs

This op writes UB memory directly and returns no SSA value. Each output record contains the original index together with the associated score, ordered by descending score.

## Side Effects

This operation mutates `%dest` in UB memory. It does not reserve buffers, signal events, or establish fences beyond the visible memory write.

## Constraints

- `%dest`, `%src`, and `%indices` MUST all refer to UB-backed storage.
- The hardware operation processes 32-element groups; `%repeat_times` scales that fixed group size.
- The page-level contract assumes descending score order.
- Buffer alignment and layout MUST satisfy the target-profile backend contract for the selected form.

## Exceptions

- The verifier rejects illegal pointer spaces or unsupported element types.
- Illegal repeat counts or malformed buffer contracts are target-profile-specific errors.

## Target-Profile Restrictions

- This operation is documented as A5-oriented sort acceleration.
- CPU simulation may preserve the visible PTO contract with a software fallback.
- Availability on narrower profiles is target-specific and should not be assumed without profile documentation.

## Performance

`pto.vbitsort` is a dedicated sort helper rather than a standard vector ALU opcode. For throughput-sensitive code, batch work in 32-element groups and keep the source and destination buffers in UB.

## Examples

### Sort one 32-element group

```mlir
pto.vbitsort %sorted_records, %score_buf, %idx_buf, %c1
    : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<i32, ub>, index) -> ()
```

### NMS-style pipeline setup

```mlir
%idx = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
pto.copy_vreg_to_ub %idx_ub, %idx, %c64 : !pto.ptr<i32, ub>, !pto.vreg<64xi32>, index
pto.vbitsort %sorted_records, %scores_ub, %idx_ub, %c1
    : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<i32, ub>, index) -> ()
```

## Detailed Notes

`pto.vbitsort` is the score-ordering primitive in proposal and top-K style pipelines. The installed public C++ surface also exposes a score-only `VBS16` form, but this PTO page documents the score-plus-index form that preserves original element identity through the sorted output stream.

## Related Ops / Instruction Set Links

- Instruction set overview: [SFU And DSA Instructions](../../sfu-and-dsa-ops.md)
- Previous op in instruction set: [pto.vsort32](./vsort32.md)
- Next op in instruction set: [pto.vmrgsort](./vmrgsort.md)
