# pto.vmrgsort

`pto.vmrgsort` is part of the [SFU And DSA Instructions](../../sfu-and-dsa-ops.md) instruction set.

## Summary

Merge four pre-sorted UB segments into one sorted UB destination.

## Mechanism

`pto.vmrgsort` is a UB-to-UB merge accelerator. The concrete public mnemonic exposed by the installed Bisheng headers is `vmrgsort4`, which merges four sorted sources according to the configuration word supplied by `%config`.

## Syntax

### PTO Assembly Form

```mlir
pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : (!pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64, i64) -> ()
```

### AS Level 1 (SSA)

```mlir
pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : (!pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64, i64) -> ()
```

## C++ Intrinsic

The installed Bisheng public intrinsic spelling is `vmrgsort4(...)`; one common overload takes an array of four UB source pointers plus a packed configuration word.

```cpp
__ubuf__ float *dst;
__ubuf__ float *src[4];
uint64_t config;
vmrgsort4(dst, src, config);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%dest` | `!pto.ptr<T, ub>` | UB destination buffer for merged output |
| `%src0` | `!pto.ptr<T, ub>` | First pre-sorted source segment |
| `%src1` | `!pto.ptr<T, ub>` | Second pre-sorted source segment |
| `%src2` | `!pto.ptr<T, ub>` | Third pre-sorted source segment |
| `%src3` | `!pto.ptr<T, ub>` | Fourth pre-sorted source segment |
| `%count` | `i64` | Element count per source segment in the PTO surface |
| `%config` | `i64` | Packed configuration word controlling sort/merge behavior |

## Expected Outputs

This op writes merged results to `%dest` in UB memory and returns no SSA value.

## Side Effects

This operation mutates `%dest` in UB memory. It does not reserve buffers, signal events, or establish fences beyond the visible destination write.

## Constraints

- All four source segments MUST already be sorted according to the order encoded by `%config`.
- All pointers MUST target UB storage and agree on element type `T`.
- `%config` is target-profile-specific; code that depends on its bit layout is not portable without profile documentation.
- The PTO family name on this page is `pto.vmrgsort`, but the concrete documented form is `pto.vmrgsort4`.

## Exceptions

- The verifier rejects illegal pointer spaces or unsupported element types.
- Misaligned buffers or illegal configuration layouts are target-profile-specific errors.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual for this family.
- CPU simulation may preserve the visible merge behavior with a software fallback.
- Availability and exact configuration encoding on other profiles are target-specific.

## Performance

### Timing Disclosure

The current public VPTO timing material for PTO micro instructions remains limited.
For `pto.vmrgsort`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vmrgsort`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

### Merge four sorted UB segments

```mlir
pto.vmrgsort4 %dest, %sorted_a, %sorted_b, %sorted_c, %sorted_d, %count, %config
    : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, i64, i64) -> ()
```

### Sort-then-merge pipeline

```mlir
pto.vsort32 %sorted_a, %unsorted_a, %config : !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, i64
pto.vsort32 %sorted_b, %unsorted_b, %config : !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, i64
pto.vsort32 %sorted_c, %unsorted_c, %config : !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, i64
pto.vsort32 %sorted_d, %unsorted_d, %config : !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, i64
pto.vmrgsort4 %dest, %sorted_a, %sorted_b, %sorted_c, %sorted_d, %count, %config
    : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, i64, i64) -> ()
```

## Detailed Notes

`pto.vmrgsort` is the merge-stage companion to local sort helpers such as `pto.vsort32`. The installed C++ surface includes several overload families, including packed-address and list-driven forms, but the page-level PTO contract stays focused on the visible four-way merge behavior.

## Related Ops / Instruction Set Links

- Instruction set overview: [SFU And DSA Instructions](../../sfu-and-dsa-ops.md)
- Previous op in instruction set: [pto.vsort32](./vsort32.md)
