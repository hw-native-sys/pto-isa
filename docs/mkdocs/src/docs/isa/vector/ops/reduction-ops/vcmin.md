<!-- Generated from `docs/isa/vector/ops/reduction-ops/vcmin.md` -->

# pto.vcmin

Standalone reference page for `pto.vcmin`. This page belongs to the [Reduction Ops](../../reduction-ops.md) family in the PTO ISA manual.

## Summary

Find min element with argmin. Result value + index in lane 0.

## Mechanism

`pto.vcmin` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vcmin %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

Documented A5 types or forms: `i16-i32, f16, f32`.

## Inputs

`%input` is the source vector and `%mask` selects participating
  lanes.

## Expected Outputs

`%result` carries the reduction result in the low destination
  positions.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

As with `pto.vcmax`, the exact value/index
  packing depends on the chosen form and MUST be preserved consistently.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `i16-i32, f16, f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
T mn = INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] < mn) { mn = src[i]; idx = i; }
dst_val[0] = mn;
dst_idx[0] = idx;
```

```
vreg layout (f32 example, 64 elements total):
VLane 0: [0..7]   VLane 1: [8..15]  VLane 2: [16..23] VLane 3: [24..31]
VLane 4: [32..39] VLane 5: [40..47] VLane 6: [48..55] VLane 7: [56..63]
```

## Detailed Notes

```c
T mn = INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] < mn) { mn = src[i]; idx = i; }
dst_val[0] = mn;
dst_idx[0] = idx;
```

## Per-VLane (Group) Reductions

The vector register is organized as **8 VLanes** of 32 bytes each. Group reductions operate within each VLane independently.

```
vreg layout (f32 example, 64 elements total):
VLane 0: [0..7]   VLane 1: [8..15]  VLane 2: [16..23] VLane 3: [24..31]
VLane 4: [32..39] VLane 5: [40..47] VLane 6: [48..55] VLane 7: [56..63]
```

## Related Ops / Family Links

- Family overview: [Reduction Ops](../../reduction-ops.md)
- Previous op in family: [pto.vcmax](./vcmax.md)
- Next op in family: [pto.vcgadd](./vcgadd.md)
