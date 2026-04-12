<!-- Generated from `docs/isa/vector/ops/reduction-ops/vcgmin.md` -->

# pto.vcgmin

Standalone reference page for `pto.vcgmin`. This page belongs to the [Reduction Ops](../../reduction-ops.md) family in the PTO ISA manual.

## Summary

Min within each VLane.

## Mechanism

`pto.vcgmin` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vcgmin %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

Documented A5 types or forms: `i16-i32, f16, f32`.

## Inputs

`%input` is the source vector and `%mask` selects participating
  lanes.

## Expected Outputs

`%result` contains one minimum per 32-byte VLane group.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Grouping is by hardware 32-byte VLane, not by
  arbitrary software subvector.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `i16-i32, f16, f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
int K = N / 8;
for (int g = 0; g < 8; g++) {
    T mn = INF;
    for (int i = 0; i < K; i++)
        if (src[g*K + i] < mn) mn = src[g*K + i];
    dst[g*K] = mn;
    for (int i = 1; i < K; i++)
        dst[g*K + i] = 0;
}
```

## Detailed Notes

```c
int K = N / 8;
for (int g = 0; g < 8; g++) {
    T mn = INF;
    for (int i = 0; i < K; i++)
        if (src[g*K + i] < mn) mn = src[g*K + i];
    dst[g*K] = mn;
    for (int i = 1; i < K; i++)
        dst[g*K + i] = 0;
}
```

## Prefix Operations

## Related Ops / Family Links

- Family overview: [Reduction Ops](../../reduction-ops.md)
- Previous op in family: [pto.vcgmax](./vcgmax.md)
- Next op in family: [pto.vcpadd](./vcpadd.md)
