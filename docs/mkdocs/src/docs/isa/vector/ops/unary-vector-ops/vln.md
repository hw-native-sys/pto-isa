<!-- Generated from `docs/isa/vector/ops/unary-vector-ops/vln.md` -->

# pto.vln

Standalone reference page for `pto.vln`. This page belongs to the [Unary Vector Ops](../../unary-vector-ops.md) family in the PTO ISA manual.

## Summary

`%result` holds the natural logarithm per active lane.

## Mechanism

`pto.vln` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vln %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

Documented A5 types or forms: `f16, f32`.

## Inputs

`%input` is the source vector and `%mask` selects active lanes.

## Expected Outputs

`%result` holds the natural logarithm per active lane.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Only floating-point element types are legal.
  For real-number semantics, active inputs SHOULD be strictly positive; non-
  positive inputs follow the target's exception/NaN rules.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Target-defined numeric exceptional behavior, such as divide-by-zero or out-of-domain inputs, remains subject to the selected backend profile unless this page narrows it further.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `f16, f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = logf(src[i]);
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = logf(src[i]);
```

## Related Ops / Family Links

- Family overview: [Unary Vector Ops](../../unary-vector-ops.md)
- Previous op in family: [pto.vexp](./vexp.md)
- Next op in family: [pto.vsqrt](./vsqrt.md)
