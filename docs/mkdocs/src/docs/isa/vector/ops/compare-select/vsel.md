<!-- Generated from `docs/isa/vector/ops/compare-select/vsel.md` -->

# pto.vsel

Standalone reference page for `pto.vsel`. This page belongs to the [Compare And Select](../../compare-select.md) family in the PTO ISA manual.

## Summary

Per-lane select based on mask.

## Mechanism

`pto.vsel` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vsel %src0, %src1, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

## Inputs

`%src0` is the true-path vector, `%src1` is the false-path vector,
  and `%mask` selects between them.

## Expected Outputs

`%result` is the selected vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Source vectors and result MUST have matching
  vector shapes and element types.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src0[i] : src1[i];
```

```mlir
// dst = mask ? true_vals : false_vals
%result = pto.vsel %true_vals, %false_vals, %condition
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src0[i] : src1[i];
```

**Example — Conditional assignment:**
```mlir
// dst = mask ? true_vals : false_vals
%result = pto.vsel %true_vals, %false_vals, %condition
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## Related Ops / Family Links

- Family overview: [Compare And Select](../../compare-select.md)
- Previous op in family: [pto.vcmps](./vcmps.md)
- Next op in family: [pto.vselr](./vselr.md)
