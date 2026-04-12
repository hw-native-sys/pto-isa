<!-- Generated from `docs/isa/vector/ops/compare-select/vcmps.md` -->

# pto.vcmps

Standalone reference page for `pto.vcmps`. This page belongs to the [Compare And Select](../../compare-select.md) family in the PTO ISA manual.

## Summary

Compare vector against scalar.

## Mechanism

`pto.vcmps` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vcmps %src, %scalar, %seed, "CMP_MODE" : !pto.vreg<NxT>, T, !pto.mask -> !pto.mask
```

## Inputs

`%src` is the vector source, `%scalar` is the scalar comparison
  value, and `%seed` is the incoming predicate.

## Expected Outputs

`%result` is the generated predicate mask.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

For 32-bit scalar forms, the scalar source
  MUST satisfy the backend's legal scalar-source constraints for this family.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src[i] CMP scalar) ? 1 : 0;
```

```mlir
%positive_mask = pto.vcmps %values, %c0_f32, %all_active, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
// positive_mask[i] = 1 if values[i] > 0
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src[i] CMP scalar) ? 1 : 0;
```

**Example:**
```mlir
%positive_mask = pto.vcmps %values, %c0_f32, %all_active, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
// positive_mask[i] = 1 if values[i] > 0
```

## Selection Operations

## Related Ops / Family Links

- Family overview: [Compare And Select](../../compare-select.md)
- Previous op in family: [pto.vcmp](./vcmp.md)
- Next op in family: [pto.vsel](./vsel.md)
