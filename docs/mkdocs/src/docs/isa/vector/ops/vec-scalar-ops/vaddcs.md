<!-- Generated from `docs/isa/vector/ops/vec-scalar-ops/vaddcs.md` -->

# pto.vaddcs

Standalone reference page for `pto.vaddcs`. This page belongs to the [Vec Scalar Ops](../../vec-scalar-ops.md) family in the PTO ISA manual.

## Summary

Add with carry-in and carry-out.

## Mechanism

`pto.vaddcs` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask
```

## Inputs

`%lhs` and `%rhs` are the value vectors, `%carry_in` is the
  incoming carry predicate, and `%mask` selects active lanes.

## Expected Outputs

`%result` is the arithmetic result and `%carry` is the carry-out
  predicate.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

This is the scalar-extended carry-chain
  family. Treat it as an unsigned integer operation unless the verifier states a
  wider legal domain.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i] + carry_in[i];
    dst[i] = (T)r;
    carry_out[i] = (r >> bitwidth);
}
```

## Detailed Notes

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i] + carry_in[i];
    dst[i] = (T)r;
    carry_out[i] = (r >> bitwidth);
}
```

## Related Ops / Family Links

- Family overview: [Vec Scalar Ops](../../vec-scalar-ops.md)
- Previous op in family: [pto.vlrelu](./vlrelu.md)
- Next op in family: [pto.vsubcs](./vsubcs.md)
