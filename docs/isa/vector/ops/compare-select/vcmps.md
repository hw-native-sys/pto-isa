# pto.vcmps

`pto.vcmps` is part of the [Compare And Select](../../compare-select.md) instruction set.

## Summary

Element-wise vector-to-scalar comparison that produces a predicate mask.

## Mechanism

For each lane `i` where `%seed[i]` is true, `result[i]` is set to the outcome of applying `CMP_MODE` to `src[i]` and the broadcast scalar. Lanes disabled by `%seed` produce a zero bit in the result mask.

## Syntax

### PTO Assembly Form

```text
vcmps %dst, %src, %scalar, %seed, "CMP_MODE" : !pto.mask<G>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vcmps %src, %scalar, %seed, "CMP_MODE" : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.mask<G>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %src | `!pto.vreg<NxT>` | Vector operand |
| %scalar | `T` | Scalar comparison value broadcast to every active lane |
| %seed | `!pto.mask<G>` | Incoming predicate mask that limits which lanes are compared |
| `CMP_MODE` | enum | Comparison predicate such as `eq`, `ne`, `lt`, `le`, `gt`, or `ge` |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.mask<G>` | Predicate mask whose active bits record the comparison result |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - The seed mask width MUST match `N`.
    - The scalar source MUST satisfy any backend-specific scalar-source legality rule for this instruction family.
    - Floating-point and integer comparisons follow the element-type-specific comparison rules of the selected target profile.
    Supported compare modes are `eq`, `ne`, `lt`, `le`, `gt`, and `ge`.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Code that depends on an implicit predicate source or a target-specific encoding variant should treat that dependency as target-profile-specific unless the manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    result[i] = seed[i] ? cmp(src[i], scalar, CMP_MODE) : 0;
```

```mlir
%positive_mask = pto.vcmps %values, %c0_f32, %all_active, "gt" : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Compare And Select](../../compare-select.md)
- Previous op in instruction set: [pto.vcmp](./vcmp.md)
- Next op in instruction set: [pto.vsel](./vsel.md)
