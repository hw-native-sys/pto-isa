# pto.vcmp

`pto.vcmp` is part of the [Compare And Select](../../compare-select.md) instruction set.

## Summary

Element-wise vector-to-vector comparison that produces a predicate mask.

## Mechanism

For each lane `i` where `%seed[i]` is true, `result[i]` is set to the outcome of applying `CMP_MODE` to `src0[i]` and `src1[i]`. Lanes disabled by `%seed` produce a zero bit in the result mask.

## Syntax

### PTO Assembly Form

```text
vcmp %dst, %src0, %src1, %seed, "CMP_MODE" : !pto.mask
```

### AS Level 1 (SSA)

```mlir
%result = pto.vcmp %src0, %src1, %seed, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.mask
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %src0 | `!pto.vreg<NxT>` | Left-hand vector operand |
| %src1 | `!pto.vreg<NxT>` | Right-hand vector operand |
| %seed | `!pto.mask` | Incoming predicate mask that limits which lanes are compared |
| `CMP_MODE` | enum | Comparison predicate such as `eq`, `ne`, `lt`, `le`, `gt`, or `ge` |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.mask` | Predicate mask whose active bits record the comparison result |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- `%src0` and `%src1` MUST have the same vector width `N` and element type `T`.
- The seed mask width MUST match `N`.
- Floating-point and integer comparisons follow the element-type-specific comparison rules of the selected target profile.
Supported compare modes are `eq`, `ne`, `lt`, `le`, `gt`, and `ge`.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an implicit predicate source or a target-specific encoding variant should treat that dependency as target-profile-specific unless the manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    result[i] = seed[i] ? cmp(src0[i], src1[i], CMP_MODE) : 0;
```

```mlir
%lt_mask = pto.vcmp %a, %b, %all_active, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Compare And Select](../../compare-select.md)
- Previous op in instruction set: (none)
- Next op in instruction set: [pto.vcmps](./vcmps.md)
