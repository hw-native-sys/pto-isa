# pto.vusqz

`pto.vusqz` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Expand a front-packed stream back into the positions selected by a mask.

## Mechanism

`pto.vusqz` is the inverse placement form of `pto.vsqz`. It consumes an implicit front-packed source stream and scatters those elements into the lanes selected by `%mask`; lanes not selected by the mask are zero-filled.

## Syntax

### PTO Assembly Form

```text
vusqz %dst, %mask
```

### AS Level 1 (SSA)

```mlir
%result = pto.vusqz %mask : !pto.mask<G> -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %mask | `!pto.mask<G>` | Predicate mask that selects the lanes that should receive front-packed elements |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Expanded vector image with selected lanes filled and other lanes zeroed |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - The front-packed source stream is implicit in the selected instruction form and target profile.
    - Lane placement for active and inactive positions MUST be preserved exactly.
    - Unselected lanes are zero-filled.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Code that depends on an instruction-set-specific packing, selector, or permutation mode should treat that dependency as target-profile-specific unless the manual states cross-target portability explicitly.

## Examples

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src_front[j++];
    else dst[i] = 0;
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vsqz](./vsqz.md)
- Next op in instruction set: [pto.vperm](./vperm.md)
