# pto.vsqz

`pto.vsqz` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Compress active lanes to the front of the destination vector.

## Mechanism

`pto.vsqz` packs the lanes selected by `%mask` toward the front of the result while preserving their original lane order. Remaining lanes in the destination are zero-filled.

## Syntax

### PTO Assembly Form

```text
vsqz %dst, %src, %mask
```

### AS Level 1 (SSA)

```mlir
%result = pto.vsqz %src, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %src | `!pto.vreg<NxT>` | Source vector |
| %mask | `!pto.mask` | Predicate mask selecting which lanes are kept |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Compacted vector with selected lanes at the front and zeros in the tail |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- The mask width MUST match `N`.
- The relative order of surviving lanes MUST match their original lane order.
- Unselected destination lanes are zero-filled.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific packing, selector, or permutation mode should treat that dependency as target-profile-specific unless the manual states cross-target portability explicitly.

## Examples

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[j++] = src[i];
while (j < N) dst[j++] = 0;
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vshift](./vshift.md)
- Next op in instruction set: [pto.vusqz](./vusqz.md)
