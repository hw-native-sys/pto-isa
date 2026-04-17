# pto.vsunpack

`pto.vsunpack` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Unpack one half of a narrow vector with sign extension.

## Mechanism

`pto.vsunpack` widens one selected half of the source vector. Each narrow element is sign-extended into the wider destination element type.

## Syntax

### PTO Assembly Form

```text
vsunpack %dst, %src, %part
```

### AS Level 1 (SSA)

```mlir
%result = pto.vsunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %src | `!pto.vreg<NxT_narrow>` | Packed narrow source vector |
| %part | `index` | Selector for which half of the source vector to unpack |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<N/2xT_wide>` | Widened vector with sign extension |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- The selected half and widening mode MUST be supported by the target profile.
- The widening behavior is sign-extending.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific packing, selector, or permutation mode should treat that dependency as target-profile-specific unless the manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N/2; i++)
    dst[i] = sign_extend(src[part_offset + i]);
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vpack](./vpack.md)
- Next op in instruction set: [pto.vzunpack](./vzunpack.md)
