# pto.vpack

`pto.vpack` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Pack two wide vectors into one narrower vector.

## Mechanism

`pto.vpack` narrows the two source vectors and concatenates the narrowed halves into one destination vector. The exact narrowing mode is controlled by `%part` and the selected target profile.

## Syntax

### PTO Assembly Form

```text
vpack %dst, %src0, %src1, %part
```

### AS Level 1 (SSA)

```mlir
%result = pto.vpack %src0, %src1, %part : !pto.vreg<NxT_wide>, !pto.vreg<NxT_wide>, index -> !pto.vreg<2NxT_narrow>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %src0 | `!pto.vreg<NxT_wide>` | First wide source vector |
| %src1 | `!pto.vreg<NxT_wide>` | Second wide source vector |
| %part | `index` | Packing mode or submode selector |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<2NxT_narrow>` | Packed narrow vector built from both sources |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- Packing is a narrowing conversion; values that do not fit the destination width follow the truncation or saturation behavior of the selected form.
- Lowering MUST preserve the destination ordering between the first and second source halves.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific packing, selector, or permutation mode should treat that dependency as target-profile-specific unless the manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++) {
    dst[i] = truncate(src0[i]);
    dst[N + i] = truncate(src1[i]);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vperm](./vperm.md)
- Next op in instruction set: [pto.vsunpack](./vsunpack.md)
