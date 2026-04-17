# pto.vsel

`pto.vsel` is part of the [Compare And Select](../../compare-select.md) instruction set.

## Summary

Per-lane select between two vector operands using an explicit predicate mask.

## Mechanism

For each lane `i`, `result[i] = mask[i] ? src0[i] : src1[i]`. The operation consumes two same-shaped vector operands and chooses one value per lane according to the explicit predicate mask.

## Syntax

### PTO Assembly Form

```text
vsel %dst, %src_true, %src_false, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vsel %src0, %src1, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %src0 | `!pto.vreg<NxT>` | Value selected when the mask bit is 1 |
| %src1 | `!pto.vreg<NxT>` | Value selected when the mask bit is 0 |
| %mask | `!pto.mask` | Predicate mask that chooses between `%src0` and `%src1` per lane |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Selected vector result |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- `%src0`, `%src1`, and `%result` MUST have the same vector width `N` and element type `T`.
- The mask width MUST match `N`.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an implicit predicate source or a target-specific encoding variant should treat that dependency as target-profile-specific unless the manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    result[i] = mask[i] ? src0[i] : src1[i];
```

```mlir
%result = pto.vsel %true_vals, %false_vals, %condition : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Compare And Select](../../compare-select.md)
- Previous op in instruction set: [pto.vcmps](./vcmps.md)
- Next op in instruction set: [pto.vselr](./vselr.md)
