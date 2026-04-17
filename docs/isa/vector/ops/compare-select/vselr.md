# pto.vselr

`pto.vselr` is part of the [Compare And Select](../../compare-select.md) instruction set.

## Summary

Per-lane select form with reversed predicate polarity.

## Mechanism

The visible result follows the reverse-select contract: when the associated predicate lane is true, the instruction selects `%src1`; otherwise it selects `%src0`. Some lowerings surface the controlling predicate implicitly rather than as an explicit SSA operand, but the result polarity is the architectural difference from `pto.vsel`.

## Syntax

### PTO Assembly Form

```text
vselr %dst, %src0, %src1 : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vselr %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %src0 | `!pto.vreg<NxT>` | Default vector value used when the associated predicate lane is false |
| %src1 | `!pto.vreg<NxT>` | Vector value used when the associated predicate lane is true |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Selected vector result using the reverse-select polarity |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- `%src0`, `%src1`, and `%result` MUST have the same vector width `N` and element type `T`.
- If the controlling predicate is implicit in the lowering, that implicit source MUST be documented by the surrounding IR pattern or target profile.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an implicit predicate source or a target-specific encoding variant should treat that dependency as target-profile-specific unless the manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    result[i] = pred[i] ? src1[i] : src0[i];
```

```mlir
%result = pto.vselr %fallback, %preferred : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Compare And Select](../../compare-select.md)
- Previous op in instruction set: [pto.vsel](./vsel.md)
- Next op in instruction set: [pto.vselrv2](./vselrv2.md)
