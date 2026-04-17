# pto.vintlv

`pto.vintlv` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Interleave two source vectors into an ordered low/high result pair.

## Mechanism

`pto.vintlv` merges the two source vectors lane-by-lane. The low result receives the first half of the interleaved stream and the high result receives the second half. Conceptually: `low = {lhs[0], rhs[0], lhs[1], rhs[1], ...}` and `high = {lhs[N/2], rhs[N/2], lhs[N/2+1], rhs[N/2+1], ...}`.

## Syntax

### PTO Assembly Form

```text
vintlv %low, %high, %lhs, %rhs
```

### AS Level 1 (SSA)

```mlir
%low, %high = pto.vintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %lhs | `!pto.vreg<NxT>` | First source vector |
| %rhs | `!pto.vreg<NxT>` | Second source vector |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %low | `!pto.vreg<NxT>` | First half of the interleaved stream |
| %high | `!pto.vreg<NxT>` | Second half of the interleaved stream |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- `%lhs`, `%rhs`, `%low`, and `%high` MUST have the same element type and vector width.
- The result pair ordering is architectural and MUST be preserved by lowering.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific packing, selector, or permutation mode should treat that dependency as target-profile-specific unless the manual states cross-target portability explicitly.

## Examples

```c
// low  = {lhs[0], rhs[0], lhs[1], rhs[1], ...}
// high = {lhs[N/2], rhs[N/2], lhs[N/2+1], rhs[N/2+1], ...}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: (none)
- Next op in instruction set: [pto.vdintlv](./vdintlv.md)
