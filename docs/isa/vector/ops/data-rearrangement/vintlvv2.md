# pto.vintlvv2

`pto.vintlvv2` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Variant interleave form that returns one selected half of the interleaved result.

## Mechanism

`pto.vintlvv2` preserves the same interleave semantics as `pto.vintlv`, but returns only one half of the result pair. The `PART` selector chooses which half of the interleaved stream is materialized in SSA form.

## Syntax

### PTO Assembly Form

```text
vintlvv2 %dst, %lhs, %rhs, "PART"
```

### AS Level 1 (SSA)

```mlir
%result = pto.vintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %lhs | `!pto.vreg<NxT>` | First source vector |
| %rhs | `!pto.vreg<NxT>` | Second source vector |
| `PART` | enum | Selector for which interleave half is returned |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Selected half of the interleave result |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- The `PART` selector determines which half of the paired interleave result is returned.
- `%lhs`, `%rhs`, and `%result` MUST have the same element type and vector width.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific packing, selector, or permutation mode should treat that dependency as target-profile-specific unless the manual states cross-target portability explicitly.

## Examples

```mlir
%result = pto.vintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vzunpack](./vzunpack.md)
- Next op in instruction set: [pto.vdintlvv2](./vdintlvv2.md)
