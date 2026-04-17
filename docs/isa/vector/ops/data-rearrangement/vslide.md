# pto.vslide

`pto.vslide` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Concatenate two vectors and extract an N-element window at a scalar offset.

## Mechanism

`pto.vslide` forms a logical `2N`-element window by concatenating `%src1` followed by `%src0`, then extracts `N` elements starting at `%amt`. This makes the operation useful for shift-register and sliding-window patterns without touching UB memory.

## Syntax

### PTO Assembly Form

```text
vslide %dst, %src0, %src1, %amt
```

### AS Level 1 (SSA)

```mlir
%result = pto.vslide %src0, %src1, %amt : !pto.vreg<NxT>, !pto.vreg<NxT>, i16 -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %src0 | `!pto.vreg<NxT>` | Right-hand half of the logical concatenation |
| %src1 | `!pto.vreg<NxT>` | Left-hand half of the logical concatenation |
| %amt | `i16` | Window start offset in the concatenated stream |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Extracted `N`-element window |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- `%src0`, `%src1`, and `%result` MUST have the same element type and vector width.
- The slide amount MUST satisfy the range supported by the selected target profile.
- Lowering MUST preserve the `%src1 || %src0` concatenation order.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific packing, selector, or permutation mode should treat that dependency as target-profile-specific unless the manual states cross-target portability explicitly.

## Examples

```c
// tmp[0..2N-1] = {src1, src0}
// dst[i] = tmp[amt + i]
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vdintlv](./vdintlv.md)
- Next op in instruction set: [pto.vshift](./vshift.md)
