# pto.vperm

`pto.vperm` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

In-register permutation by per-lane index lookup.

## Mechanism

`pto.vperm` performs a register-local lookup: each lane of `%index` chooses which lane of `%src` is copied into the corresponding result lane. Unlike `pto.vgather2`, the data source is another vector register, not UB memory.

## Syntax

### PTO Assembly Form

```text
vperm %dst, %src, %index
```

### AS Level 1 (SSA)

```mlir
%result = pto.vperm %src, %index : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %src | `!pto.vreg<NxT>` | Source vector to permute |
| %index | `!pto.vreg<NxI>` | Per-lane source-index selector |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Permuted vector |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- `%index` values outside the supported range follow the wrap or clamp behavior of the selected form.
- `%src` and `%result` MUST have the same element type and vector width.
- This is an in-register permutation and does not access UB memory.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific packing, selector, or permutation mode should treat that dependency as target-profile-specific unless the manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = src[index[i] % N];
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vusqz](./vusqz.md)
- Next op in instruction set: [pto.vpack](./vpack.md)
