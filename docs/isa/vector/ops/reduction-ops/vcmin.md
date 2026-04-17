# pto.vcmin

`pto.vcmin` is part of the [Reduction Instructions](../../reduction-ops.md) instruction set.

## Summary

Full-vector minimum reduction with argmin information packed into the low result lanes.

## Mechanism

The instruction scans all active lanes and finds the minimum value. The low result lanes carry the minimum and its lane index using the form defined by the selected target profile; the remaining result lanes are zero-filled.

## Syntax

### PTO Assembly Form

```text
vcmin %dst, %src, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vcmin %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %input | `!pto.vreg<NxT>` | Source vector register to reduce |
| %mask | `!pto.mask` | Predicate mask; inactive lanes do not participate |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Low result lanes carry the minimum value and its lane index; other lanes are zero-filled |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- The exact value/index packing MUST follow the selected target profile.
- If all predicate bits are zero, the result follows the instruction family zero-fill convention.
- The mask width MUST match `N`.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `i16-i32`, `f16`, `f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.

## Examples

```c
T mn = INF;
int idx = 0;
for (int i = 0; i < N; i++)
    if (mask[i] && src[i] < mn) { mn = src[i]; idx = i; }
result_value = mn;
result_index = idx;
```

```mlir
%result = pto.vcmin %input, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Reduction Instructions](../../reduction-ops.md)
- Previous op in instruction set: [pto.vcmax](./vcmax.md)
- Next op in instruction set: [pto.vcgmax](./vcgmax.md)
