# pto.vcpadd

`pto.vcpadd` is part of the [Reduction Instructions](../../reduction-ops.md) instruction set.

## Summary

Inclusive prefix-sum (scan) over the active lanes.

## Mechanism

The instruction computes an inclusive prefix sum across the active lanes: lane `i` receives the sum of all active-source values from lane `0` through lane `i`. Inactive lanes contribute zero and keep the prefix state unchanged for later active lanes.

## Syntax

### PTO Assembly Form

```text
vcpadd %dst, %src, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vcpadd %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %input | `!pto.vreg<NxT>` | Source vector register to scan |
| %mask | `!pto.mask` | Predicate mask; inactive lanes contribute zero |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| %result | `!pto.vreg<NxT>` | Inclusive prefix-sum vector |

## Side Effects

This operation has no architectural side effect beyond producing its destination values. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

- The current manual documents floating-point forms.
- The mask width MUST match `N`.
- Inactive lanes contribute zero to later active-lane prefixes.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `f16`, `f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.

## Examples

```c
T running = 0;
for (int i = 0; i < N; i++) {
    if (mask[i]) running += src[i];
    dst[i] = running;
}
```

```mlir
%cdf = pto.vcpadd %pdf, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Reduction Instructions](../../reduction-ops.md)
- Previous op in instruction set: [pto.vcgmin](./vcgmin.md)
- Next op in instruction set: (none)
