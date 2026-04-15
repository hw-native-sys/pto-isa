# pto.vcmps

`pto.vcmps` is part of the [Compare And Select](../../compare-select.md) instruction set.

## Summary

Compare vector against scalar.

## Mechanism

`pto.vcmps` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the instruction set operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vcmps %src, %scalar, %seed, "CMP_MODE" : !pto.vreg<NxT>, T, !pto.mask -> !pto.mask
```

## Inputs

`%src` is the vector source, `%scalar` is the scalar comparison
  value, and `%seed` is the incoming predicate.

## Expected Outputs

`%result` is the generated predicate mask.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

For 32-bit scalar forms, the scalar source
  MUST satisfy the backend's legal scalar-source constraints for this instruction set.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing material for PTO micro instructions remains limited.
For `pto.vcmps`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vcmps`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src[i] CMP scalar) ? 1 : 0;
```

```mlir
%positive_mask = pto.vcmps %values, %c0_f32, %all_active, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
// positive_mask[i] = 1 if values[i] > 0
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src[i] CMP scalar) ? 1 : 0;
```

**Example:**
```mlir
%positive_mask = pto.vcmps %values, %c0_f32, %all_active, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
// positive_mask[i] = 1 if values[i] > 0
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Compare And Select](../../compare-select.md)
- Previous op in instruction set: [pto.vcmp](./vcmp.md)
- Next op in instruction set: [pto.vsel](./vsel.md)
