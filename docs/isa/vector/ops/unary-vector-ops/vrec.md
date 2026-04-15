# pto.vrec

`pto.vrec` is part of the [Unary Vector Instructions](../../unary-vector-ops.md) instruction set.

## Summary

`%result` holds the reciprocal per active lane.

## Mechanism

`pto.vrec` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the instruction set operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vrec %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

Documented A5 types or forms: `f16, f32`.

## Inputs

`%input` is the source vector and `%mask` selects active lanes.

## Expected Outputs

`%result` holds the reciprocal per active lane.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Only floating-point element types are legal.
  Active inputs containing `+0` or `-0` follow the target's divide-style
  exceptional behavior.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Target-defined numeric exceptional behavior, such as divide-by-zero or out-of-domain inputs, remains subject to the selected backend profile unless this page narrows it further.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `f16, f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing material for PTO micro instructions remains limited.
For `pto.vrec`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vrec`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = 1.0f / src[i];
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = 1.0f / src[i];
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Unary Vector Instructions](../../unary-vector-ops.md)
- Previous op in instruction set: [pto.vrsqrt](./vrsqrt.md)
- Next op in instruction set: [pto.vrelu](./vrelu.md)
