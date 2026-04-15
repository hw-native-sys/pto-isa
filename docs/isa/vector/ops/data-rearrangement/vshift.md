# pto.vshift

`pto.vshift` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Single-source slide (shift with zero fill).

## Mechanism

`pto.vshift` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the instruction set operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vshift %src, %amt : !pto.vreg<NxT>, i16 -> !pto.vreg<NxT>
```

## Inputs

`%src` is the source vector and `%amt` is the slide amount.

## Expected Outputs

`%result` is the shifted vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

This instruction set represents the single-source
  slide/shift instruction set. Zero-fill versus other fill behavior MUST match the
  selected form.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The timing sources currently used for PTO micro-instruction pages are `~/visa.txt` and `PTOAS/docs/vpto-spec.md` on the latest fetched `feature_vpto_backend` branch.
For `pto.vshift`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | `visa.txt`, `PTOAS/docs/vpto-spec.md` |
| Steady-state throughput | Not publicly published | `visa.txt`, `PTOAS/docs/vpto-spec.md` |

If software scheduling or performance modeling depends on the exact cost of `pto.vshift`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = (i >= amt) ? src[i - amt] : 0;
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = (i >= amt) ? src[i - amt] : 0;
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vslide](./vslide.md)
- Next op in instruction set: [pto.vsqz](./vsqz.md)
