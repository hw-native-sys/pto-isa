# pto.vslide

`pto.vslide` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Concatenate two vectors and extract N-element window at offset.

## Mechanism

`pto.vslide` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the instruction set operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vslide %src0, %src1, %amt : !pto.vreg<NxT>, !pto.vreg<NxT>, i16 -> !pto.vreg<NxT>
```

## Inputs

`%src0` and `%src1` provide the concatenated source window and
  `%amt` selects the extraction offset.

## Expected Outputs

`%result` is the extracted destination window.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

`pto.vslide` operates on the logical
  concatenation of `%src1` and `%src0`. The source order and extraction offset
  MUST be preserved exactly.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The timing sources currently used for PTO micro-instruction pages are `~/visa.txt` and `PTOAS/docs/vpto-spec.md` on the latest fetched `feature_vpto_backend` branch.
For `pto.vslide`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | `visa.txt`, `PTOAS/docs/vpto-spec.md` |
| Steady-state throughput | Not publicly published | `visa.txt`, `PTOAS/docs/vpto-spec.md` |

If software scheduling or performance modeling depends on the exact cost of `pto.vslide`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```c
// Conceptually: tmp[0..2N-1] = {src1, src0}
// dst[i] = tmp[amt + i]
if (amt >= 0)
    for (int i = 0; i < N; i++)
        dst[i] = (i >= amt) ? src0[i - amt] : src1[N - amt + i];
```

## Detailed Notes

```c
// Conceptually: tmp[0..2N-1] = {src1, src0}
// dst[i] = tmp[amt + i]
if (amt >= 0)
    for (int i = 0; i < N; i++)
        dst[i] = (i >= amt) ? src0[i - amt] : src1[N - amt + i];
```

**Use case:** Sliding window operations, shift register patterns.

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vdintlv](./vdintlv.md)
- Next op in instruction set: [pto.vshift](./vshift.md)
