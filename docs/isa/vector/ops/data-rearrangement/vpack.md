# pto.vpack

`pto.vpack` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Narrowing pack — two wide vectors to one narrow vector.

## Mechanism

`pto.vpack` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the instruction set operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vpack %src0, %src1, %part : !pto.vreg<NxT_wide>, !pto.vreg<NxT_wide>, index -> !pto.vreg<2NxT_narrow>
```

## Inputs

`%src0` and `%src1` are wide source vectors and `%part` selects
  the packing submode.

## Expected Outputs

`%result` is the packed narrow vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Packing is a narrowing conversion. Source
  values that do not fit the destination width follow the truncation semantics
  of the selected packing mode.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The timing sources currently used for PTO micro-instruction pages are `~/visa.txt` and `PTOAS/docs/vpto-spec.md` on the latest fetched `feature_vpto_backend` branch.
For `pto.vpack`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | `visa.txt`, `PTOAS/docs/vpto-spec.md` |
| Steady-state throughput | Not publicly published | `visa.txt`, `PTOAS/docs/vpto-spec.md` |

If software scheduling or performance modeling depends on the exact cost of `pto.vpack`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```c
// e.g., two vreg<64xi32> → one vreg<128xi16>
for (int i = 0; i < N; i++) {
    dst[i]     = truncate(src0[i]);
    dst[N + i] = truncate(src1[i]);
}
```

## Detailed Notes

```c
// e.g., two vreg<64xi32> → one vreg<128xi16>
for (int i = 0; i < N; i++) {
    dst[i]     = truncate(src0[i]);
    dst[N + i] = truncate(src1[i]);
}
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vperm](./vperm.md)
- Next op in instruction set: [pto.vsunpack](./vsunpack.md)
