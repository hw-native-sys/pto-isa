# pto.vsld

`pto.vsld` is part of the [Vector Load Store](../../vector-load-store.md) instruction set.

## Summary

Strided load with fixed stride pattern.

## Mechanism

`pto.vsld` is part of the PTO vector memory/data-movement instruction set. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

### PTO Assembly Form

```text
vsld %result, %source[%offset], "STRIDE"
```

### AS Level 1 (SSA)

```mlir
%result = pto.vsld %source[%offset], "STRIDE" : !pto.ptr<T, ub> -> !pto.vreg<NxT>
```

## Inputs

`%source` is the UB base pointer and `%offset` is the displacement encoded
  with the selected fixed stride mode.

## Expected Outputs

`%result` is the loaded vector.

## Side Effects

This operation reads UB-visible storage and returns SSA results. It does not by itself allocate buffers, signal events, or establish a fence.

## Constraints

The selected stride token determines which sub-elements are read from each source block.

## Exceptions

- It is illegal to use addresses outside the required UB-visible space or to violate the alignment/distribution contract of the selected form.
- Masked-off lanes or inactive blocks do not make an otherwise-illegal address valid unless the operation text explicitly says so.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing material for PTO micro instructions remains limited.
For `pto.vsld`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vsld`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```mlir
%result = pto.vsld %source[%offset], "STRIDE" : !pto.ptr<T, ub> -> !pto.vreg<NxT>
```

## Detailed Notes

**Stride modes:** `STRIDE_S3_B16`, `STRIDE_S4_B64`, `STRIDE_S8_B32`, `STRIDE_S2_B64`

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in instruction set: [pto.vldx2](./vldx2.md)
- Next op in instruction set: [pto.vsldb](./vsldb.md)
