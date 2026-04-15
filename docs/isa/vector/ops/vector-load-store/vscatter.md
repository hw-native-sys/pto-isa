# pto.vscatter

`pto.vscatter` is part of the [Vector Load Store](../../vector-load-store.md) instruction set.

## Summary

Indexed scatter to UB.

## Mechanism

`pto.vscatter` is part of the PTO vector memory/data-movement instruction set. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
pto.vscatter %value, %dest, %offsets, %active_lanes : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vreg<NxI>, index
```

## Inputs

`%value` is the source vector, `%dest` is the UB base pointer, `%offsets`
  provides per-lane or per-block indices, and `%active_lanes` bounds the active
  requests.

## Expected Outputs

This op writes UB memory and returns no SSA value.

## Side Effects

This operation writes UB-visible memory and/or updates streamed alignment state. Stateful unaligned forms expose their evolving state in SSA form, but a trailing flush form may still be required to complete the stream.

## Constraints

Only `b8`, `b16`, and `b32` element sizes are supported. The index vector
  must use a supported integer element type and layout for this instruction set.
  Each computed address MUST be element-aligned. If two or more indices alias,
  only one write is guaranteed and the winning lane is implementation-defined.

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
For `pto.vscatter`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vscatter`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```c
for (int i = 0; i < active_lanes; i++)
    UB[base + offsets[i] * sizeof(T)] = src[i];
```

## Detailed Notes

```c
for (int i = 0; i < active_lanes; i++)
    UB[base + offsets[i] * sizeof(T)] = src[i];
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in instruction set: [pto.vsstb](./vsstb.md)
- Next op in instruction set: [pto.vsta](./vsta.md)
