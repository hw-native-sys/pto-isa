# pto.vgather2

`pto.vgather2` is part of the [Vector Load Store](../../vector-load-store.md) instruction set.

## Summary

Indexed gather from UB.

## Mechanism

`pto.vgather2` is part of the PTO vector memory/data-movement instruction set. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
%result = pto.vgather2 %source, %offsets, %active_lanes : !pto.ptr<T, ub>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>
```

## Inputs

`%source` is the UB base pointer, `%offsets` provides per-lane element
  offsets, and `%active_lanes` bounds how many lanes participate.

## Expected Outputs

`%result` is the gathered vector.

## Side Effects

This operation reads UB-visible storage and returns SSA results. It does not by itself allocate buffers, signal events, or establish a fence.

## Constraints

Only the first `%active_lanes` indices participate. The index element width
  and interpretation MUST match the selected gather form, and each effective
  address must satisfy that form's alignment rules.

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
For `pto.vgather2`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vgather2`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i] * sizeof(T)];
```

## Detailed Notes

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i] * sizeof(T)];
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in instruction set: [pto.vsldb](./vsldb.md)
- Next op in instruction set: [pto.vgatherb](./vgatherb.md)
