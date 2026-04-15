# pto.vldas

`pto.vldas` is part of the [Vector Load Store](../../vector-load-store.md) instruction set.

## Summary

Prime alignment buffer for subsequent unaligned load.

## Mechanism

`pto.vldas` is part of the PTO vector memory/data-movement instruction set. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
%result = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align
```

## Inputs

`%source` is the UB address whose surrounding aligned block seeds the load
  alignment state.

## Expected Outputs

`%result` is the initialized load-alignment state.

## Side Effects

This operation reads UB-visible storage and returns SSA results. It does not by itself allocate buffers, signal events, or establish a fence.

## Constraints

This op is the required leading operation for a `pto.vldus` stream using the
  same alignment state. The source address itself need not be 32-byte aligned;
  hardware truncates it to the aligned block boundary for the priming load.

## Exceptions

- It is illegal to use addresses outside the required UB-visible space or to violate the alignment/distribution contract of the selected form.
- Masked-off lanes or inactive blocks do not make an otherwise-illegal address valid unless the operation text explicitly says so.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing sources for `pto.vldas` are `~/visa.txt` and `PTOAS/docs/vpto-spec.md` on the latest fetched `feature_vpto_backend` branch.
Those sources do **not** publish a standalone numeric latency for the priming op itself, but `visa.txt` is explicit about the stream-level throughput contract.

| Metric | Value | Source Basis |
|--------|-------|--------------|
| A5 priming-op latency | Not publicly published | Current public VPTO timing material |
| Subsequent unaligned-load throughput | One CPI for each subsequent unaligned load instruction in the same stream | Public ISA timing note for the primed unaligned-load stream |

`pto.vldas` should therefore be read as a setup instruction: the public timing contract is about the **following unaligned load stream**, not an isolated latency number for the setup op.

## Examples

```mlir
%result = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align
```

## Detailed Notes

The instruction set overview carries the remaining shared rules for this operation.

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in instruction set: [pto.vlds](./vlds.md)
- Next op in instruction set: [pto.vldus](./vldus.md)
