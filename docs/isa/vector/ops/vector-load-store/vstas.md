# pto.vstas

`pto.vstas` is part of the [Vector Load Store](../../vector-load-store.md) instruction set.

## Summary

Scalar-register-offset form of alignment-state flush.

## Mechanism

`pto.vstas` is part of the PTO vector memory/data-movement instruction set. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32
```

## Inputs

`%value` is the pending store-alignment state, `%dest` is the UB base
  pointer, and `%offset` is the scalar-register style displacement.

## Expected Outputs

This op writes buffered tail bytes to UB and returns no SSA value.

## Side Effects

This operation writes UB-visible memory and/or updates streamed alignment state. Stateful unaligned forms expose their evolving state in SSA form, but a trailing flush form may still be required to complete the stream.

## Constraints

This instruction set uses the same buffered-tail semantics as `pto.vsta` but keeps the
  scalar-offset form explicit.

## Exceptions

- It is illegal to use addresses outside the required UB-visible space or to violate the alignment/distribution contract of the selected form.
- Masked-off lanes or inactive blocks do not make an otherwise-illegal address valid unless the operation text explicitly says so.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing sources for `pto.vstas` are `~/visa.txt` and `PTOAS/docs/vpto-spec.md` on the latest fetched `feature_vpto_backend` branch.
Those sources describe the buffered-tail flush semantics precisely, but they do **not** publish a numeric latency or steady-state throughput for `pto.vstas`.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | `visa.txt` §8.6 `VSTAS`, `PTOAS/docs/vpto-spec.md` |
| Steady-state throughput | Not publicly published | `visa.txt` §8.6 `VSTAS`, `PTOAS/docs/vpto-spec.md` |

Code that depends on the cost of the trailing flush step MUST measure the concrete backend rather than assuming a public cycle constant from the ISA text.

## Examples

```mlir
pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32
```

## Detailed Notes

The instruction set overview carries the remaining shared rules for this operation.

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in instruction set: [pto.vsta](./vsta.md)
- Next op in instruction set: [pto.vstar](./vstar.md)
