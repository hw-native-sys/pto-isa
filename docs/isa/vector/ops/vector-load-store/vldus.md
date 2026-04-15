# pto.vldus

`pto.vldus` is part of the [Vector Load Store](../../vector-load-store.md) instruction set.

## Summary

Unaligned load using primed align state.

## Mechanism

`pto.vldus` is part of the PTO vector memory/data-movement instruction set. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
%result, %align_out, %base_out = pto.vldus %source, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align, !pto.ptr<T, ub>
```

## Inputs

`%source` is the current UB address and `%align` is the incoming load
  alignment state primed by `pto.vldas` or a prior `pto.vldus`.

## Expected Outputs

`%result` is the assembled vector value, `%align_out` is the updated alignment
  state, and `%base_out` is the post-update base pointer state exposed in SSA
  form.

## Side Effects

This operation reads UB-visible storage and returns SSA results. It does not by itself allocate buffers, signal events, or establish a fence.

## Constraints

A matching `pto.vldas` MUST appear before the first dependent `pto.vldus`
  stream in the same vector loop. Both the alignment state and the base address
  advance across the stream, and the PTO ISA vector instructions representation exposes those updates as SSA results.

## Exceptions

- It is illegal to use addresses outside the required UB-visible space or to violate the alignment/distribution contract of the selected form.
- Masked-off lanes or inactive blocks do not make an otherwise-illegal address valid unless the operation text explicitly says so.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing sources for `pto.vldus` are `~/visa.txt` and `PTOAS/docs/vpto-spec.md` on the latest fetched `feature_vpto_backend` branch.
It does **not** publish a standalone numeric latency for `pto.vldus`, but it does publish the throughput contract for the `pto.vldas`-primed unaligned load stream that `pto.vldus` participates in.

| Metric | Value | Source Basis |
|--------|-------|--------------|
| A5 standalone latency | Not publicly published | Current public VPTO timing material |
| Stream throughput with matching `pto.vldas` primer | One CPI for each subsequent unaligned load instruction | Public ISA timing note for the primed unaligned-load stream |

When documentation or scheduling depends on the throughput claim, treat it as a property of the **primed unaligned-load stream**, not as an isolated latency guarantee for `pto.vldus` alone.

## Examples

```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align2, %ub2 = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align, !pto.ptr<f32, ub>
```

## Detailed Notes

**Unaligned load pattern:**
```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align2, %ub2 = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align, !pto.ptr<f32, ub>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in instruction set: [pto.vldas](./vldas.md)
- Next op in instruction set: [pto.vldx2](./vldx2.md)
