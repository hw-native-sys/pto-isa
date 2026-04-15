# pto.vsts

`pto.vsts` is part of the [Vector Load Store](../../vector-load-store.md) instruction set.

## Summary

Vector store with distribution mode.

## Mechanism

`pto.vsts` is part of the PTO vector memory/data-movement instruction set. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
pto.vsts %value, %dest[%offset], %mask {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.mask
```

## Inputs

`%value` is the source vector, `%dest` is the UB base pointer, `%offset` is
  the displacement, `%mask` selects the active lanes or sub-elements, and
  `DIST` selects the store distribution.

## Expected Outputs

This op has no SSA result; it writes to UB memory.

## Side Effects

This operation writes UB-visible memory and/or updates streamed alignment state. Stateful unaligned forms expose their evolving state in SSA form, but a trailing flush form may still be required to complete the stream.

## Constraints

The effective destination address MUST satisfy the alignment rule of the
  selected store mode. Narrowing/packing modes may only preserve a subset of the
  source bits. Merge-channel modes reinterpret the source vector as channel
  planes and interleave them on store.

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
For `pto.vsts`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vsts`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```mlir
pto.vsts %v, %ub[%offset], %mask {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
```

## Detailed Notes

**Distribution modes:**

| Mode | Description | C Semantics |
|------|-------------|-------------|
| `NORM_B8/B16/B32` | Contiguous store | `UB[base + i] = src[i]` |
| `PK_B16/B32` | Pack/narrowing store | `UB_i16[base + 2*i] = truncate_16(src_i32[i])` |
| `MRG4CHN_B8` | Merge 4 channels (R,G,B,A → RGBA) | Interleave 4 planes |
| `MRG2CHN_B8/B16` | Merge 2 channels | Interleave 2 planes |

**Example — Contiguous store:**
```mlir
pto.vsts %v, %ub[%offset], %mask {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in instruction set: [pto.vgather2_bc](./vgather2-bc.md)
- Next op in instruction set: [pto.vstx2](./vstx2.md)
