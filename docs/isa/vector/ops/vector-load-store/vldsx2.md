# pto.vldsx2

`pto.vldsx2` is part of the [Vector Load Store](../../vector-load-store.md) instruction set.

## Summary

Dual load with deinterleave (AoS → SoA conversion).

## Mechanism

`pto.vldsx2` is part of the PTO vector memory/data-movement instruction set. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

### PTO Assembly Form

```text
vldsx2 %low, %high, %source[%offset], "DIST"
```

### AS Level 1 (SSA)

```mlir
%low, %high = pto.vldsx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

## Inputs

`%source` is the UB base pointer, `%offset` is the displacement, and `DIST`
  selects a dual-load/deinterleave layout.

## Expected Outputs

`%low` and `%high` are the two destination vectors.

## Side Effects

This operation reads UB-visible storage and returns SSA results. It does not by itself allocate buffers, signal events, or establish a fence.

## Constraints

!!! warning "Constraints"
    This instruction set is only legal for interleave/deinterleave style distributions.
      The two outputs form an ordered pair, and that pairing MUST be preserved.

## Exceptions

!!! danger "Exceptions"
    - It is illegal to use addresses outside the required UB-visible space or to violate the alignment/distribution contract of the selected form.
    - Masked-off lanes or inactive blocks do not make an otherwise-illegal address valid unless the operation text explicitly says so.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

PTO-Gym v0.6 SPEC publishes a uniform 9-cycle latency for all `pto.vldsx2` distribution families on the A5 profile.

| Metric | Value | Source Basis |
|--------|-------|--------------|
| A5 latency (`BDINTLV`, `DINTLV_B8`, `DINTLV_B16`, `DINTLV_B32`) | **9** cycles | PTO-Gym v0.6 SPEC, §III Vector Load/Store |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

Other target profiles (CPU simulation, A2/A3) treat the cost as target-defined; measure on the concrete backend rather than reusing the A5 number.

## Examples

```c
// DINTLV_B32: deinterleave 32-bit elements
for (int i = 0; i < 64; i++) {
    low[i]  = UB[base + 8*i];       // even elements
    high[i] = UB[base + 8*i + 4];   // odd elements
}
```

```mlir
%x, %y = pto.vldsx2 %ub[%offset], "DINTLV_B32" : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
```

## Detailed Notes

**Distribution modes:** `DINTLV_B8`, `DINTLV_B16`, `DINTLV_B32`, `BDINTLV`

```c
// DINTLV_B32: deinterleave 32-bit elements
for (int i = 0; i < 64; i++) {
    low[i]  = UB[base + 8*i];       // even elements
    high[i] = UB[base + 8*i + 4];   // odd elements
}
```

**Example — Load interleaved XY pairs into separate X/Y vectors:**
```mlir
%x, %y = pto.vldsx2 %ub[%offset], "DINTLV_B32" : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector Load Store](../../vector-load-store.md)
- Previous op in instruction set: [pto.vldus](./vldus.md)
- Next op in instruction set: [pto.vsld](./vsld.md)
