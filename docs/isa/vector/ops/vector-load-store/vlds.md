# pto.vlds

`pto.vlds` is part of the [Vector Load Store](../../vector-load-store.md) instruction set.

## Summary

Vector load with distribution mode.

## Mechanism

`pto.vlds` is part of the PTO vector memory/data-movement instruction set. It keeps UB addressing, distribution, mask behavior, and any alignment-state threading explicit in SSA form rather than hiding those details in backend-specific lowering.

## Syntax

```mlir
%result = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>
```

## Inputs

`%source` is the UB base address, `%offset` is the load displacement, and
  `DIST` selects the distribution mode.

## Expected Outputs

`%result` is the loaded vector register value.

## Side Effects

This operation reads UB-visible storage and returns SSA results. It does not by itself allocate buffers, signal events, or establish a fence.

## Constraints

The effective address MUST satisfy the alignment rule of the selected
  distribution mode. `NORM` reads one full vector footprint. Broadcast,
  upsample, downsample, unpack, split-channel, and deinterleave modes change
  how memory bytes are mapped into destination lanes, but they do not change the
  fact that the source is UB memory.

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
For `pto.vlds`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vlds`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```mlir
%v = pto.vlds %ub[%offset] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

```mlir
%v = pto.vlds %ub[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

## Detailed Notes

**Distribution modes:**

| Mode | Description | C Semantics |
|------|-------------|-------------|
| `NORM` | Contiguous 256B load | `dst[i] = UB[base + i * sizeof(T)]` |
| `BRC_B8/B16/B32` | Broadcast single element | `dst[i] = UB[base]` for all i |
| `US_B8/B16` | Upsample (duplicate each element) | `dst[2*i] = dst[2*i+1] = UB[base + i]` |
| `DS_B8/B16` | Downsample (every 2nd element) | `dst[i] = UB[base + 2*i]` |
| `UNPK_B8/B16/B32` | Unpack (zero-extend to wider type) | `dst_i32[i] = (uint32_t)UB_i16[base + 2*i]` |
| `SPLT4CHN_B8` | Split 4-channel (RGBA → R plane) | Extract every 4th byte |
| `SPLT2CHN_B8/B16` | Split 2-channel | Extract every 2nd element |
| `DINTLV_B32` | Deinterleave 32-bit | Even elements only |
| `BLK` | Block load | Blocked access pattern |

**Example — Contiguous load:**
```mlir
%v = pto.vlds %ub[%offset] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

**Example — Broadcast scalar to all lanes:**
```mlir
%v = pto.vlds %ub[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Vector Load Store](../../vector-load-store.md)
- Next op in instruction set: [pto.vldas](./vldas.md)
