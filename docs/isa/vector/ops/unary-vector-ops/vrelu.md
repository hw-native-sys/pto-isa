# pto.vrelu

`pto.vrelu` is part of the [Unary Vector Instructions](../../unary-vector-ops.md) instruction set.

## Summary

`%result` holds `max(input[i], 0)` per active lane.

## Mechanism

`pto.vrelu` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the instruction set operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vrelu %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

Documented A5 types or forms: `f16, f32`.

## Inputs

`%input` is the source vector and `%mask` selects active lanes.

## Expected Outputs

`%result` holds `max(input[i], 0)` per active lane.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Only floating-point element types are legal
  on the current A5 instruction set described here.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `f16, f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### A5 Latency

| Element Type | Latency (cycles) | A5 RV |
|---|---|---|
| `f32` | 5 | `RV_VRELU` |
| `f16` | 5 | `RV_VRELU` |

### A2/A3 Throughput

| Metric | Value | Constant |
|--------|-------|----------|
| Startup latency | 14 | `A2A3_STARTUP_BINARY` |
| Completion latency | 19 | `A2A3_COMPL_FP_BINOP` |
| Per-repeat throughput | 1 | `A2A3_RPT_1` |
| Pipeline interval | 18 | `A2A3_INTERVAL` |

**Performance note:** `vrelu` has the **lowest latency** of all unary vector ops on A5 (5 cycles). For leaky ReLU, use `vlrelu` which adds one scalar multiply.

---

## Examples

### C Semantics

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > 0) ? src[i] : 0;
```

`vrelu` is the **lowest-latency unary operation** on A5 (5 cycles). For Leaky ReLU with a learned negative slope α, use `vlrelu` instead (adds one scalar multiply).

---

## Related Ops / Instruction Set Links

- Instruction set overview: [Unary Vector Instructions](../../unary-vector-ops.md)
- Previous op in instruction set: [pto.vrec](./vrec.md)
- Next op in instruction set: [pto.vnot](./vnot.md)
