# pto.vneg

`pto.vneg` is part of the [Unary Vector Instructions](../../unary-vector-ops.md) instruction set.

## Summary

`%result` is the lane-wise arithmetic negation.

## Mechanism

`pto.vneg` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the instruction set operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vneg %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

Documented A5 types or forms: `i8-i32, f16, f32`.

## Inputs

`%input` is the source vector and `%mask` selects active lanes.

## Expected Outputs

`%result` is the lane-wise arithmetic negation.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Source and result types MUST match.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `i8-i32, f16, f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### A5 Latency

| Element Type | Latency (cycles) | A5 RV |
|---|---|---|
| `f32` | 8 | `RV_VMULS` (scalar-mul path) |
| `f16` | 8 | `RV_VMULS` (scalar-mul path) |
| `i32` | 8 | `RV_VMULS` (scalar-mul path) |
| `i16` | 8 | `RV_VMULS` (scalar-mul path) |

### A2/A3 Throughput

| Metric | Value | Constant |
|--------|-------|----------|
| Startup latency | 14 | `A2A3_STARTUP_BINARY` |
| Completion latency | 20 (FP) / 18 (INT) | `A2A3_COMPL_FP_MUL` / `A2A3_COMPL_INT_MUL` |
| Per-repeat throughput | 1 | `A2A3_RPT_1` |
| Pipeline interval | 18 | `A2A3_INTERVAL` |

---

## Examples

### C Semantics

```c
for (int i = 0; i < N; i++)
    dst[i] = -src[i];
```

`vneg` is implemented via the scalar-multiply (`RV_VMULS`) hardware path with a -1 scalar, giving it the same latency as scalar multiplication.

---

## Related Ops / Instruction Set Links

- Instruction set overview: [Unary Vector Instructions](../../unary-vector-ops.md)
- Previous op in instruction set: [pto.vabs](./vabs.md)
- Next op in instruction set: [pto.vexp](./vexp.md)
