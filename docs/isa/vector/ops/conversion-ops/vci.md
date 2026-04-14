# pto.vci

`pto.vci` is part of the [Conversion Ops](../../conversion-ops.md) instruction set.

## Summary

Generate a vector of lane indices from a scalar seed.

## Mechanism

`pto.vci` materializes a vector whose elements are derived from the scalar seed `%index` and the selected ordering mode. In the common ascending form, lane `i` receives `index + i`; in the descending form, lane `i` receives `index - i`.

## Syntax

### PTO Assembly Form

```mlir
%indices = pto.vci %index {order = "ASC"} : i32 -> !pto.vreg<64xi32>
```

### AS Level 1 (SSA)

```mlir
%indices = pto.vci %index {order = "ASC"} : i32 -> !pto.vreg<64xi32>
```

### AS Level 2 (DPS)

```mlir
pto.vci ins(%index : i32) outs(%indices : !pto.vreg<64xi32>) {order = "ASC"}
```

## C++ Intrinsic

The installed Bisheng public intrinsic uses `__cce_simd::INC_ORDER` / `__cce_simd::DEC_ORDER` tokens rather than the PTO string attribute.

```cpp
vector_s32 dst;
int32_t index = 0;
vci(dst, index, __cce_simd::INC_ORDER);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%index` | `i32` | Scalar seed used to generate the first lane value |

**Attributes:**

| Attribute | Values | Description |
|-----------|--------|-------------|
| `order` | `"ASC"`, `"DESC"` | Selects increasing or decreasing lane numbering |

## Expected Outputs

| Operand | Type | Description |
|---------|------|-------------|
| `%indices` | `!pto.vreg<64xi32>` | Generated index vector |

## Side Effects

This operation has no architectural side effect beyond producing its SSA result. It does not reserve buffers, signal events, or establish fences.

## Constraints

- `%index` MUST have a type compatible with the selected result element type.
- `order` MUST be one of the documented values.
- The visible PTO contract on this page is index generation; target-specific lowering may use helper forms with different concrete element types.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and invalid `order` values.
- Any additional illegality stated in the [Conversion Ops](../../conversion-ops.md) page is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual.
- Under the current documented A5 contract, `pto.vci` does not map to a sampled vector `RV_*` compute event in `veccore0` trace.
- CPU simulation and A2/A3-class targets may realize the same visible behavior with software or helper-lowered implementations.

## Performance

### Execution Model

`pto.vci` is an index-materialization helper executed inside `pto.vecscope`. It is typically dominated by setup and lane generation rather than a standard vector ALU pipeline.

### A2/A3 Throughput

`vci` does not have a direct binary/unary cost-table entry in the current A2/A3 summary. Treat it as target-profile-specific helper work rather than a standard arithmetic RV opcode.

## Examples

### Generate ascending indices

```mlir
%indices = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
```

### Generate descending indices

```mlir
%indices = pto.vci %c63 {order = "DESC"} : i32 -> !pto.vreg<64xi32>
```

### Prepare gather indices

```mlir
%idx = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
%data = pto.vgather2 %ub_table[%c0], %idx {dist = "DIST"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

## Detailed Notes

`pto.vci` is most commonly used to initialize sequential index vectors for gather/scatter pipelines, ranking kernels, and sort-key preparation. The installed public C++ intrinsic also exposes concrete overloads beyond the common `i32` PTO example, but the page-level PTO contract remains lane-index generation from a scalar seed.

## Related Ops / Instruction Set Links

- Instruction set overview: [Conversion Ops](../../conversion-ops.md)
- Next op in instruction set: [pto.vcvt](./vcvt.md)
- Vector instruction overview: [Vector Instructions](../../../instruction-surfaces/vector-instructions.md)
