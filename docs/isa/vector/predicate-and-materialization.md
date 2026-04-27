# Vector Predicate And Materialization

This page is a redirect. Predicate and materialization instructions for the Vector ISA are documented in the canonical Scalar/Control ISA page:

- [Predicate Load/Store (scalar)](../scalar/predicate-load-store.md) — `pto.plds`, `pto.pld`, `pto.pldi`, `pto.psts`, `pto.pst`, `pto.psti`, `pto.pstu`

## Why This Page Exists

Vector instructions can consume predicate masks produced by scalar load/store operations. The Vector ISA reuses the same predicate-register infrastructure defined by the Scalar/Control ISA. Rather than duplicating the full specification, this page clarifies the Vector ISA's relationship to the scalar predicate model.

## Vector ISA Predicate Consumption

Vector instructions consume `!pto.mask` operands for conditional lane execution:

```mlir
%vdst = pto.vadd %vsrc0, %vsrc1, %mask
    : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask) -> !pto.vreg<64xf32>
```

The mask is produced by the scalar predicate load/store operations documented in [Predicate Load/Store](../scalar/predicate-load-store.md). See the scalar page for full semantics including distribution modes, scalar load semantics, and constraint tables.

## Vector-Specific Predicate Ops

The Vector ISA adds two materialization operations that generate masks from vector data:

- [pto.vbr](./ops/predicate-and-materialization/vbr.md) — Vector broadcast: replicate a scalar value into all lanes of a vector register
- [pto.vdup](./ops/predicate-and-materialization/vdup.md) — Vector duplicate: copy a scalar value into a vector register

## See Also

- [Predicate Load/Store (canonical)](../scalar/predicate-load-store.md) — Full specification of `plds`, `pld`, `pldi`, `psts`, `pst`, `psti`, `pstu`
- [Vector ISA overview](../instruction-families/vector-families.md) — Instruction set contracts
