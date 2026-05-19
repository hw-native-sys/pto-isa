# pto.mte_ub_l1 (cube view)

`pto.mte_ub_l1` is the reverse direction of [`pto.mte_l1_ub`](./mte-l1-ub.md). The canonical reference page for this instruction lives under the scalar DMA Copy section because the AIV-side scalar program issues the transfer:

- **Reference page**: [pto.mte_ub_l1 — Scalar DMA Copy](../../../scalar/ops/dma-copy/mte-ub-l1.md)

This stub exists so the cube section index resolves the link cleanly. Refer to the scalar DMA Copy reference page for the full syntax, parameter table, and constraints.

## Why it lives under scalar DMA

The UB→L1 transfer is launched from the Vector (AIV) scalar program because the source pointer is in UB — the buffer owned by the Vector block. The cube (AIC) consumes the resulting L1 tile through [`pto.mte_l1_l0a`](./mte-l1-l0a.md) / [`pto.mte_l1_l0b`](./mte-l1-l0b.md) once the AIC has been notified that L1 is ready.

## Related Ops

- Reverse direction (L1 → UB): [pto.mte_l1_ub](./mte-l1-ub.md)
- Cube operand staging: [pto.mte_l1_l0a](./mte-l1-l0a.md), [pto.mte_l1_l0b](./mte-l1-l0b.md)
- Inter-block sync: [Cluster Programming Model](../../../machine-model/execution-agents.md)
