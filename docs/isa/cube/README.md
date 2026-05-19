# Cube Micro-Instruction Reference

This section documents the PTO **Cube micro-instruction surface**: the matrix-multiply (MAD) and cube-side data-movement ops that program the cube core (AIC) and its dedicated buffer hierarchy (L1 / L0A / L0B / L0C / BT).

!!! note "Scope and audience"
    Tile-level matrix ops such as `pto.tmatmul` (covered under [Tile ISA Matrix & Matrix-Vector](../tile/matrix-and-matrix-vector.md)) hide most of these primitives behind a tile-shaped interface. The cube micro-instructions documented here are the lower-level surface that compiler back-ends and hand-tuned cube kernels target directly. They make NZ fractal layout, L1/L0 buffer hierarchy, and FIXPIPE writeback explicit.

## Architectural Background

| Page | Purpose |
|------|---------|
| [NZ Fractal Layout](./nz-fractal-layout.md) | The fractal NZ format used by L1, L0A, L0B, and L0C. Defines the `(k1, m1, m0, k0)` re-indexing and per-buffer layout variants. |
| [Buffer Hierarchy](./buffer-hierarchy.md) | The L1 / L0A / L0B / L0C / BT memory hierarchy: address spaces, sizes, and data-flow contracts. |
| [FIXPIPE Model](./fixpipe-model.md) | The FIXPIPE writeback path: how L0C results are converted back to ND and routed to UB or GM. |

## Matrix Multiply (MAD) Ops

The MAD family computes `dst = lhs @ rhs` on tiles staged into the cube's L0A / L0B / L0C buffers. All variants share the same `(M, N, K)` shape parameters and a common set of optional clauses (`unit_flag`, `disable_gemv`, `sat`/`nosat`, `tf32_mode`, `n_dir`).

| Op | Semantics |
|----|-----------|
| [pto.mad](./ops/mad/mad.md) | Zero-init: `dst = lhs @ rhs` |
| [pto.mad_acc](./ops/mad/mad-acc.md) | Accumulate: `dst = dst + lhs @ rhs` |
| [pto.mad_bias](./ops/mad/mad-bias.md) | Bias-init: `dst = lhs @ rhs + bias[n]` |
| [pto.mad_mx](./ops/mad/mad-mx.md) | Zero-init MX (microscaled) matmul |
| [pto.mad_mx_acc](./ops/mad/mad-mx-acc.md) | Accumulating MX matmul |
| [pto.mad_mx_bias](./ops/mad/mad-mx-bias.md) | Bias-init MX matmul |

## Cube Data Movement Ops

These ops move tiles between GM, L1, L0A/L0B, and L0C using grouped `nburst(...)` / `loop(...)` clauses analogous to the [scalar DMA Copy](../scalar/dma-copy.md) surface.

### GM → L1

- [pto.mte_gm_l1](./ops/data-movement/mte-gm-l1.md) — Direct GM→L1 load (no layout transform)
- [pto.mte_gm_l1_frac](./ops/data-movement/mte-gm-l1-frac.md) — GM→L1 with ND→NZ fractal repack

### L1 ↔ UB

- [pto.mte_l1_ub](./ops/data-movement/mte-l1-ub.md) — L1→UB transfer (cube-to-vector data path)
- [pto.mte_ub_l1](../scalar/ops/dma-copy/mte-ub-l1.md) — UB→L1 transfer (vector-to-cube data path; lives in the scalar DMA section)

### L1 → L0A / L0B (cube operand load)

- [pto.mte_l1_l0a](./ops/data-movement/mte-l1-l0a.md) — Stage L1 NZ tile into L0A (left operand)
- [pto.mte_l1_l0b](./ops/data-movement/mte-l1-l0b.md) — Stage L1 NZ tile into L0B (right operand, K-innermost transpose)
- [pto.mte_l1_l0a_mx](./ops/data-movement/mte-l1-l0a-mx.md) — Load MX scale payload for L0A
- [pto.mte_l1_l0b_mx](./ops/data-movement/mte-l1-l0b-mx.md) — Load MX scale payload for L0B

### L1 → BT (bias)

- [pto.mte_l1_bt](./ops/data-movement/mte-l1-bt.md) — Stage bias vector into BT for `pto.mad_bias` / `pto.mad_mx_bias`
- [pto.mte_l1_fb](./ops/data-movement/mte-l1-fb.md) — Stage FIXPIPE-relevant payload (e.g., dequant params)

### L0C writeback (FIXPIPE)

- [pto.mte_l0c_l1](./ops/data-movement/mte-l0c-l1.md) — FIXPIPE: L0C → L1
- [pto.mte_l0c_gm](./ops/data-movement/mte-l0c-gm.md) — FIXPIPE: L0C → GM
- [pto.mte_l0c_ub](./ops/data-movement/mte-l0c-ub.md) — FIXPIPE: L0C → UB

## Full Cube Pipeline

```text
GM (ND)          L1/cbuf (NZ)            L0A/B (NZ)          L0C (NZ)    GM (ND)

A[M,K] --mte_gm_l1_frac/mte_gm_l1--> K1 M1 M0 K0 --mte_l1_l0a-->  K1 M1 M0 K0 -+
                                                             +-MAD-> N1 M1 M0 N0 --> C[M,N]
B[K,N] --mte_gm_l1_frac/mte_gm_l1--> K1 N1 K0 N0 --mte_l1_l0b--> K1 N1 N0 K0 -+
                               ^
                    transpose as part of mte_l1_l0b when requested
                    NOT at GM->L1
```

## Related Sections

- [Tile ISA: Matrix and Matrix-Vector](../tile/matrix-and-matrix-vector.md) — Tile-level matrix ops
- [Scalar DMA Copy](../scalar/dma-copy.md) — UB-side DMA grouped transfers
- [Pipeline Synchronization](../scalar/ops/pipeline-sync/) — Cube/Vector synchronization primitives
