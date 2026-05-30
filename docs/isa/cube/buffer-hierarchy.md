# Cube Buffer Hierarchy

The cube core (AIC) operates on a dedicated buffer hierarchy distinct from the Unified Buffer (UB) that Vector blocks use. Cube operands move through `L1` (cbuf) → `L0A` / `L0B` → `L0C` → writeback, with optional `BT` (bias table) and `FB` (FIXPIPE buffer) helpers.

## Address Spaces

| Space | Role | Layout | Typical Producer | Typical Consumer |
|-------|------|--------|------------------|------------------|
| `gm`  | Global Memory (off-chip HBM/DDR) | ND row-major | host / kernel | DMA loaders |
| `l1`  | Cube CBUF, ~1 MB on-chip | NZ fractal | `pto.mte_gm_l1`, `pto.mte_gm_l1_frac`, `pto.mte_ub_l1` | `pto.mte_l1_l0a`, `pto.mte_l1_l0b`, `pto.mte_l1_ub`, `pto.mte_l1_bt` |
| `l0a` | Cube left-operand scratchpad | FRACTAL_NZ (A5) / FRACTAL_ZZ (A3) | `pto.mte_l1_l0a` | `pto.mad*` |
| `l0b` | Cube right-operand scratchpad | FRACTAL_ZN (K innermost) | `pto.mte_l1_l0b` | `pto.mad*` |
| `l0c` | Cube accumulator | FRACTAL_NZ output of MMAD | `pto.mad*` | FIXPIPE writeback (`pto.mte_l0c_*`) |
| `bt`  | Bias Table | element-type-matched vector | `pto.mte_l1_bt` | `pto.mad_bias`, `pto.mad_mx_bias` |
| `fb`  | FIXPIPE auxiliary buffer | implementation-defined | `pto.mte_l1_fb` | FIXPIPE writeback ops |
| `ub`  | Vector Unified Buffer | ND | DMA loaders | vector pipe |

See [NZ Fractal Layout](./nz-fractal-layout.md) for the precise per-buffer NZ index orders.

## Data-Flow Contract

```text
                +----------------- AIC issue queues -----------------+
                |    MTE2     MTE1    CUBE (MMAD)    FIXP            |
                |     |         |        |             |             |
GM (ND) --- pto.mte_gm_l1 / pto.mte_gm_l1_frac        |             |
              |   |                                                  |
              v   v                                                  |
              L1 (NZ) <-- pto.mte_ub_l1 --- UB                       |
              |                                                      |
       +------+-----+---------------------+                          |
       |            |                     |                          |
   mte_l1_l0a   mte_l1_l0b           mte_l1_bt / mte_l1_fb            |
       |            |                     |                          |
       v            v                     |                          |
      L0A          L0B                    |                          |
       |            |                     |                          |
       +-----+------+                     |                          |
             |                            |                          |
             |     pto.mad / pto.mad_acc / pto.mad_bias / *_mx*       |
             |     <----------------------+                          |
             v                                                       |
            L0C                                                      |
             |                                                       |
             +-- pto.mte_l0c_l1 / pto.mte_l0c_gm / pto.mte_l0c_ub ---+
                  (FIXPIPE writeback)
```

## Alignment and Sizing Conventions

- All cube buffer pointers (L1 / L0A / L0B / L0C / BT / FB) are 32-byte aligned.
- L0A and L0B fractal tiles are 512B (one 32B-wide × 16-row block in the appropriate inner orientation).
- L0C accumulator tiles use the `N1 M1 M0 N0` order so that FIXPIPE can stream out one M-row of results at a time.
- Element-type-derived inner widths (`K0 = N0 = C0 / sizeof(T)`) follow [NZ Fractal Layout](./nz-fractal-layout.md).

## Synchronization

The cube programs are issued from the AIC's Scalar Unit (SU) into the MTE2 / MTE1 / CUBE / FIXP issue queues. Synchronization with the Vector blocks happens through the System Controller (SC) semaphores and the dedicated 1:2 fixpipe broadcast path. See:

- [Pipeline Synchronization](../scalar/ops/pipeline-sync/) for the intra-block (`pto.set_flag` / `pto.wait_flag`) primitives that order MTE2 → MTE1 → CUBE → FIXP within the AIC.
- [Cluster Programming Model](../machine-model/execution-agents.md) for inter-block (`pto.set_intra_block` / `pto.wait_intra_core`) primitives used between AIC and AIV.

## Related Sections

- [NZ Fractal Layout](./nz-fractal-layout.md)
- [FIXPIPE Model](./fixpipe-model.md)
- [Cube Data Movement Ops](./README.md#cube-data-movement-ops)
