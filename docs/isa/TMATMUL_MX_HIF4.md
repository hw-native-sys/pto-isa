# TMATMUL_MX HiF4 (HiFloat4 Cube Matmul)

## Introduction

`TMATMUL_MX` with the `hifloat4x2_t` overload performs a Cube matmul where both
A and B operands are HiF4 (4-bit) packed data, each accompanied by the three-level
HiF4 scale (Ea/Eb/Ec). The result accumulates in FP32 in L0C, and is typically
cast to BF16 on store via FIXPIPE.

This is the A6 (dav-920r1) HiF4 matmul pipeline:
**TLOAD → TEXTRACT → TMATMUL_MX → TSTORE (FIXPIPE)**.

## Pipeline

```
GM (BF16) ──TLOAD──▶ L1 ──TEXTRACT──▶ L0A/L0B + L0AMX/L0BMX ──TMATMUL_MX──▶ L0C (FP32) ──TSTORE──▶ GM (BF16)
```

- **TLOAD**: loads the HiF4 data (GM→L1) and the scale bytes (GM→L1, HIF4_A_ZZ /
  HIF4_B_NN layouts). Data uses `copy_gm_to_cbuf_multi_nd2nz` (ND→NZ fractal).
- **TEXTRACT**: moves data L1→L0A/L0B via `load_c_buf_to_ca_s4`, and scale
  L1→L0AMX/L0BMX via `load_c_buf_to_ca_mx`.
- **TMATMUL_MX**: the Cube unit's `mad_mx` with the `hifloat4x2_t` type tag. This
  triggers the three-level Ea/Eb/Ec scale application internally.
- **TSTORE**: FIXPIPE casts FP32→BF16 and writes to GM.

## Scale Layout ([16,4] = 64B cell)

Each HiF4 scale patch covers one M- or N-fractal × one K-group (64 elements). The
patch is 64 bytes, internally:

```
bytes  0..31: [Ea(g0), Eb(g0), Ea(g1), Eb(g1), ... × 16 groups]  (EaEb half)
bytes 32..63: [Ec_lo(g0), Ec_hi(g0), ... × 16 groups]            (Ec half)
```

- **Ea**: 8-bit e6m2 exponent (per-64 group).
- **Eb**: 8 bits packed (one per 8-element subgroup), all 8 bits preserved.
- **Ec**: 16 bits packed (one per 4-element subgroup).

The CCE TQuant stores Eb via `pstu` (predicate → align register) + `vstas`
(align → UB), which packs the predicate at output frequency without the old
`DS_B8` downsample that dropped Eb bits b4–b7.

## L0C Capacity & N-tiling

L0C is 256 KB and accumulates in FP32 (4 B/element). When `M × N × 4 > 256 KB`,
the kernel tiles over N:

- `tileN = floor(L0C_SIZE / (M × 4))`, floored to 64 (TEXTRACT col alignment).
- A side (shared across N tiles) is loaded + extracted once before the loop.
- Each iteration extracts a `tileN`-wide B column slice, runs `TMATMUL_MX`, and
  stores the `M × tileN` chunk to GM at offset `j × tileN` with `stride = N`.

## Accumulator dtype

The L0C accumulator is always `float` (4 B), enforced by `CheckMadMxValid`:
`static_assert(Rows × Cols × sizeof(float) <= PTO_L0C_SIZE_BYTES)`.

## Testcases

| Testcase | Shapes | Purpose |
|---|---|---|
| `tmatmul_mx_hif4` | 128×128×128, 128×256×128, 256×128×128, 64×64×64, 256×256×256, 128×512×128, 512×128×512, 128×128×256, 256×128×512 | HiF4 Cube matmul end-to-end |
| `tmatmul_mx_e1m2` | 128×128×128 | e1m2 MX oracle (same pipeline, plain MX scale) |

## References

- HiF4 standard: `docs/isa/TQUANT_HIF4.md`, arXiv:2602.11287
- CCE implementation: `include/pto/npu/a6/TQuant.hpp`, `TMatmul.hpp`, `TLoad.hpp`, `TExtract.hpp`
