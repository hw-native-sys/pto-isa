# NZ Fractal Layout

The cube's internal buffers (`L1` / `cbuf`, `L0A`, `L0B`, `L0C`) all use a **fractal NZ layout** rather than row-major ND. Understanding NZ layout is essential when authoring cube data-movement ops or reasoning about MAD operand organization.

## Definition

Given the hardware constant `C0 = 32 bytes`, for an element type with byte width `E = sizeof(T)`:

- Inner tile width: `K0 = N0 = C0 / E` (for example, `K0 = 16` for `f16` and `bf16`; `K0 = 8` for `f32`)
- Inner tile height: `M0 = 16`

NZ re-indexing for a logical `[M, K]` tensor:

```text
NZ index: (k1, m1, m0, k0)
  where  k1 = k / K0,  k0 = k % K0
         m1 = m / M0,  m0 = m % M0
Physical layout: K1 x M1 x M0 x K0  (last dimension contiguous)
```

The same outer / inner factorization is applied to `[K, N]` tensors, swapping the inner-width axis.

## Per-Buffer NZ Layouts

| Buffer | Logical shape | Physical NZ layout | Notes |
|--------|---------------|--------------------|-------|
| L1 (cbuf) — Tensor A | `[M, K]` | `K1 M1 M0 K0` | Row-major A staged into NZ layout |
| L1 (cbuf) — Tensor B | `[K, N]` | `K1 N1 K0 N0` | Row-major B staged into NZ layout |
| L0A (left operand)   | —         | `K1 M1 M0 K0` | FRACTAL_NZ on A5 / FRACTAL_ZZ on A3: same NZ order as L1 cbuf |
| L0B (right operand)  | —         | `K1 N1 N0 K0` | FRACTAL_ZN: row-major outer, col-major inner (K0 innermost) |
| L0C (accumulator)    | `[M, N]`  | `N1 M1 M0 N0` | Output of MMAD (FRACTAL_NZ: col-major outer, row-major inner) |

## Why K-Innermost on L0B?

The cube reduction axis is `K`. L0B requires K innermost (`K1 N1 N0 K0`) so the cube hardware reads all `K0` elements per cycle without striding.

The inner-box transpose is performed as part of the [`pto.mte_l1_l0b`](./ops/data-movement/mte-l1-l0b.md) structured right-load movement itself; no separate user-visible pass is required. Each 512B fractal Z-block is permuted as it moves from L1 to L0B.

## Data Flow: GM → L1 → L0A/B → L0C

```text
+------------------------------------------------------------------------------+
|              GEMM Data Layout: GM -> L1 (NZ) -> L0A/B -> L0C                |
+------------------------------------------------------------------------------+

STEP 1 - Global Memory (ND, row-major)
--------------------------------------
 Tensor A [M, K]                     Tensor B [K, N]
 (K is the contiguous axis)          (N is the contiguous axis)
 Physical: A[m*K + k]                Physical: B[k*N + n]

STEP 2 - GM -> L1 (cbuf): ND-to-NZ fractal repack
-------------------------------------------------
 A in L1: K1 x M1 x M0 x K0          B in L1: K1 x N1 x K0 x N0
 For each outer block (k1, m1):       For each outer block (k1, n1):
   inner is M0 rows x K0 cols           inner is K0 rows x N0 cols
   (16x16 elems contiguous)             (16x16 elems contiguous)
 Physical: A_nz[k1][m1][m0][k0]       Physical: B_nz[k1][n1][k0][n0]

STEP 3 - L1 -> L0A / L0B
--------------------------
 L0A: cbuf K1 M1 M0 K0 --mte_l1_l0a--> L0A K1 M1 M0 K0  (FRACTAL_NZ on A5)
 L0B: cbuf K1 N1 K0 N0 --mte_l1_l0b--> L0B K1 N1 N0 K0  (FRACTAL_ZN, K0 innermost)

STEP 4 - MAD: L0A x L0B -> L0C
-------------------------------
 dst[m, n] = sum k in 0..K-1: lhs[m, k] * rhs[k, n]
 L0C layout: N1 M1 M0 N0

STEP 5 - L0C writeback (FIXPIPE)
---------------------------------
 FIXPIPE MTE ops (mte_l0c_l1 / mte_l0c_gm / mte_l0c_ub) convert the L0C NZ
 result to the requested destination layout (typically ND) and memory space.
```

## Authoring Guidance

When the source GEMM operand is already in a transposed logical layout, express that at the structured load level (`pto.mte_l1_l0a` / `pto.mte_l1_l0b`) instead of relying on a later reinterpretation of the same bytes. Operating on a reinterpreted NZ buffer with the wrong outer / inner factorization is a verifier error and a common source of correctness bugs.

## Related Sections

- [Buffer Hierarchy](./buffer-hierarchy.md)
- [FIXPIPE Model](./fixpipe-model.md)
- [Cube MAD Ops](./README.md#matrix-multiply-mad-ops)
- [Cube Data Movement Ops](./README.md#cube-data-movement-ops)
