# TREMS / TREM High-Precision Remainder on A2/A3 — Algorithm, Interface and Tiling Budget

This document describes the A2/A3 (Ascend910B1) implementation of the
high-precision FP32 remainder from
[issue #2](https://gitcode.com/bynshard/pto-isa/issues/2): the scalar-divisor
`TREMS<RemSAlgorithm::HIGH_PRECISION>` and the vector-divisor
`TREM<RemAlgorithm::HIGH_PRECISION>`. It covers the numerical algorithm, the
calling interfaces, the `tmp` tile memory contracts (including which scratch
regions deliberately alias each other and why that is safe), and the maximum
tile sizes that fit the 192 KB A2/A3 UB budget.

Code: `include/pto/npu/a2a3/custom/TFmodRemHp.hpp` (algorithm),
`include/pto/npu/a2a3/custom/TRemHpStaged.hpp` (staged / per-vector drivers),
dispatch in `include/pto/npu/a2a3/TRemS.hpp` and `TRem.hpp`.

---

## 1. Semantics

`dst = rem(dst, src, scalar)` computes the **floored remainder**

$$
\begin{aligned}
r &= x - \lfloor x / s \rfloor \cdot s && \text{(as if computed in exact arithmetic, rounded once to f32)} \\
r &\leftarrow r + s && \text{if } r \text{ and } s \text{ have different signs}
\end{aligned}
$$

with IEEE special values (`x % s` in numpy semantics):

| x | s | result |
|---|---|--------|
| ±inf | finite | NaN |
| finite | ±inf | x (NaN if x is also ±inf) |
| ±0 | s ≠ 0 | ±0 (sign of x) |
| NaN | any | NaN |

This differs from the DEFAULT (fast) TREMS path, which computes
$\lfloor x/s \rfloor$ in one f32 division: whenever $|x/s| > 2^{24}$ the
single-shot product $q \cdot s$ rounds at the magnitude of $x$, the
subtraction cancels catastrophically, and the remainder comes out as
$\pm(1\ldots 2)\,\mathrm{ulp}(x)$ (or $\pm\infty$ at the FLT_MAX
boundary). The high-precision path resolves the remainder to full
f32 precision; its reference golden is simply `np.float32(x) % np.float32(s)`.

A scalar of `0` is not special-cased (same as the A5 implementation): the
iteration arithmetic propagates it. Use a nonzero finite scalar.

## 2. Algorithm — base-$2^{24}$ digit-peeling long division

The numerics follow the A5 implementation (`a5/custom/TFmodRemHp.hpp`, same
family as AscendC's `fmod_3510`).

### 2.1 Why 11 iterations

Work on $\mathit{state} = |x|$ and $\mathit{div} = |s|$. The f32 quotient
range spans up to $\sim 2^{276}$ (FLT_MAX divided by the minimum
subnormal). The algorithm peels that exponent gap from the top in ~24-bit
chunks:


| iteration (runs high→low) | divisor scale $\mathit{div} \cdot 2^{e}$ | chunk handled |
|---|---|---|
| 11 | $2^{240}$ |   |
| 10 | $2^{216}$ |   |
| 9 | $2^{192}$ |   |
| 8 | $2^{168}$ |   |
| 7 | $2^{144}$ |   |
| 6 | $2^{120}$ |   |
| 5 | $2^{96}$ |   |
| 4 | $2^{72}$ |   |
| 3 | $2^{48}$ |   |
| 2 | $2^{24}$ |   |
| 1 | $2^{0}$ (unscaled) + subnormal regime | final floor |

$\lceil 276/24 \rceil = 11$ covers the whole range (the top ladder entries overflow
to inf for large `|s|`; the `vmins` clamp and the saturating integer convert
act as a quotient clamp, so those iterations degrade to no-ops instead of
corrupting the state).

### 2.2 One iteration

$$
\begin{aligned}
b &= \mathit{div} \cdot 2^{e(i)} \\[2pt]
k &= \begin{cases} \lfloor \mathit{state}/b \rfloor & \text{remainder (\texttt{ROUND\_F})} \\[2pt]
      \operatorname{trunc}(\mathit{state}/b) & \text{fmod (\texttt{ROUND\_Z}}) \end{cases} \\[2pt]
\mathit{state} &\leftarrow \mathit{state} - k \cdot b
\end{aligned}
$$

$b$ is an exact power-of-two scaling of the divisor. $k$ is an integer and
therefore exact in f32; the int32 conversion saturates, which acts as a free
quotient clamp at extreme exponent gaps. The subtraction cancels the top
~24 bits of the gap with at most 1 ulp of rounding at the chunk's magnitude —
never at $|x|$'s.

Iteration 1 additionally:
- if $|s| \le \mathit{minNormal}$: scale $\mathit{state}$ and the divisor by
  $2^{24}$ first (lifts subnormals into the normal range; the post pass
  multiplies back by $2^{-24}$);
- final unscaled $k = \lfloor \mathit{state}/|s| \rfloor$;
- $\mathit{state} \leftarrow \mathit{state} + [\,\mathit{state} < 0\,] \cdot b$
  (subnormal-scale sign fix; $[\,P\,]$ is the Iverson bracket, 1 when $P$
  holds and 0 otherwise).

Post pass (`HpPostS`): dividend sign restore
($\mathit{state} \leftarrow -\mathit{state}$ where $\mathit{src0} < 0$),
subnormal rescale ($\mathit{state} \leftarrow \mathit{state} \cdot 2^{-24}$ when
$|s| \le \mathit{minNormal}$),
exception fixups from §1, remainder sign correction
($\mathit{state} \cdot s < 0 \ \Rightarrow\ \mathit{state} \leftarrow \mathit{state} + s$).

For a quotient that already fits in 24 bits (e.g. `x = 100.7, s = 7.5`)
iterations 11..2 compute `k = 0` and iteration 1 alone produces the result;
the ladder exists for huge exponent gaps (`x = 1e30`, FLT_MAX dividends,
subnormal divisors).

## 3. Interface

```cpp
#include <pto/npu/a2a3/TRemS.hpp>

TREMS<RemSAlgorithm::HIGH_PRECISION>(dstTile, srcTile, scalar, tmpTile);
TREM<RemAlgorithm::HIGH_PRECISION>(dstTile, src0Tile, src1Tile, tmpTile);  // vector divisor
```

- Element type: `float` only (`HIGH_PRECISION` + non-float falls back to the
  static_assert; int32 keeps the DEFAULT path).
- Tiles: `TileType::Vec`, row-major, dst and src(s) with identical valid
  shapes. Tile `Cols` must satisfy `Cols * sizeof(float) % 32 == 0`; express
  odd valid widths as `validCol < tileCols`.
- TREM divisor contract: src1 elements must be nonzero and not NaN (same
  domain as the A5 implementation). $\pm\infty$ is fine ($x \bmod \pm\infty = x$).
  The guaranteed domain is $|y| \ge 1$ (or $\pm\infty$): the remainder sign
  correction probes $r \cdot y$, which underflows to $\pm 0$ when
  $|r \cdot y| < 2^{-149}$ and would then miss the correction — an A5
  limitation inherited by design.
  Smaller divisors follow the A5 iteration arithmetic including that
  limitation; NaN divisors propagate through the iterations but race the
  `src0 == +-0` fixup, and +-0 divisors are outside the contract.
- Dispatch (mirrors the A5 structure; TREM staging additionally requires dst
  disjoint from BOTH src0 and src1):

| condition | path |
|---|---|
| `validRows * (validCols / 64) >= TREM_HP_STAGED_MIN_VECS` **and** dst tile does not alias the src tile | **staged** (loop-inverted, §4) |
| otherwise (small tiles, aliased tiles, row tails) | **per-vector** fallback: all 11 iterations per 64-element vector through the scratch state buffer |

  `TREM_HP_STAGED_MIN_VECS` defaults to 48 and can be overridden at build
  time (`-DTREM_HP_STAGED_MIN_VECS=...`).
- The staged path additionally requires `validCols / 64 <= 255` (the
  row-batching repeat count is 8-bit). Wider rows assert; split the tile.
- `tmp` tile: 1 row, `HpScratchRequiredCols(cap)` columns (§5; TREM uses the
  V variant from §8.1). Both ops validate this at runtime.

### 3.1 Usage example

```cpp
#include <pto/pto-inst.hpp>
#include <pto/npu/a2a3/TRemS.hpp>
#include <pto/npu/a2a3/TRem.hpp>

// Scalar divisor, 8x448 tile: dst = rem(x, 7.5f) at full f32 precision.
constexpr unsigned capVecs  = 448 / 64;                       // full vectors per row
constexpr unsigned maskF    = ((capVecs * 8 + 31) / 32) * 8;  // packed mask, 32B units
constexpr unsigned tmpCols  = 128 * capVecs + maskF + 8;      // TREMS contract (§5)

using DstTile = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1>;
using SrcTile = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1>;
using TmpTile = Tile<TileType::Vec, float, 1, tmpCols, BLayout::RowMajor, -1, -1>;

DstTile dst(8, 448);
SrcTile src(8, 448);
TmpTile tmp(1, tmpCols);
// ... TASSIGN the tiles into UB and TLOAD src (see the test kernels) ...
TREMS<RemSAlgorithm::HIGH_PRECISION>(dst, src, 7.5f, tmp);

// Vector divisor: same shapes, tmp sized with the 192C variant (§8.1)
TREM<RemAlgorithm::HIGH_PRECISION>(dst, src0, src1, tmpV);
```

Complete kernels (TASSIGN/TLOAD/TSTORE plumbing, strided and tailed tiles)
live in `tests/npu/a2a3/src/st/testcase/trems/trems_kernel.cpp` and
`tests/npu/a2a3/src/st/testcase/trem/trem_kernel.cpp`.

## 4. Staged execution (the issue #2 reordering, A2/A3 form)

Two execution organizations exist. The per-vector path runs all 11 iterations
on one 64-element vector at a time through the scratch state buffer. The
staged path loop-inverts the iteration loop: **every full vector advances one
SolveScale iteration per pass**, with the per-vector state staged in the dst
rows. On A2/A3 each op of a pass is one repeat instruction covering all full
vectors of a row:

```
seed:      dst[row] = |src[row]|                     (vabs, rep = fullPerRow)
iters 11→1: for each row: SolveScale iteration         (ops with rep = fullPerRow,
             on dst[row] with state in place            state = dst[row])
post:      for each row: HpPostS on dst[row]
tails:     per-vector fallback on the <64-element tail of each row
```

## 5. `tmp` tile contract — layout and safe aliasing

with $C = \max(\mathit{validCols}/64,\ 1)$ the vector capacity and
$\mathit{maskF}(C) = \lceil 8C/32 \rceil \cdot 8$ floats:

$$\mathit{HpScratchRequiredCols}(C) = 128C + \mathit{maskF}(C) + 8 + 64\,[\,C = 1\,]\ \text{floats}$$

| region | floats | lifetime | notes |
|---|---|---|---|
| `bTmp` | `[0, 64C)` | within one iteration | scaled divisor, `vector_dup`'d per iteration from host scalars — **no divisor buffer lives across iterations** |
| `k` | `[64C, 128C)` | see below | quotient **and** post-quotient temp share this region |
| `cmpMask` | `[128C, +maskF)` | from `vcmp` until the `vsel` that consumes it | packed compare bitmask, 32-byte aligned |
| `addrBuf` | `+8` | around each `set_cmpmask` | holds the full 64-bit address of `cmpMask` |
| `state` | `+64`, only when `C == 1` | whole per-vector call | the per-vector fallback's state; the staged path keeps state in dst, so this never scales with `C` |

The two aliasing decisions, and why they are bug-free:

1. **quotient / temp sharing (`k`)** — `k` is dead the moment
   `vmul(k, k, bTmp)` reads it to form the $k \cdot \mathit{bTmp}$ product; everything the
   temp is used for afterwards (sign factor, select operands, probe,
   correction candidate) begins with a fresh write of the region. In-place
   `dst == src0` vector ops are the established A2/A3 idiom (e.g. TPow's
   `vmul(dst, dst, exp)`). Within the post pass the probe
   $k = \mathit{state} \cdot \mathit{scalar}$ is consumed by `vcmpvs_lt` (with a `pipe_barrier`)
   **before** `vadds(k, state, scalar)` overwrites it with the correction
   candidate — every handoff is separated by a barrier.
2. **no persistent sign/constant buffers** — A5's `GetSignBit` needed a
   `vshrs`-based bit extract plus 1.0/0.0 vectors. The A2/A3 spelling does the
   sign restore as `vcmpvs_lt(src0, 0)` + `vmuls(k, state, -1)` + `vsel`,
   which needs no storage beyond `k`, and the iteration-1 fix as
   `k = (state < 0) ? bTmp : 0` via the same mask machinery.


## 6. Maximum tiling given the 192 KB UB budget

A2/A3 (Ascend910B1) UB limit: **196608 B** (compile-time check,
`error: UBUF memory (...) exceeds limit`). For a packed tile
(src = dst = `R × W`, `validCols = W`):

$$
\mathit{UBUF}(R, W) \;=\; \underbrace{8\,R\,W}_{\text{src + dst tiles}} \;+\;
\underbrace{4\,\bigl(128C + \mathit{maskF}(C) + 8\bigr)}_{\text{tmp}} \;\le\; 196608\ \mathrm{B},
\qquad C = \max(W/64,\ 1),\quad C \le 255\ \text{(8-bit repeatTimes)}
$$

Largest packed `W` (multiple of 64) per row count:

| rows R | max W | UBUF at max W | C |
|---|---|---|---|
| 1 | 12160 | 196128 | 190 |
| 2 | 8128 | 196128 | 127 |
| 3 | 6080 | 195360 | 95 |
| 4 | 4864 | 195200 | 76 |
| 7 | 3008 | 192928 | 47 |
| 8 | 2688 | 193920 | 42 |
| 12 | 1856 | 193312 | 29 |
| 16 | 1408 | 191712 | 22 |
| 32 | 704 | 185984 | 11 |

(All of these stay above the 48-full-vector staging gate, so they run
staged.) Worked examples from the test suite: `1×8192` → 132128 B, `2×2048`
→ 49440 B, `8×448` → 32352 B.

### 6.1 TREM budget

TREM carries three tiles (dst, src0, src1) and the 192C-per-vector scratch
from section 8:

$$
\mathit{UBUF}(R, W) \;=\; \underbrace{12\,R\,W}_{\text{dst + src0 + src1 tiles}} \;+\;
\underbrace{4\,\bigl(192C + \mathit{maskF}(C) + 8\bigr)}_{\text{tmp}} \;\le\; 196608\ \mathrm{B},
\qquad C = \max(W/64,\ 1),\quad C \le 255
$$

Largest packed `W` (multiple of 64) per row count:

| rows R | max W | UBUF at max W | C |
|---|---|---|---|
| 1 | 8128 | 196128 | 127 |
| 2 | 5440 | 196576 | 85 |
| 3 | 4032 | 194080 | 63 |
| 4 | 3264 | 196288 | 51 |
| 7 | 1984 | 190752 | 31 |
| 8 | 1792 | 193792 | 28 |
| 12 | 1216 | 189888 | 19 |
| 16 | 960 | 196000 | 15 |
| 32 | 448 | 177504 | 7 |

Rule of thumb: $W \lesssim 16384/(R+1)$ — one notch below TREMS because of the
second input tile. Worked examples: `1×8128` → 196128 B (the widest legal
single row), `2×2048` → 74016 B, `8×448` → 48480 B.

Rules of thumb:
- **Width dominates**: scratch grows with $W$ only ($\sim 8W$ bytes), tiles
  grow with $RW$. Roughly $W \lesssim 24576/(R+1)$.
- Tiles wider than the table allow must be **split into multiple TREMS
  calls** (column blocking) — the op processes one tile per invocation.
- The DEFAULT (fast) TREMS path has a much smaller tmp contract
  (`dstRowStride + maskF + 8` floats), so the same tile can be legal for
  DEFAULT and illegal for HIGH_PRECISION; check the budget for the precision
  mode you dispatch.
- `validCol < tileCols` (tails) costs nothing extra in tmp; tails always take
  the per-vector path regardless of the staging gate.

### 6.2 Per-tile occupancy in terms of the tile dimensions $[M, N]$

For a packed tile of $M$ rows × $N$ columns, the tmp occupancy depends only on
the width $N$ (the vector capacity $C$); the data tiles depend on $M \cdot N$:

$$
\mathit{tmpCols}(N)=
\begin{cases}
208, & \text{TREMS},\ C=1 \\[2pt]
128C+8\lceil C/4\rceil+8, & \text{TREMS},\ C\ge 2 \\[2pt]
272, & \text{TREM},\ C=1 \\[2pt]
192C+8\lceil C/4\rceil+8, & \text{TREM},\ C\ge 2
\end{cases}
\qquad
C=\max\Bigl(\Bigl\lfloor \frac{N}{64}\Bigr\rfloor,\ 1\Bigr)
$$

The $8\lceil C/4\rceil$ term is the packed compare mask (32-byte padded,
§5); the constant $C=1$ branch folds in the 64-float per-vector state buffer.

$$
\mathit{UBUF}(M,N) \;=\;
\underbrace{n_{\text{tiles}}\cdot 4MN}_{\text{data tiles}} \;+\;
\underbrace{4\,\mathit{tmpCols}(N)}_{\text{tmp}}
\;\le\; 196608\ \mathrm{B},
\qquad n_{\text{tiles}} = 2\ \text{(TREMS)},\ 3\ \text{(TREM)}
$$

(For padded/strided tiles, size each data tile by its physical
$M \times \mathit{Cols}$ but keep the tmp contract on the **valid** width;
the $+64$ floats appear only when a row holds a single full vector or a pure
tail, funding the per-vector state buffer.)

Worked examples — byte size of each tile:

TREMS:

| tile $[M, N]$ | dst | src | tmp | total UBUF | of 196608 B | dispatch ($M \cdot C$ vs 48) |
|---|---|---|---|---|---|---|
| $[1, 64]$ | 256 | 256 | 832 | 1344 | 1% | per-vector (1) |
| $[32, 64]$ | 8192 | 8192 | 832 | 17216 | 9% | per-vector (32) |
| $[12, 256]$ | 12288 | 12288 | 2112 | 26688 | 14% | staged (48) |
| $[8, 448]$ | 14336 | 14336 | 3680 | 32352 | 16% | staged (56) |
| $[2, 2048]$ | 16384 | 16384 | 16672 | 49440 | 25% | staged (64) |
| $[1, 8192]$ | 32768 | 32768 | 66592 | 132128 | 67% | staged (128) |
| $[8, 2688]$ | 86016 | 86016 | 21888 | 193920 | 99% | staged (336) |
| $[1, 12160]$ | 48640 | 48640 | 98848 | 196128 | 100% | staged (190) |

TREM:

| tile $[M, N]$ | dst | src0 | src1 | tmp | total UBUF | of 196608 B | dispatch |
|---|---|---|---|---|---|---|---|
| $[1, 64]$ | 256 | 256 | 256 | 1088 | 1856 | 1% | per-vector (1) |
| $[32, 64]$ | 8192 | 8192 | 8192 | 1088 | 25664 | 13% | per-vector (32) |
| $[8, 448]$ | 14336 | 14336 | 14336 | 5472 | 48480 | 25% | staged (56) |
| $[2, 2048]$ | 16384 | 16384 | 16384 | 24864 | 74016 | 38% | staged (64) |
| $[1, 8128]$ | 32512 | 32512 | 32512 | 98592 | 196128 | 100% | staged (127) |

Design checklist:
- Pick the working width $N$ first — it alone fixes the tmp size; then bound
  the rows by $M \le (196608 - 4\,\mathit{tmpCols}(N))\,/\,(4N \cdot
  \#\text{tiles})$.
- $C \le 255$ and $M \cdot C \ge 48$ if you want the staged path.
- Column-block anything larger: each call processes one tile.

## 7. Tests

`tests/npu/a2a3/src/st/testcase/trems` — cases 50–59 cover staged,
register-fallback, the 48-vector threshold, tails, strided dst and a negative
scalar, against `np.float32(x) % np.float32(s)` goldens with corner pools
(±0, ±inf, NaN, subnormals, FLT_MAX). The comparator allows $8\cdot\mathrm{ulp}$
cancellation slack (§1's ±ulp(x) artifacts do not exist in the HP path, but
the hardware divider/multiplier deviates from IEEE-RN by ~2 ulp and the
post-pass multiplication by `s` rounds at `|s|`'s scale for huge dividends).


---

## 8. TREM — the vector-divisor variant

`TREM<RemAlgorithm::HIGH_PRECISION>` runs the same 11-iteration digit-peeling
(§2) with the same staged loop inversion (§4) and the same dispatch rules
(48-vector gate, dst disjoint from the inputs, ≤255 vectors per row). The differences all come from the
divisor being a tensor:

1. **Per-pass divisor reload.** Each pass re-reads `|src1|` from the original
   tile (`vabs` into the scratch `div` region), because a scaled divisor must
   not feed the next pass. The seed, iteration and postprocessing passes are
   otherwise identical to §4.
2. **Per-element subnormal regime.** With a scalar divisor, "is the divisor
   subnormal" is a host-side constant. With a vector divisor, iteration 1
   decides per element: where $|\mathit{src1}| \le \mathit{minNormal}$, lift divisor
   AND state by $2^{24}$ (packed-mask compare + two selects), and the post
   pass rescales exactly those lanes by $2^{-24}$ (again per element).
3. **Exception fixups become per-element selects** (A5's
   `SolveExceptionScenarios`): `src1 = ±inf → src0`; then (after it, since
   the previous select would otherwise leave the dividend there)
   `src0 = ±inf ∧ src1 = ±inf → NaN`; then `src0 = ±0` with an in-domain
   divisor `→ src0`. `x = ±inf` with a finite divisor already collapses to
   NaN inside the ladder (`inf / inf` in the overflowing top iterations), so
   it needs no explicit select.
4. **The sign correction multiplies by the divisor row**
   ($\mathit{state} \cdot \mathit{src1} < 0 \ \Rightarrow\ \mathit{state} \leftarrow \mathit{state} + \mathit{src1}$)
   instead of a scalar.

### 8.1 TREM `tmp` contract — one region more than TREMS, and why

$$\mathit{HpScratchRequiredColsV}(C) = 192C + \mathit{maskF}(C) + 8 + 64\,[\,C = 1\,]\ \text{floats}$$

| region | floats | lifetime |
|---|---|---|
| `div` | `[0, 64C)` | whole pass: re-`vabs`'d from src1 each pass, read by every iteration's divisor scaling and by iteration 1's subnormal branch |
| `bTmp` | `[64C, 128C)` | within one iteration (scaled divisor) |
| `k` | `[128C, 192C)` | quotient, then (after the product `vmul`) the shared temp — same k/t aliasing as TREMS §5 |
| `cmpMask`, `addrBuf`, `state` | as in §5 | |

TREMS folds its divisor away entirely (128C) because a scalar can be
`vector_dup`'d into `bTmp` per iteration directly. TREM's divisor is a
tensor, so the unscaled `|src1|` base must **persist across the pass's
iterations** as the rescaling source — it cannot live in `bTmp` (iteration 1
needs the unscaled value after `bTmp` is written) nor in `k` (the quotient
overwrites it mid-iteration). Dropping the third region would require
re-`vabs`-ing the divisor into `bTmp` in every iteration; the contract keeps
the third region instead.

## 9. Tests (both ops)

- TREMS `tests/npu/a2a3/src/st/testcase/trems` cases 50–59: staged, register
  fallback, the 48-vector threshold, tails, strided dst and a negative
  scalar.
- TREM `tests/npu/a2a3/src/st/testcase/trem` `case_hp_*` (10 cases): the
  same coverage plus the widest legal single row (`1×8128`, the §6.1 R=1
  maximum); the HP divisor corner pool excludes ±0 and NaN per the §3
  divisor contract while the dividend pool keeps the full corner set
  (±0, ±inf, NaN, subnormals, FLT_MAX).

Both suites compare against `np.float32(x) % np.float32(divisor)` goldens
with 8-ulp cancellation slack (§7's rationale) plus a last-digit-collapse
rule: when the final iteration's $k \cdot \mathit{bTmp}$ rounds to exactly the state on a
half-ulp tie, the staged remainder zeroes out while the true floored
remainder lies within one divisor of zero on either side — A5 shows the same
artifact — so a collapsed-to-zero result is accepted when the golden is
within one divisor of zero.
