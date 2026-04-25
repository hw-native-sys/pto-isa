# Memory Optimization

This document covers memory optimization techniques for PTO kernels. For the abstract on-chip memory model see [`docs/machine/abstract-machine.md`](../machine/abstract-machine.md); for example-driven tuning see [`opt.md`](opt.md).

---

## 1. On-Chip Storage Model

PTO exposes on-chip storage through tile types. Each `TileType` maps to a storage class:

| Tile type | Storage | Typical use |
|-----------|---------|-------------|
| `TileType::Vec` | Vector buffer (UB) | Elementwise ops, reductions |
| `TileType::Mat` | Matrix staging (L1-like) | TLOAD target, GEMM staging |
| `TileType::Left/Right` | Matrix operand regs (L0A/L0B) | TMATMUL inputs |
| `TileType::Acc` | Accumulator regs (L0C) | TMATMUL output |

Data flows along a fixed path:

```
GM --TLOAD--> Mat --TMOV/TEXTRACT--> Left/Right --TMATMUL--> Acc --TSTORE--> GM
```

Vector ops (`TADD`, `TEXP`, `TROWSUM`, etc.) work directly on `Vec` tiles.
Conversion between `Mat` and `Vec` uses `TMOV` or `TEXTRACT`.

---

## 2. Tile Sizing and Capacity

### Estimate footprint before declaring tiles

```cpp
constexpr int TM = 128, TK = 64, TN = 256;
// Mat staging (double-buffered)
constexpr size_t staging = 2*(TM*TK + TK*TN)*sizeof(half);  // 2*(16+32) KB = 96 KB
// Accumulator
constexpr size_t accum   = TM * TN * sizeof(float);          // 128 KB
// Total: 224 KB -- must fit within target on-chip limit
```

If the total exceeds the limit: reduce tile dimensions, drop to single buffering, or split the accumulator range.

### Alignment rules (enforced by `static_assert`)

- Row-major tile: `Cols * sizeof(Element)` must be a multiple of 32 bytes.
- Col-major tile: `Rows * sizeof(Element)` must be a multiple of 32 bytes.
- Boxed tiles (`TileLeft`, `TileRight`, `TileAcc`): shape must match fractal base-tile size (`fractalABSize=512`, `fractalCSize=1024`).

Common `TileLeft`/`TileRight` base shapes for `fractalABSize=512`:

| Element | Base rows×cols |
|---------|---------------|
| `fp32`  | 16×8 |
| `fp16`  | 16×16 |
| `int8`  | 16×32 |

---

## 3. Data Reuse

### K-dimension blocking (GEMM)

Load each A/B panel once, accumulate across the full K loop:

```cpp
TileAcc<float, TM, TN> acc;
TFILL(acc, 0.0f);

for (int k = 0; k < K; k += TK) {
    Tile<TileType::Mat, half, TM, TK> a_mat;
    Tile<TileType::Mat, half, TK, TN> b_mat;
    TLOAD(a_mat, gA_view);    // 1 GM access per TM×TK block
    TLOAD(b_mat, gB_view);

    TileLeft<half, TM, TK>   a_left;
    TileRight<half, TK, TN>  b_right;
    TMOV(a_left, a_mat);
    TMOV(b_right, b_mat);

    TMATMUL_ACC(acc, a_left, b_right);  // acc stays on-chip
}
TSTORE(gC_view, acc);   // 1 GM write at the end
```

### Cache row statistics (Softmax)

```cpp
Tile<TileType::Vec, float, R, C> input, shifted, exp_v, output;
Tile<TileType::Vec, float, R, 1>  row_max, row_sum;

TLOAD(input, gInput);
TROWMAX(row_max, input);             // computed once, stays on-chip
TROWEXPANDSUB(shifted, input, row_max);
TEXP(exp_v, shifted);
TROWSUM(row_sum, exp_v);             // computed once, stays on-chip
TROWEXPANDDIV(output, exp_v, row_sum);
TSTORE(gOutput, output);
```

Storing and reloading `row_max`/`row_sum` would roughly triple GM traffic for the statistics.

---

## 4. Layout and Alignment

**Match layout to the consuming instruction** to avoid implicit conversions.

- Load `Mat` tiles with `BLayout::RowMajor` when GM data is row-major.
- Use `Layout::NZ` on `GlobalTensor` for GEMM inputs to eliminate the `TMOV` stage on supported targets.
- Use `TTRANS` only when source and target layouts genuinely differ.

`TileLeft` / `TileRight` / `TileAcc` aliases automatically select the correct `SLayout` and `SFractalSize`:

```cpp
TileLeft<half, 128, 64>   a_left;   // outer col-major + inner row-major
TileRight<half, 64, 256>  b_right;  // outer row-major + inner col-major
TileAcc<float, 128, 256>  acc;
```

Do not override `SLayout` or `SFractalSize` manually without a specific reason.

---

## 5. Reducing GM Traffic

### Operator fusion

Fuse consecutive operations to eliminate intermediate GM stores/reloads:

```cpp
// Unfused: 4 GM round-trips
TLOAD(a, gInput); TADD(b, a, s); TSTORE(gTmp, b);
TLOAD(c, gTmp);   TMUL(d, c, s2); TSTORE(gOut, d);

// Fused: 2 GM round-trips
TLOAD(a, gInput);
TADD(b, a, s);    // b stays on-chip
TMUL(d, b, s2);
TSTORE(gOut, d);
```

### Contiguous access

Prefer row-major traversal; strided column access reduces burst efficiency.
If column access is unavoidable, load a row-major block then `TTRANS` on-chip.

### TPREFETCH

Issue non-blocking hints to overlap data movement with compute:

```cpp
if (k + TK < K) { TPREFETCH(gA_next); TPREFETCH(gB_next); }
TMATMUL_ACC(acc, a_left, b_right);  // overlaps with prefetch
```

---

## 6. Double Buffering

Alternate between two staging tile sets to overlap the memory and compute pipelines:

```cpp
Tile<TileType::Mat, half, TM, TK> a_mat[2];
Tile<TileType::Mat, half, TK, TN> b_mat[2];
TileLeft<half, TM, TK>            a_left[2];
TileRight<half, TK, TN>           b_right[2];

Event<Op::TLOAD, Op::TMOV>    ev_load[2];
Event<Op::TMOV,  Op::TMATMUL> ev_mov[2];

// Warm-up
ev_load[0] = TLOAD(a_mat[0], gA_view_0);
ev_load[0] = TLOAD(b_mat[0], gB_view_0);

for (int k = 0, ping = 0; k < K; k += TK, ping ^= 1) {
    int pong = ping ^ 1;
    // Load next (pong) while computing current (ping)
    if (k + TK < K) {
        ev_load[pong] = TLOAD(a_mat[pong], gA_view_next);
        ev_load[pong] = TLOAD(b_mat[pong], gB_view_next);
    }
    ev_mov[ping]  = TMOV(a_left[ping],  a_mat[ping],  ev_load[ping]);
    ev_mov[ping]  = TMOV(b_right[ping], b_mat[ping],  ev_load[ping]);
    TMATMUL_ACC(acc, a_left[ping], b_right[ping], ev_mov[ping]);
}
TSTORE(gC_view, acc);
```

Steady-state throughput approaches `max(T_load, T_compute)` instead of `T_load + T_compute`.

For the event model see [`Event.md`](Event.md).

---

## 7. Valid Region and Padding

When dimensions are not multiples of tile sizes, use the valid-region rather than allocating multiple tile sizes:

```cpp
// Static valid region
using TileT = pto::Tile<pto::TileType::Vec, float, 128, 256,
                        pto::BLayout::RowMajor, 100, 200>;

// Dynamic valid region
using TileD = pto::Tile<pto::TileType::Vec, float, 128, 256,
                        pto::BLayout::RowMajor,
                        pto::DYNAMIC, pto::DYNAMIC>;
TileD t(actual_rows, actual_cols);

// Pad capacity to alignment; use valid region for actual extent
constexpr int PADDED = ((127*4+31)/32)*32/4;  // 127 > 128 cols
using TileP = pto::Tile<pto::TileType::Vec, float, 16, PADDED,
                        pto::BLayout::RowMajor, 16, 127>;
```

---

## 8. Checklist

**Capacity**
- [ ] Total tile footprint (staging + operands + accumulator × buffer count) fits on-chip.
- [ ] For double buffering, staging tile count × 2 before checking.

**Alignment**
- [ ] Let `static_assert` in `pto::Tile` catch violations at compile time.
- [ ] Use `TileLeft` / `TileRight` / `TileAcc` aliases for correct fractal layout.

**Data reuse**
- [ ] `TileAcc` stays on-chip for the full K loop; write back once.
- [ ] Row/column statistics (`TROWMAX`, `TROWSUM`) cached in `Vec` tiles.
- [ ] No unnecessary TSTORE/TLOAD pairs within a kernel.

**GM traffic**
- [ ] Fuse consecutive elementwise ops into one kernel.
- [ ] Row-major GM access; if column access needed, load block then `TTRANS`.
- [ ] `TPREFETCH` used to overlap data movement with compute.

**Synchronization**
- [ ] `Event<SrcOp, DstOp>` for fine-grained ordering; no global `TSYNC` in the steady-state loop.
- [ ] Each consumer waits only on its specific producer event.

---

## References

- [Abstract Machine Model](../machine/abstract-machine.md)
- [Tile Programming Model](Tile.md)
- [GlobalTensor Programming Model](GlobalTensor.md)
- [Events and Synchronization](Event.md)
- [PTO Optimization Guide](opt.md)
- [Pipeline and Parallel Execution](pipeline-parallel.md)
- [GEMM Tutorial](tutorials/gemm.md)
- [GEMM Performance Kernel](../../kernels/manual/a2a3/gemm_performance/README.md)
- [Flash Attention Kernel](../../kernels/manual/common/flash_atten/README.md)
