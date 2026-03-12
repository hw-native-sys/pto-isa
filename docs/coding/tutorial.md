# PTO ISA Quickstart (C++ Tile Intrinsics)

This quickstart is for operator/kernel developers who want to get a first PTO kernel running quickly and understand the core mental model.

It is **not** a full instruction encyclopedia. For detailed instruction semantics and constraints, see `docs/isa/README.md`.

## 0. What you will learn

After reading this document, you should be able to:

1. Recognize the key concepts in PTO code: `GlobalTensor`, `Tile`, `TileType::Vec`, events, and `TSYNC`.
2. Write a simple **PTO-Auto** style kernel: `TLOAD → compute → TSTORE`.
3. Write a **PTO-Manual** style kernel: explicit tile buffer binding (`TASSIGN`) and explicit ordering (events/flags).
4. Understand the typical shape of “bigger” kernels like row-softmax and GEMM at a high level.

## 1. Where PTO code lives (what you are writing)

In this repository, you write kernels in **C++ + PTO intrinsics**. A minimal kernel looks like:

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T>
__global__ AICORE void MyKernel(__gm__ T* out, __gm__ T* in0, __gm__ T* in1) {
  // Use GlobalTensor + Tile + TLOAD/T* ops + TSTORE here.
}
```

Terminology:

- `__gm__`: global memory pointer (GM).
- `AICORE`: runs on a single “core” on the device backend (CPU simulation defines it as a normal function annotation).
- `GlobalTensor`: a *view* of GM data with shape/stride/layout metadata (see `docs/coding/GlobalTensor.md`).
- `Tile`: an on-chip tile object, conceptually a 2-D buffer in tile storage (see `docs/coding/Tile.md`).

## 2. One-page cheat sheet (core concepts)

### 2.1 GlobalTensor: “the big tensor in GM”

`GlobalTensor` is the operand type for memory instructions such as `TLOAD`/`TSTORE`.

Recommended 2-D “syntax sugar” (shape + stride helpers):

```cpp
template <typename T, int rows, int cols>
using Shape2D = TileShape2D<T, rows, cols, Layout::ND>;

template <typename T, int rows, int cols>
using Stride2D = BaseShape2D<T, rows, cols, Layout::ND>;

template <typename T, int rows, int cols>
using GT2D = GlobalTensor<T, Shape2D<T, rows, cols>, Stride2D<T, rows, cols>, Layout::ND>;
```

Mental model: `GlobalTensor = “interpret this GM pointer as a (rows × cols) matrix view”`.

### 2.2 Tile: “the small 2-D block you compute on”

A Tile is the core compute unit. A typical vector tile looks like:

```cpp
template <typename T, int rows, int cols>
using VecTile = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor>;
```

Common tile locations:

- `TileType::Vec`: elementwise / reduction style operations.
- `TileType::Mat`: general matrix tiles for movement/transform paths.
- `TileType::Left`, `TileType::Right`, `TileType::Acc`: matmul tiles.

### 2.3 Two styles: PTO-Auto vs PTO-Manual

PTO-Auto (high level):

- You describe the dataflow: `TLOAD → compute → TSTORE`.
- Tile buffer management and some synchronization can be handled by the compiler/runtime.
- In the API model, `TASSIGN(tile, addr)` is a no-op when `__PTO_AUTO__` is enabled (see `docs/isa/TASSIGN.md`).

PTO-Manual (expert mode):

- You bind explicit tile buffer addresses with `TASSIGN`.
- You express ordering explicitly (events or low-level flags).
- You can build double-buffer pipelines and overlap load/compute/store.

## 3. Your first kernel: vector add (PTO-Auto style)

Goal: `out = in0 + in1` for a single tile.

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T, int kRows, int kCols>
AICORE void VecAddAutoOneTile(__gm__ T* out, __gm__ T* in0, __gm__ T* in1) {
  using GT = GT2D<T, kRows, kCols>;
  using TileT = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

  GT g0(in0), g1(in1), gout(out);
  TileT t0(kRows, kCols), t1(kRows, kCols), tout(kRows, kCols);

  TLOAD(t0, g0);
  TLOAD(t1, g1);
  TADD(tout, t0, t1);
  TSTORE(gout, tout);
}
```

Why this is “Auto style”:

- No explicit `TASSIGN`.
- No explicit flags/events in the source code.
- The kernel is written as a direct dataflow.

On CPU simulation (`python3 tests/run_cpu.py`), this style is typically enough to validate correctness.

## 4. The same kernel: vector add (PTO-Manual style)

Now we write the same logic but explicitly:

- Bind tile buffers (`TASSIGN`)
- Express ordering (events or flags)

### 4.1 Manual ordering with events (recommended)

This matches the programming model in `docs/coding/Event.md` (device-only `Event` types).

```cpp
template <typename T, int kRows, int kCols>
__global__ AICORE void VecAddManual(__gm__ T* out, __gm__ T* in0, __gm__ T* in1) {
  using GT = GT2D<T, kRows, kCols>;
  using TileT = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

  GT g0(in0), g1(in1), gout(out);
  TileT t0(kRows, kCols), t1(kRows, kCols), tout(kRows, kCols);

#ifndef __PTO_AUTO__
  constexpr uint32_t kT0Addr = 0x0000;
  constexpr uint32_t kT1Addr = 0x4000;
  constexpr uint32_t kOutAddr = 0x8000;
  TASSIGN(t0, kT0Addr);
  TASSIGN(t1, kT1Addr);
  TASSIGN(tout, kOutAddr);
#endif

#ifdef __CCE_AICORE__
  Event<Op::TLOAD, Op::TADD> e_load_to_add;
  Event<Op::TADD, Op::TSTORE_VEC> e_add_to_store;

  TLOAD(t0, g0);
  e_load_to_add = TLOAD(t1, g1);
  e_add_to_store = TADD(tout, t0, t1, e_load_to_add);
  TSTORE(gout, tout, e_add_to_store);
#else
  TLOAD(t0, g0);
  TLOAD(t1, g1);
  TADD(tout, t0, t1);
  TSTORE(gout, tout);
#endif
}
```

### 4.2 Manual ordering with low-level flags (legacy style)

Some existing device kernels use `set_flag`/`wait_flag` directly. This is more hardware-coupled than events, but useful to recognize when reading older kernels.

On CPU simulation, these are stubs (no-ops).

See also: `tests/cpu/st/testcase/tadd/tadd_kernel.cpp`.

## 5. A slightly bigger pattern: row-wise softmax (Auto style)

Row-softmax is a common pattern behind attention kernels. For a tile `X` shaped `[M, N]`:

1. `row_max = TROWMAX(X)` → `[M, 1]`
2. `X = X - expand(row_max)` (`TROWEXPAND` + `TSUB`)
3. `X = exp(X)` (`TEXP`)
4. `row_sum = TROWSUM(X)` → `[M, 1]`
5. `X = X / expand(row_sum)` (`TROWEXPAND` + `TDIV`)

Example (single tile):

```cpp
template <typename T, int M, int N>
AICORE void RowSoftmaxAutoOneTile(__gm__ T* out, __gm__ T* in) {
  using GT = GT2D<T, M, N>;
  using XTile = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
  using Col1 = Tile<TileType::Vec, T, M, 1, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

  GT gin(in), gout(out);
  XTile x(M, N), tmp(M, N);
  Col1 row_max(M, 1), row_sum(M, 1);

  TLOAD(x, gin);

  TROWMAX(row_max, x);
  TROWEXPAND(tmp, row_max);
  TSUB(x, x, tmp);

  TEXP(x, x);

  TROWSUM(row_sum, x);
  TROWEXPAND(tmp, row_sum);
  TDIV(x, x, tmp);

  TSTORE(gout, x);
}
```

For tiled softmax over a larger tensor, the same structure applies; you iterate over tiles and set valid regions on edge tiles when needed.

## 6. GEMM skeleton: TMATMUL + TMOV (high-level shape)

A minimal GEMM at the tile level typically looks like:

1. `TLOAD` A and B tiles (often into `Mat` tiles).
2. `TMOV` into `Left/Right` tiles (to satisfy boxed/fractal layout requirements).
3. `TMATMUL` into an accumulator tile.
4. Convert/move/store the result (depending on backend and destination layout).

Skeleton (single-tile shape, illustrative):

```cpp
template <typename A, typename B, typename Acc, int TM, int TK, int TN>
__global__ AICORE void GemmAutoOneTile(__gm__ A* a, __gm__ B* b, __gm__ Acc* c) {
  using GA = GT2D<A, TM, TK>;
  using GB = GT2D<B, TK, TN>;
  using GC = GT2D<Acc, TM, TN>;

  GA gA(a);
  GB gB(b);
  GC gC(c);

  Tile<TileType::Mat, A, TM, TK, BLayout::RowMajor> a_mat;
  Tile<TileType::Mat, B, TK, TN, BLayout::RowMajor> b_mat;

  TileLeft<A, TM, TK> a_l;
  TileRight<B, TK, TN> b_r;
  TileAcc<Acc, TM, TN> acc;

  TLOAD(a_mat, gA);
  TLOAD(b_mat, gB);
  TMOV(a_l, a_mat);
  TMOV(b_r, b_mat);

  TMATMUL(acc, a_l, b_r);

  // Result writeback can be backend-specific; see the GEMM demos/kernels in this repo.
  // For example, some flows move `acc` to a vec/mat tile before `TSTORE`.
}
```

To build real GEMM/attention kernels you add:

- Tiling of M/K/N across blocks and loops.
- `TMATMUL_ACC` to accumulate across K tiles.
- `TEXTRACT`/`TRESHAPE`/`TTRANS` as needed for slicing/layout.
- Events for overlapping load/compute/store.

## 7. Next steps

- Learn the models precisely:
  - `docs/coding/Tile.md`
  - `docs/coding/GlobalTensor.md`
  - `docs/coding/Event.md`
- Debugging and assertion lookup:
  - `docs/coding/debug.md`
- Browse more examples (expanded walkthroughs):
  - `docs/coding/tutorials/README.md`
- Run CPU simulation first:
  - `python3 tests/run_cpu.py --verbose`
- Use the instruction reference as needed:
  - `docs/isa/README.md`

## Appendix: compiling PTO-Auto with Bisheng (Ascend CANN)

PTO-Auto has two parts:

1. **Library-level auto semantics**: compile with `-D__PTO_AUTO__` so Tiles use compiler-managed storage and `TASSIGN(tile, addr)` becomes a no-op (see `docs/isa/TASSIGN.md`).
2. **Compiler pipeline**: enable the PTO lowering/bufferization passes in the Bisheng CCE toolchain.

### Finding the correct “enable PTO passes” flag (CANN toolchain)

The exact flag name is toolchain-version dependent. On some releases it is a driver flag (e.g. `--cce-enable-pto-passes`); on others it is exposed as an LLVM option (passed via `-mllvm`).

Use your installed Bisheng to discover the supported spelling:

```bash
bisheng --help | rg -n "pto|PTO" || true
bisheng -mllvm --help | rg -n "pto|PTO" || true
```

### Example (device compilation)

This compiles a single CCE kernel source into an object file. Adjust:

- `--cce-aicore-arch=...` for your SoC (examples in this repo use `dav-c220-vec`, `dav-c310-vec`, etc.).
- `-DMEMORY_BASE` vs `-DREGISTER_BASE` for the target backend used in this repo.
- the “enable PTO passes” flag spelling based on your Bisheng help output.

```bash
source /usr/local/Ascend/cann/set_env.sh

bisheng -c -xcce -O2 --cce-aicore-only \
  --cce-aicore-arch=dav-c310-vec \
  -std=c++17 \
  -I"$ASCEND_HOME_PATH/include" -I./include \
  -DREGISTER_BASE -D__PTO_AUTO__ \
  -mllvm -cce-aicore-stack-size=0x8000 \
  -mllvm -cce-aicore-function-stack-size=0x8000 \
  -mllvm -cce-aicore-record-overflow=true \
  -mllvm -cce-aicore-addr-transform \
  -mllvm -cce-aicore-dcci-insert-for-scalar=false \
  <ENABLE_PTO_PASSES_FLAG> \
  tadd.cpp -o tadd.o
```

If your toolchain uses a driver-style flag, `<ENABLE_PTO_PASSES_FLAG>` might look like:

- `--cce-enable-pto-passes`

If your toolchain exposes it as an LLVM option, it might look like:

- `-mllvm -cce-enable-pto-passes`

When in doubt, prefer “what `bisheng --help` and `bisheng -mllvm --help` show” over hard-coding a particular spelling in docs.
