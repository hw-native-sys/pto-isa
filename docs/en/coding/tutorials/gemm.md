# Tutorial: GEMM (Patterns and Tile Types)

This tutorial shows the common “shape” of GEMM code when written with PTO tile intrinsics.

It is intentionally high level. For fully working examples, see the CPU demos and kernels in this repository (e.g., `demos/cpu/gemm_demo/` and `kernels/`).

## 1. Tile roles for GEMM

GEMM uses a few specialized tile roles:

- `TileType::Mat`: general matrix tiles for memory movement and transforms.
- `TileLeft<A, TM, TK>`: the left operand tile layout expected by the matmul engine.
- `TileRight<B, TK, TN>`: the right operand tile layout expected by the matmul engine.
- `TileAcc<Acc, TM, TN>`: accumulator tile used by `TMATMUL`/`TMATMUL_ACC`.

The boxed/fractal layout requirements behind these aliases are enforced by `pto::Tile` compile-time checks and by per-instruction constraints.

## 2. Single-tile GEMM skeleton

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename A, typename B, typename Acc, int TM, int TK, int TN>
__global__ AICORE void GemmOneTile(__gm__ A* a, __gm__ B* b, __gm__ Acc* c) {
  using GA = GT2D<A, TM, TK>;
  using GB = GT2D<B, TK, TN>;

  GA gA(a);
  GB gB(b);

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

  // Writeback is typically done via a backend-specific move and store path.
}
```

## 3. From skeleton to real GEMM

To make this into a real GEMM kernel you usually add:

- Tiling over `M`, `N`, and `K`:
  - `TEXTRACT` to slice from a larger GM view into a tile-shaped view.
  - loops over K tiles and `TMATMUL_ACC` for accumulation.
- Synchronization for overlap:
  - events to order memory and compute pipelines,
  - ping-pong buffers to reuse tile storage safely.
