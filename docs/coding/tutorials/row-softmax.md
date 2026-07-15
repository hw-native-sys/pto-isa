# Tutorial: Row Softmax (Building Block)

Row-softmax is a standard pattern used in attention kernels. The PTO tile-level decomposition is:

1. `row_max = TROWMAX(x)` → `[M, 1]`
2. `x = x - expand(row_max)` (`TROWEXPAND` + `TSUB`)
3. `x = exp(x)` (`TEXP`)
4. `row_sum = TROWSUM(x)` → `[M, 1]`
5. `x = x / expand(row_sum)` (`TROWEXPAND` + `TDIV`)

## Single-tile example

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T, int M, int N>
AICORE void RowSoftmaxOneTile(__gm__ T* out, __gm__ T* in) {
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

## Notes for real kernels

- If `N` is large, you usually tile along columns and combine partial reductions.
- For numerical stability, the “subtract max” step is essential.
- The valid region matters for edge tiles; interpret semantics using `docs/isa/conventions.md`.
