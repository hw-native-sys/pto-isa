# 教程：按行 Softmax（基础组件）

按行 softmax 是 attention 内核中的标准模式。PTO 的 tile 级分解为：

1. `row_max = TROWMAX(x)` → `[M, 1]`
2. `x = x - expand(row_max)`（`TROWEXPAND` + `TSUB`）
3. `x = exp(x)`（`TEXP`）
4. `row_sum = TROWSUM(x)` → `[M, 1]`
5. `x = x / expand(row_sum)`（`TROWEXPAND` + `TDIV`）

## 单 Tile 示例

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

## 真实内核注意事项

- 当 `N` 很大时，通常沿列做 tiling，并组合部分归约结果。
- 数值稳定性上，“减去最大值”步骤非常关键。
- 边界 tile 的有效区域很重要；语义解释以 `docs/isa/conventions_zh.md` 为准。
