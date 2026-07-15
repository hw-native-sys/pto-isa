# 教程：向量加法（Tiled）+ Ping-Pong（Manual）

本文在“向量加法”示例基础上扩展两点：

1. 对更大 2D 张量做分块（tiled）计算。
2. 在 manual 模式下给出 ping-pong（双缓冲）概念结构，用于重叠 `TLOAD`、计算与 `TSTORE`。

关于 Tile、有效区域与 events 的背景知识参见：

- `docs/coding/Tile_zh.md`
- `docs/coding/GlobalTensor_zh.md`
- `docs/coding/Event_zh.md`

## 1. 分块向量加（Auto 风格结构）

设 2D 矩阵 `A[GRows, GCols]`，tile 形状为 `(TRows, TCols)`，并假设 `GCols % TCols == 0` 且 `GRows % TRows == 0`。

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T, int GRows, int GCols, int TRows, int TCols>
__global__ AICORE void VecAddTiledAuto(__gm__ T* out, __gm__ T* a, __gm__ T* b) {
  constexpr int tiles_per_row = GCols / TCols;
  const int tile_row = static_cast<int>(block_idx) / tiles_per_row;
  const int tile_col = static_cast<int>(block_idx) % tiles_per_row;
  const int base = tile_row * (GCols * TRows) + tile_col * TCols;

  using GT = GT2D<T, TRows, TCols>;
  using TileT = Tile<TileType::Vec, T, TRows, TCols, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

  GT ga(a + base), gb(b + base), gout(out + base);
  TileT ta(TRows, TCols), tb(TRows, TCols), tc(TRows, TCols);

  TLOAD(ta, ga);
  TLOAD(tb, gb);
  TADD(tc, ta, tb);
  TSTORE(gout, tc);
}
```

## 2. 边界 tile（动态有效区域）

如果 `GRows` 或 `GCols` 不能被 tile 形状整除，需要对边界 tile 使用动态有效区域：

- 对每个 tile 计算 `valid_rows` 与 `valid_cols`。
- 构造 tile 时传入运行时有效区域。

“有效区域外元素如何解释”的具体行为依赖指令；请按 `docs/isa/conventions_zh.md` 解释语义。

## 3. Manual Ping-Pong 结构（概念）

手动性能调优常用两套 tile buffer：

- buffer 0 在计算/写回时，buffer 1 同时加载；随后交换。

下面是**概念结构**，具体地址选择与 event wiring 与平台相关。

```cpp
template <typename T, int TRows, int TCols, int NumTiles>
__global__ AICORE void VecAddPingPong(__gm__ T* out, __gm__ T* a, __gm__ T* b) {
  using GT = GT2D<T, TRows, TCols>;
  using TileT = Tile<TileType::Vec, T, TRows, TCols, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

  TileT ta[2] = {TileT(TRows, TCols), TileT(TRows, TCols)};
  TileT tb[2] = {TileT(TRows, TCols), TileT(TRows, TCols)};
  TileT tc[2] = {TileT(TRows, TCols), TileT(TRows, TCols)};

#ifndef __PTO_AUTO__
  constexpr uint32_t kBufStride = 0x8000;
  constexpr uint32_t kA0 = 0x0000, kB0 = 0x2000, kC0 = 0x4000;
  for (int p = 0; p < 2; ++p) {
    TASSIGN(ta[p], kA0 + p * kBufStride);
    TASSIGN(tb[p], kB0 + p * kBufStride);
    TASSIGN(tc[p], kC0 + p * kBufStride);
  }
#endif

  for (int i = 0; i < NumTiles; ++i) {
    const int p = i & 1;
    GT ga(a + i * TRows * TCols), gb(b + i * TRows * TCols), gout(out + i * TRows * TCols);

#ifdef __CCE_AICORE__
    Event<Op::TLOAD, Op::TADD> e0;
    Event<Op::TADD, Op::TSTORE_VEC> e1;

    TLOAD(ta[p], ga);
    e0 = TLOAD(tb[p], gb);
    e1 = TADD(tc[p], ta[p], tb[p], e0);
    TSTORE(gout, tc[p], e1);
#else
    TLOAD(ta[p], ga);
    TLOAD(tb[p], gb);
    TADD(tc[p], ta[p], tb[p]);
    TSTORE(gout, tc[p]);
#endif
  }
}
```

该版本在源码形式上仍是顺序结构；真实“重叠”来自设备流水线并行能力、正确的 event 顺序，以及 buffer 的复用策略。
