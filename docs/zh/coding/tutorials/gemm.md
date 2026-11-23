# 教程：GEMM（模式与 Tile 类型）

本文展示使用 PTO tile 内建接口编写 GEMM 时常见的“代码形态”。

本文刻意保持高层。可运行示例可参考仓库中的 CPU demo 与 kernels（例如 `demos/cpu/gemm_demo/` 与 `kernels/`）。

## 1. GEMM 的 Tile 角色

GEMM 通常涉及几类专门的 Tile 角色：

- `TileType::Mat`：通用矩阵 Tile，用于内存搬运与变换。
- `TileLeft<A, TM, TK>`：矩阵乘引擎期望的左操作数 Tile 布局。
- `TileRight<B, TK, TN>`：矩阵乘引擎期望的右操作数 Tile 布局。
- `TileAcc<Acc, TM, TN>`：`TMATMUL`/`TMATMUL_ACC` 使用的累加器 Tile。

这些别名背后的盒化/分形布局要求由 `pto::Tile` 的编译期检查与指令约束共同强制。

## 2. 单 Tile GEMM 骨架

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

## 3. 从骨架到真实 GEMM

要变成真实 GEMM kernel，通常还需要：

- 对 `M`、`N`、`K` 做 tiling：
  - 用 `TEXTRACT` 从更大的 GM 视图切片到 tile 视图。
  - 对 K tile 做循环，并用 `TMATMUL_ACC` 做累加。
- 为重叠做同步：
  - 用 events 对内存与计算流水线建立顺序，
  - 用 ping-pong buffer 安全复用 Tile 存储。
