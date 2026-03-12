# PTO ISA 快速上手（C++ Tile Intrinsics）

本文面向算子/内核开发者：希望尽快跑通第一个 PTO kernel，并建立核心心智模型。

本文**不是**逐条指令百科。逐条指令语义与约束请参见：`docs/isa/README_zh.md`。

## 0. 你将学到什么

阅读完本文后，你应当能够：

1. 识别 PTO 代码中的关键概念：`GlobalTensor`、`Tile`、`TileType::Vec`、events 与 `TSYNC`。
2. 编写一个简单的 **PTO-Auto** 风格 kernel：`TLOAD → compute → TSTORE`。
3. 编写一个 **PTO-Manual** 风格 kernel：显式 Tile 缓冲绑定（`TASSIGN`）与显式顺序（events/flags）。
4. 在高层理解更“大”的 kernel 结构，例如 row-softmax 与 GEMM。

## 1. PTO 代码写在哪里（你在写什么）

在本仓库中，你使用 **C++ + PTO 内建接口**编写 kernel。一个最小 kernel 形态如下：

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T>
__global__ AICORE void MyKernel(__gm__ T* out, __gm__ T* in0, __gm__ T* in1) {
  // Use GlobalTensor + Tile + TLOAD/T* ops + TSTORE here.
}
```

术语说明：

- `__gm__`：全局内存指针（GM）。
- `AICORE`：在设备后端运行于单个“核心”（CPU 仿真会把它当作普通函数注解）。
- `GlobalTensor`：对 GM 数据的*视图*，携带 shape/stride/layout 元数据（参见 `docs/coding/GlobalTensor_zh.md`）。
- `Tile`：片上 Tile 对象，概念上是 Tile 存储中的二维缓冲（参见 `docs/coding/Tile_zh.md`）。

## 2. 一页速查（核心概念）

### 2.1 GlobalTensor：“GM 里的大张量”

`GlobalTensor` 是 `TLOAD`/`TSTORE` 等内存指令的操作数类型。

推荐的 2D “语法糖”（shape + stride helper）：

```cpp
template <typename T, int rows, int cols>
using Shape2D = TileShape2D<T, rows, cols, Layout::ND>;

template <typename T, int rows, int cols>
using Stride2D = BaseShape2D<T, rows, cols, Layout::ND>;

template <typename T, int rows, int cols>
using GT2D = GlobalTensor<T, Shape2D<T, rows, cols>, Stride2D<T, rows, cols>, Layout::ND>;
```

心智模型：`GlobalTensor = “把这个 GM 指针解释成一个 (rows × cols) 的矩阵视图”`。

### 2.2 Tile：“你要在上面算的小二维块”

Tile 是计算的基本单位。一个典型的向量 Tile 形态：

```cpp
template <typename T, int rows, int cols>
using VecTile = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor>;
```

常见 Tile 位置类型：

- `TileType::Vec`：逐元素 / 归约类操作。
- `TileType::Mat`：通用矩阵 Tile（搬运/变换路径）。
- `TileType::Left`、`TileType::Right`、`TileType::Acc`：矩阵乘相关 Tile。

### 2.3 两种风格：PTO-Auto vs PTO-Manual

PTO-Auto（高层）：

- 你描述数据流：`TLOAD → compute → TSTORE`。
- Tile buffer 管理与部分同步可由编译器/运行时处理。
- 在 API 模型中，当启用 `__PTO_AUTO__` 时，`TASSIGN(tile, addr)` 可能是 no-op（参见 `docs/isa/TASSIGN_zh.md`）。

PTO-Manual（专家）：

- 你用 `TASSIGN` 显式绑定 Tile 缓冲地址。
- 你显式表达顺序（events 或低层 flags）。
- 你可以构造双缓冲流水线并重叠 load/compute/store。

## 3. 第一个 kernel：向量加法（PTO-Auto 风格）

目标：对一个 Tile 做 `out = in0 + in1`。

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

它之所以是 “Auto 风格”：

- 源码没有显式 `TASSIGN`。
- 源码没有显式 flags/events。
- kernel 以直观的数据流方式表达。

在 CPU 仿真（`python3 tests/run_cpu.py`）上，Auto 风格通常足以验证正确性。

## 4. 同一个 kernel：向量加法（PTO-Manual 风格）

现在写同样逻辑，但显式做两件事：

- 绑定 Tile buffer（`TASSIGN`）
- 表达顺序（events 或 flags）

### 4.1 使用 events 的手动顺序（推荐）

对应 `docs/coding/Event_zh.md` 中的事件模型（设备侧 `Event` 类型）。

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

### 4.2 使用低层 flags 的手动顺序（遗留风格）

一些已有设备 kernel 直接使用 `set_flag`/`wait_flag`。这比 events 更强绑定硬件，但阅读旧 kernel 时常会遇到。

CPU 仿真下这些通常是 stub（no-op）。

另请参阅：`tests/cpu/st/testcase/tadd/tadd_kernel.cpp`。

## 5. 更大的模式：按行 softmax（Auto 风格）

Row-softmax 是 attention 内核的常见基础模式。对一个形状为 `[M, N]` 的 Tile `X`：

1. `row_max = TROWMAX(X)` → `[M, 1]`
2. `X = X - expand(row_max)`（`TROWEXPAND` + `TSUB`）
3. `X = exp(X)`（`TEXP`）
4. `row_sum = TROWSUM(X)` → `[M, 1]`
5. `X = X / expand(row_sum)`（`TROWEXPAND` + `TDIV`）

单 Tile 示例：

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

对更大张量做 tiled softmax 时结构相同：沿 tile 迭代，并在边界 tile 上设置有效区域即可。

## 6. GEMM 骨架：TMATMUL + TMOV（高层结构）

一个最小的 tile 级 GEMM 通常是：

1. `TLOAD` A/B tile（常加载到 `Mat` tile）。
2. `TMOV` 到 `Left/Right` tile（满足盒化/分形布局约束）。
3. `TMATMUL` 生成累加器 tile。
4. 将结果转换/搬运/写回（依赖后端与目的布局）。

单 tile 形态示意：

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

构建真实 GEMM/attention kernel 通常还会添加：

- M/K/N 的 tiling 与循环。
- 使用 `TMATMUL_ACC` 跨 K tile 做累加。
- 需要时使用 `TEXTRACT`/`TRESHAPE`/`TTRANS` 做切片/布局。
- 使用 events 实现 load/compute/store 重叠与 buffer 复用。

## 7. 下一步

- 精确理解模型：
  - `docs/coding/Tile_zh.md`
  - `docs/coding/GlobalTensor_zh.md`
  - `docs/coding/Event_zh.md`
- 浏览更多示例（更长的 walkthrough）：
  - `docs/coding/tutorials/README_zh.md`
- 优先跑 CPU 仿真验证正确性：
  - `python3 tests/run_cpu.py --verbose`
- 随时查阅逐条指令参考：
  - `docs/isa/README_zh.md`

## 附录：使用 Bisheng（Ascend CANN）编译 PTO-Auto

PTO-Auto 包含两部分：

1. **库侧 Auto 语义**：用 `-D__PTO_AUTO__` 编译，使 Tile 使用编译器管理的存储，同时 `TASSIGN(tile, addr)` 变为 no-op（参见 `docs/isa/TASSIGN_zh.md`）。
2. **编译器 pipeline**：在 Bisheng CCE 工具链中启用 PTO lowering/bufferization passes。

### 如何找到正确的“启用 PTO passes”编译选项

具体 flag 名称取决于工具链版本：有的版本是 driver flag（例如 `--cce-enable-pto-passes`），有的版本暴露为 LLVM 选项（通过 `-mllvm` 传入）。

建议直接查询你安装的 Bisheng 支持哪些选项：

```bash
bisheng --help | rg -n "pto|PTO" || true
bisheng -mllvm --help | rg -n "pto|PTO" || true
```

### 设备侧编译示例

示例将单个 CCE kernel 源码编译为目标文件。需要按实际环境调整：

- `--cce-aicore-arch=...`：对应你的 SoC（本仓库示例常见 `dav-c220-vec`、`dav-c310-vec` 等）。
- `-DMEMORY_BASE` vs `-DREGISTER_BASE`：对应本仓库采用的后端模式。
- `<ENABLE_PTO_PASSES_FLAG>`：基于 Bisheng help 输出选择正确拼写。

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

如果工具链使用 driver-style flag，`<ENABLE_PTO_PASSES_FLAG>` 可能类似：

- `--cce-enable-pto-passes`

如果工具链暴露为 LLVM 选项，可能类似：

- `-mllvm -cce-enable-pto-passes`

如果不确定，优先以 `bisheng --help` 与 `bisheng -mllvm --help` 的输出为准，而不要在文档中硬编码某一版本的拼写。
