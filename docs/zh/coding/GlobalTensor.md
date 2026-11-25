# GlobalTensor 编程模型

`pto::GlobalTensor` 用于描述存放在全局内存（GM）中的张量。它是一个轻量包装，包含：

- 一个 `__gm__` 指针，以及
- 一个 **5 维**的 shape 与 stride 描述

这些元数据会被 `TLOAD`、`TSTORE`、`MGATHER`、`MSCATTER` 等内存类指令消费。

除非另有说明，本文档中的标识符均对应 `include/pto/common/pto_tile.hpp` 中的定义。

## GlobalTensor 类型

```cpp
template <typename Element_, typename Shape_, typename Stride_, pto::Layout Layout_ = pto::Layout::ND>
struct GlobalTensor;
```

- `Element_`：GM 中存放的标量元素类型。
- `Shape_`：`pto::Shape<...>`（最多 5 维）。
- `Stride_`：`pto::Stride<...>`（最多 5 个 stride，以**元素数**计）。
- `Layout_`：布局 *hint*（`ND`、`DN`、`NZ` 等），用于指导 lowering 与目标相关的 fast path。

GM 指针类型是 `GlobalTensor::DType`，即 `__gm__ Element_`。

## Shape 与 Stride（5 维）

PTO 将全局内存张量统一建模为 5 维对象。多数 2 维用法会将高维设为 `1`，并用最后两维表示 `(rows, cols)`。

### `pto::Shape`

`pto::Shape<N1, N2, N3, N4, N5>` 保存 5 个整数。每个模板参数要么是编译期常量，要么是 `pto::DYNAMIC`（`-1`）。

- 静态维度通过 `Shape::staticShape[dim]` 体现在类型中。
- 动态维度存放在运行时的 `Shape::shape[dim]`，由 `Shape(...)` 构造函数填充。

构造函数通过 `static_assert` 强制“运行时参数个数必须等于动态维度个数”，不匹配将导致编译期失败。

### `pto::Stride`

`pto::Stride<S1, S2, S3, S4, S5>` 与 `Shape` 的模式一致，但保存 stride：

- stride 以**元素**计，而不是字节。
- stride 描述某一维索引加 1 时，指针应前进多少个元素。

### GlobalTensor 构造与访问

`GlobalTensor` 存放指针以及动态维度的 shape/stride：

```cpp
using GT = pto::GlobalTensor<float, pto::Shape<1,1,1,-1,-1>, pto::Stride<1,1,1,-1,1>, pto::Layout::ND>;
GT t(ptr, /*shape=*/{rows, cols}, /*stride=*/{ld});

auto* p = t.data();
int cols = t.GetShape(pto::GlobalTensorDim::DIM_4);
int ld   = t.GetStride(pto::GlobalTensorDim::DIM_3);
```

对完全静态的张量，还可查询编译期值：

```cpp
constexpr int cols = GT::GetShape<pto::GlobalTensorDim::DIM_4>();
```

## 布局 hint（`pto::Layout`）

`GlobalTensor` 包含一个布局枚举（`ND`、`DN`、`NZ`、`SCALE`、`MX_A_ZZ`、`MX_A_ND`、`MX_ADN`、`MX_B_NN`、`MX_B_ND`、`MX_B_DN` 等）。它是一个 *hint*，可用于启用目标相关 fast path。

其原因在于 GlobalTensor 与 Tile 的布局并不一一对应：

- Tile 布局（`BLayout`/`SLayout`）是二维概念（外层与可选内层盒化）。
- GlobalTensor 是 5 维对象，用单一“外层+内层”对无法覆盖全部场景。

因此 `Layout` 采用标签化方式表达常见存储模式（例如 `ND` vs `DN` 的 minor 2D 排列，以及 `NZ` 等立方友好打包方式）。

## 示例常用的 2D helper

2 维张量经常使用两类 helper：

- `pto::TileShape2D<T, rows, cols, layout>`：生成 `pto::Shape<1,1,1,rows,cols>`（当 `layout == Layout::NZ` 时会生成 NZ 特化 shape）。
- `pto::BaseShape2D<T, rows, cols, layout>`：生成适用于 2D 视图的 `pto::Stride<...>`（当 `layout == Layout::NZ` 时会生成 NZ 特化 stride）。
  - `pto::TileShape2D`、`pto::BaseShape2D` 也支持 `MX_A_ZZ`、`MX_A_ND`、`MX_ADN`、`MX_B_NN`、`MX_B_ND`、`MX_B_DN` 等布局标签。

尽管名称中含 “Shape”，`BaseShape2D` 实际上是 **stride** helper（它继承自 `pto::Stride`）。

## 地址绑定（`TASSIGN`）

`TASSIGN(globalTensor, ptr)` 会设置 `GlobalTensor` 的 GM 指针。指针类型必须与 `GlobalTensor::DType` 匹配（由 `TASSIGN_IMPL` 内部的 `static_assert` 强制）。

## 最小示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example(__gm__ float* in, __gm__ float* out) {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<float, 16, 16, Layout::ND>;
  using GT = GlobalTensor<float, GShape, GStride, Layout::ND>;

  GT gin(in);
  GT gout(out);

  TileT t;
  TLOAD(t, gin);
  TSTORE(gout, t);
}
```
