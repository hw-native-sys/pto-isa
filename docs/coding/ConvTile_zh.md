# ConvTile 编程模型

PTO Lib 程序可基于 **ConvTile** 编写卷积相关算子。`ConvTile` 是固定容量的 2D 到 6D 缓冲对象，也是 PTO 卷积类操作中的主要计算单元和数据搬运单元。

从概念上说，`ConvTile` 驻留在**片上 Tile 存储**中（类似寄存器文件或片上 SRAM），并通过 `TLOAD` / `TSTORE` 与全局内存（GM）之间搬运数据。

本文档说明 `include/pto/common/pto_tile.hpp` 中的 C++ `ConvTile` 类型及其布局和形状约束。

## ConvTile 表示什么

一个 `ConvTile` 主要由以下几类属性定义：

- **位置（Location）**：该 Tile 所属的逻辑存储类别（如矩阵/立方寄存器等）。
- **元素类型（Element type）**：标量元素类型（如 `float`、`half`、`int8_t` 等）。
- **缓冲区大小（Buffer size）**：`ConvTile` 的静态缓冲容量。
- **布局（Layout）**：如 `NCHW`、`NHWC`、`NC1HWC0` 等，用于指导 lowering 和目标相关优化路径。
- **形状（Shape）**：`pto::ConvTileShape<...>`，支持最多 6 个维度。

## `pto::ConvTile` 类型

`ConvTile` 通过 C++ 模板类型声明：

```cpp
pto::ConvTile<
  pto::TileType Loc_,
  Element_,
  BufferSize_,
  pto::Layout_ layout,
  pto::ConvTileShape Shape_
>;
```

### 位置（`TileType`）

`TileType` 表示 Tile 的逻辑/物理存储类别，同时参与重载选择和编译期检查。

常见位置包括：

- `TileType::Vec`：向量 Tile 存储（UB / 向量流水线）。
- `TileType::Mat`：通用矩阵 Tile 存储（矩阵 L1）。

每条指令允许使用哪些位置，应以 `docs/isa/` 下对应指令文档为准。

### 容量（`BufferSize_`）

`BufferSize_` 定义了 Tile 对象的**静态容量**。多数指令要求 Tile 具备静态形状，以便在编译期进行特化和优化。

### 布局（`pto::Layout`）

`ConvTile` 包含一个布局枚举，如：

- `NCHW`
- `NHWC`
- `NC1HWC0`
- `FRACTAL_Z`
- `FRACTAL_Z_S16S8`

布局信息会影响后端实现、lowering 路径以及特定目标上的快速路径选择。

### 形状（`pto::ConvTileShape`）

`pto::ConvTileShape<...Shapes>` 支持 1 到 6 个整型模板参数。每个维度既可以是编译期常量，也可以是 `pto::DYNAMIC`（即 `-1`）。

- 静态维度保存在类型信息中，可通过 `ConvTileShape::staticShape[dim]` 获取。
- 动态维度保存在运行时对象 `ConvTileShape::shape[dim]` 中，并由 `ConvTileShape(...)` 构造函数赋值。

构造函数会通过 `static_assert` 检查“运行时传入参数个数是否与动态维度数量一致”，因此若构造参数不匹配，会在编译期报错。

## 地址绑定（`TASSIGN`）

在手动放置流程中，`TASSIGN(tile, addr)` 用于把一个 `ConvTile` 对象绑定到实现定义的地址。

在自动模式中，`TASSIGN(tile, addr)` 可能根据构建配置被处理为 no-op。

具体约束请参考 `docs/isa/TASSIGN.md`。

## 最小示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example(__gm__ half* in, __gm__ half* out) {
  using TileT = ConvTile<TileType::Mat, half, 4096, Layout::NC1HWC0, pto::ConvTileShape<1, 1, 16, 16, 16>>;
  using GShape = Shape<1, 1, 16, 16, 16>;
  using GStride = Stride<1 * 16* 16* 16, 16* 16* 16, 16 * 16, 16, 1>;
  using GT = GlobalTensor<half, GShape, GStride, Layout::NC1HWC0>;
  GT gin(in);

  TileT tile5d;
  TASSIGN(tile5d, 0x0);

  TLOAD(tile5d, gin);
}
```
