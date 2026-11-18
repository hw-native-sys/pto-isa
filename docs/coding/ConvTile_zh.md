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

## 卷积配置字段

`ConvTile` 对象还携带一组运行时配置字段，供卷积相关指令读取。字段默认值通常对应
无 padding、无 dilation、stride 为 1、无转置的普通卷积语义。

| 字段 | 设置接口 | 主要用途 |
| --- | --- | --- |
| `fmapH` / `fmapW` | `SetFmapH()` / `SetFmapW()` | FMATRIX 源图形状 |
| `padList` | `SetPadListArray()` 或逐项 `SetPadList()` | 左、右、上、下 padding |
| `filterH` / `filterW` | `SetFilterH()` / `SetFilterW()` | 卷积核空间尺寸 |
| `dilationH` / `dilationW` | `SetDilationH()` / `SetDilationW()` | 卷积核空洞率 |
| `strideH` / `strideW` | `SetStrideH()` / `SetStrideW()` | IMG2COL 卷积 stride；A5 的 `NCHW -> NC1HWC0`、`NCDHW -> NDC1HWC0` `TLOAD` 路径还使用它表达 dx 输入扩张 |
| `transpose` | `SetTranspose()` | IMG2COL ConvTile 转换路径；A5 `TEXTRACT` / `TMOV` 的 ConvTile 路径也读取该字段，A5 `TIMG2COL` 的 `Right` 路径忽略 |
| `channelSize` | `SetChannelSize()` | IMG2COL 通道配置 |
| `padValue` | `SetPadValue()` | 自动模式下的 padding 值 |
| `repeatStride` / `repeatTime` / `repeatMode` | 对应 `Set*` 接口 | IMG2COL repeat 配置 |
| `dstStride` / `dstMposition` | 对应 `Set*` 接口 | A5 IMG2COL 目标摆放配置 |

这些字段不会改变 `ConvTileShape` 声明的物理形状。需要扩张、padding 或转置时，
调用方仍应声明能够覆盖实际物理存储的 shape 和 buffer size。

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

### A5 反向卷积 dx 输入扩张示例

下面的函数展示 `NCHW` 紧凑输入加载到扩张后的 `NC1HWC0` ConvTile。源 H/W 为 7，
stride 为 2 时，目的 H/W 应为 13：

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

constexpr uint32_t H = 7, W = 7;
constexpr uint32_t strideH = 2, strideW = 2;
constexpr uint32_t expandedH = (H - 1) * strideH + 1;
constexpr uint32_t expandedW = (W - 1) * strideW + 1;

using DstTile = ConvTile<TileType::Mat, bfloat16_t, 1 * 2 * expandedH * expandedW * 16,
                         Layout::NC1HWC0, ConvTileShape<1, 2, expandedH, expandedW, 16>>;
using SrcTensor = GlobalTensor<
    bfloat16_t, Shape<1, 1, 32, H, W>,
    Stride<1 * 32 * H * W, 32 * H * W, H * W, W, 1>, Layout::NCHW>;

void example(__gm__ bfloat16_t* dyInput) {
  DstTile dy;
  SrcTensor src(dyInput);
  TASSIGN(dy, 0x0);

  // 如需空洞为 0，先初始化 dy 所在存储。
  dy.SetStrideH(strideH);
  dy.SetStrideW(strideW);
  TLOAD(dy, src);
}
```
