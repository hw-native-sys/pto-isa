# pto.tload

`pto.tload` 属于[内存与数据搬运](../../memory-and-data-movement_zh.md)指令集。

## 概述

把 `GlobalTensor`（GM 视图）中的数据装入 tile。它是 tile 数据从 GM 进入本地缓冲的基本入口。

## 机制

若用带基址偏移的二维视角表示，可将其理解为：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{r_0 + i,\; c_0 + j} $$

其中 `r0 / c0` 由 `GlobalTensor` 当前视图和 tile 的装载位置共同决定。真正的搬运范围由目标 tile 的 valid region 决定，而不是物理 tile 矩形。

## 语法

同步形式：

```text
%t0 = tload %sv[%c0, %c0] : (!pto.memref<...>, index, index) -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
!pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

### AS Level 2（DPS）

```text
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);
```

## 输入

- `src`：源 `GlobalTensor`
- `dst`：目标 tile

## 预期输出

- `dst`：按 tile 布局和 `GlobalTensor` stride 解释后的装载结果

## 副作用

这条指令会从 GM 读取，并写入 tile 本地存储。

## 约束

- `sizeof(TileData::DType)` 必须与 `sizeof(GlobalData::DType)` 一致
- 搬运范围由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定
- `src.GetShape(dim)` 与 `dst.GetValidRow()/GetValidCol()` 在运行时都必须大于 0

## Target-Profile 限制

### A2A3

- 目标 tile 位置必须为 `TileType::Vec` 或 `TileType::Mat`
- 支持类型：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、`half`、`bfloat16_t`、`float`
- `Vec` 路径只支持布局完全匹配的装载：`ND→ND`、`DN→DN`、`NZ→NZ`
- `Mat` 路径额外支持 `ND→NZ`、`DN→ZN`
- 对 `ND→NZ` / `DN→ZN`，要求 `GlobalData::staticShape[0..2] == 1` 且 `TileData::SFractalSize == 512`
- 对 `int64_t/uint64_t`，仅支持 `ND→ND` 或 `DN→DN`

### A5

- `sizeof(TileData::DType)` 必须为 `1/2/4/8` 字节，并与 `GlobalData` 一致
- 对 `int64_t/uint64_t`，`TileData::PadVal` 必须为 `PadValue::Null` 或 `PadValue::Zero`
- `Vec` 路径只支持：
  - ND + row-major + `SLayout::NoneBox`
  - DN + col-major + `SLayout::NoneBox`
  - NZ + `SLayout::RowMajor`
- 编译期已知 shape 的 row-major ND→ND 路径还要求：
  - `TileData::ValidCol == GlobalData::staticShape[4]`
  - `TileData::ValidRow == product(GlobalData::staticShape[0..3])`
- `Mat` 路径额外受 `TLoadCubeCheck` 与 MX 格式规则约束

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能

当前仓内没有把 `tload` 单列成完整公开周期表，但在 A2A3 的带宽模型里，GM → Vec Buffer 取 `128 B/cycle`，GM → Mat 取 `256 B/cycle`。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T>
void example_auto(__gm__ T* in) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gin(in);
  TileT t;
  TLOAD(t, gin);
}
```

## 相关页面

- 指令集总览：[内存与数据搬运](../../memory-and-data-movement_zh.md)
- [GlobalTensor 与数据搬运](../../../programming-model/globaltensor-and-data-movement_zh.md)
