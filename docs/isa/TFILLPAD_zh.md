# TFILLPAD

## 指令示意图

![TFILLPAD tile operation](../figures/isa/TFILLPAD.svg)

## 简介

复制 Tile 并在有效区域外使用编译时填充值进行填充。

将源 Tile 复制到目标 Tile 中，并使用由 `TileDataDst::PadVal` 选定的编译时填充值（如 `PadValue::Min`/`PadValue::Max`）填充剩余（填充）元素。

此操作常用于在运行时有效区域之外确定性地物化特定值，使后续操作能够在完整的静态 Tile 形状上运算。

## 数学语义

设 `VR = src.GetValidRow()`，`VC = src.GetValidCol()`。对每个目标元素 `(i, j)`：

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src}_{i,j} & \text{if } i < VR \text{ and } j < VC \\
\mathrm{pad}       & \text{otherwise}
\end{cases}
$$

`pad` 由 `TileDataDst::PadVal` 和元素类型决定（例如，当浮点类型支持时使用 `+inf/-inf`，否则使用 `std::numeric_limits<T>::max()/min()`）。

## C++ 内建接口

在 `include/pto/common/pto_instr_impl.hpp` 引入的后端头文件中实现：

```cpp
template <typename TileData, PadValue PadVal = PadValue::Zero, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(TileData &dst, TileData &src, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

## 约束

- `TileDataDst::PadVal != PadValue::Null`.
- `sizeof(TileDataDst::DType) == sizeof(TileDataSrc::DType)`且元素大小必须是 `1`, `2`,或`4`字节.
- `TFILLPAD`: `TileDataDst::Rows/Cols` 必须匹配 `TileDataSrc::Rows/Cols`.
- `TFILLPAD_EXPAND`: `TileDataDst::Rows >= TileDataSrc::Rows`且`TileDataDst::Cols >= TileDataSrc::Cols`.
- `TFILLPAD(TileData &dst, TileData &src)`:`if TileData::TileType is Mat, layout 仅 support (!TileData::isRowMajor && TileData::SLayout::RowMajor),且PadVal 仅 support PadValue::Zero 或 PadValue::Null`

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example1() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox, TileConfig::fractalABSize, PadValue::Min>;

  SrcT src;
  DstT dst;
  TFILLPAD(dst, src);
}

void example2() {
  using TileMatData = Tile<TileType::Mat, float, 16, 256, BLayout::ColMajor, 1, 224, SLayout::RowMajor, 512>;

  TileMatData matTile;
  TFILLPAD(matTile, matTile);
}
```
