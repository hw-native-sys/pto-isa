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

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式（概念性）：

```text
%dst = tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

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
- `sizeof(TileDataDst::DType) == sizeof(TileDataSrc::DType)`且元素大小必须是 `1`, `2`,或`4` 字节.
- `TFILLPAD`: `TileDataDst::Rows/Cols` 必须匹配 `TileDataSrc::Rows/Cols`.
- `TFILLPAD_EXPAND`: `TileDataDst::Rows >= TileDataSrc::Rows`且`TileDataDst::Cols >= TileDataSrc::Cols`.
- `TFILLPAD(TileData &dst, TileData &src)`:`if TileData::TileType is Mat, layout 仅 support (!TileData::isRowMajor && TileData::Slayout::RowMajor),且PadVal 仅 support PadValue::Zero`

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

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

