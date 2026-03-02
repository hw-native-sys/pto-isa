# TFILLPAD

## 指令示意图

![TFILLPAD tile operation](../figures/isa/TFILLPAD.svg)

## 简介

复制 Tile 并在有效区域外使用编译时填充值进行填充。

Copy a source tile into a destination tile and fill the remaining (padded) elements with a compile-time pad value
selected by `TileDataDst::PadVal` (e.g., `PadValue::Min`/`PadValue::Max`).

This is commonly used to materialize deterministic values outside the runtime valid region so that subsequent ops can
operate on a full static tile shape.

## 数学语义

Let `VR = src.GetValidRow()` and `VC = src.GetValidCol()`. 对每个 destination element `(i, j)`:

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src}_{i,j} & \text{if } i < VR \text{ and } j < VC \\
\mathrm{pad}       & \text{otherwise}
\end{cases}
$$

`pad` is determined by `TileDataDst::PadVal` and the element type (e.g., `+inf/-inf` for floating types when available,
otherwise `std::numeric_limits<T>::max()/min()`).

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

Synchronous form (conceptual):

```text
%dst = tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tfillpad %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

Implemented in the backend headers pulled in by `include/pto/common/pto_instr_impl.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD(TileDataDst& dst, TileDataSrc& src);

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_INPLACE(TileDataDst& dst, TileDataSrc& src);

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_EXPAND(TileDataDst& dst, TileDataSrc& src);

template <typename TileData, PadValue PadVal = PadValue::Zero>
PTO_INTERNAL void TFILLPAD(TileData &dst, TileData &src);
```

## 约束

- `TileDataDst::PadVal != PadValue::Null`.
- `sizeof(TileDataDst::DType) == sizeof(TileDataSrc::DType)` and element size must be `1`, `2`, or `4` bytes.
- `TFILLPAD`: `TileDataDst::Rows/Cols` must match `TileDataSrc::Rows/Cols`.
- `TFILLPAD_EXPAND`: `TileDataDst::Rows >= TileDataSrc::Rows` and `TileDataDst::Cols >= TileDataSrc::Cols`.
- `TFILLPAD(TileData &dst, TileData &src)`:`if TileData::TileType is Mat, layout only support (!TileData::isRowMajor && TileData::Slayout::RowMajor), and PadVal only support PadValue::Zero`

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
# IR Level 2 (DPS)
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

