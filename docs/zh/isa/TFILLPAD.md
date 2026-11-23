# TFILLPAD

## 产品支持情况<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->


## 指令示意图

![TFILLPAD tile operation](../../figures/isa/TFILLPAD.svg)

## 简介

复制Tile并在有效区域外使用编译时填充值进行填充。

将源Tile复制到目标Tile中，并使用由 `TileDataDst::PadVal` 选定的编译时填充值（如 `PadValue::Min`/`PadValue::Max`）填充剩余（填充）元素。

此操作常用于在运行时有效区域之外确定性地物化特定值，使后续操作能够在完整的静态Tile形状上运算。

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

## C++内建接口

在 `include/pto/common/pto_instr_impl.hpp` 引入的后端头文件中实现：

```cpp
template <typename TileData, PadValue PadVal = PadValue::Zero, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(TileData &dst, TileData &src, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

## 约束

- `TileDataDst::PadVal != PadValue::Null`（Vec类型重载）.
- `sizeof(TileDataDst::DType) == sizeof(TileDataSrc::DType)`且元素大小必须是 `1`， `2`，或`4`字节。
  打包的 `fp4x2` 按 1 字节 DType 计算（每个元素两个 nibble），与 `s8`/`u8` 相同。
  在 A5 上，`fp4x2` 的 `ValidCol`/`Cols` 按 nibble 计数（与 TLOAD/TSTORE/TCVT 相同）；填充长度为 `ceil(Cols/2)` 个打包字节。
- `TFILLPAD`： `TileDataDst::Rows/Cols` 必须匹配 `TileDataSrc::Rows/Cols`。
- `TFILLPAD_EXPAND`： `TileDataDst::Rows >= TileDataSrc::Rows`且`TileDataDst::Cols >= TileDataSrc::Cols`。
- `TFILLPAD(TileData &dst, TileData &src)`（Mat 类型重载）：当 `TileData::TileType` 为 `Mat` 时，布局必须满足 `!TileData::isRowMajor && TileData::SLayout::RowMajor`，且 `PadVal` 必须为 `PadValue::Zero` 或 `PadValue::Null`。该 Mat 重载与首条 Vec 重载（`PadVal != PadValue::Null`）分属不同 SFINAE 重载，二者并不矛盾。

## PadValue 映射（Vec）

`PadValue::Zero` / `Min` / `Max` 是由 `PadValueMap` 决定的、与数据类型相关的位模式。
`PadValue::Null` 为 0。自定义取值（`PadValueCustom(...)`）按原始位透传。

| DType | Zero | Min | Max | 说明 |
| --- | --- | --- | --- | --- |
| `float` / `half` / `bfloat16_t` | `0` | `-inf` | `+inf` | IEEE inf |
| 整型类型 | `0` | 类型最小值 | 类型最大值 | |
| `float8_e4m3_t` | `0x00` | `0xFE` | `0x7E` | 无 inf；取有限最大/最小值 |
| `float8_e5m2_t` | `0x00` | `0xFC` | `0x7C` | `±inf` |
| `hifloat8_t` | `0x00` | `0xEF` | `0x6F` | HiF8 `±inf`（`S1101111`） |
| `float4_e2m1x2_t` | `0x00` | `0xFF` | `0x77` | 两个 nibble 均为 `-6` / `+6` |
| `float4_e1m2x2_t` | `0x00` | `0xFF` | `0x77` | 两个 nibble 均为有限最小/最大值 |

`float8_e8m0_t` 没有 Zero/Min/Max 语法糖，须使用 `PadValueCustom`。低精度映射适用于 A5（fp8/fp4 另支持 CPU 仿真）。`hifloat8_t` 仅 A5 支持。

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
