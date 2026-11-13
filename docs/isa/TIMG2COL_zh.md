# TIMG2COL

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

![TIMG2COL tile operation](../figures/isa/TIMG2COL.svg)

## 简介

用于类卷积工作负载的图像到列变换。

## 数学语义

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TIMG2COL(TileData &dst, ConvTileData &src, uint16_t posM = 0, uint16_t posK = 0, WaitEvents &... events);
```

## 约束

- 此指令是目标/实现特定的。有关支持的tile类型/布局和配置字段，请参见 `include/pto/npu/*/TImg2col.hpp`。
- **实现检查 (Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品)**：`TileData::DType` 必须是以下之一：`int8_t`、`half`、`bfloat16_t`、`float`。
- **实现检查 (Ascend 950PR/Ascend 950DT)**：`TileData::DType` 必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。

### ConvTile 转换路径

ConvTile 路径要求 `ConvTileData::Loc == TileType::Mat`，源布局为 `NC1HWC0` 或 `NDC1HWC0`。
目的路径和布局要求如下：

| 架构 | 目的 | 片上存储 | 布局要求 |
| --- | --- | --- | --- |
| A2A3 | `TileType::Left` | L0A | `SFractal == SLayout::RowMajor` 且 `BLayout == RowMajor` |
| A5 | `TileType::Left` | L0A | `SFractal == SLayout::RowMajor` 且 `BLayout == ColMajor` |
| A5 | `TileType::Right` | L0B | `SFractal == SLayout::ColMajor` 且 `BLayout == RowMajor` |

该路径要求目的与源 dtype 完全相同；卷积配置字段仍从源 ConvTile 读取。

当前 A5 的 `Right` 路径不使用源 ConvTile 的 `transpose` 配置；该字段不会改变 `Right`
目的的输出排列。A2A3 实现仍只接受 `Left` 目的。

## 示例

参见 `docs/isa/` 和 `docs/coding/tutorials/` 中的相关示例。
