# TIMG2COL

## 指令示意图

![TIMG2COL tile operation](../figures/isa/TIMG2COL.svg)

## 简介

用于类卷积工作负载的图像到列变换。

## 数学语义

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TIMG2COL(TileData &dst, ConvTileData &src, uint16_t posM = 0, uint16_t posK = 0, WaitEvents &... events);
```

## 约束

- 此指令是目标/实现特定的。有关支持的 tile 类型/布局和配置字段，请参见 `include/pto/npu/*/TImg2col.hpp`。

## 示例

参见 `docs/isa/` 和 `docs/coding/tutorials/` 中的相关示例。
