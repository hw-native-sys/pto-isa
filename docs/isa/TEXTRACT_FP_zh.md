# TEXTRACT_FP

## 指令示意图

![TEXTRACT_FP tile operation](../figures/isa/TEXTRACT_FP.svg)

## 简介

带 fp/缩放 Tile 的提取（向量量化参数）。

## 数学语义

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);
```

## 约束

类型/布局/位置/形状的合法性取决于后端；将实现特定的说明视为该后端的规范。

## 示例

参见 `docs/isa/` 和 `docs/coding/tutorials/` 中的相关示例。
