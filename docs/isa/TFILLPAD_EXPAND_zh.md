# TFILLPAD_EXPAND

## 指令示意图

![TFILLPAD_EXPAND tile operation](../figures/isa/TFILLPAD_EXPAND.svg)

## 简介

填充/填充时允许目标大于源。

## 数学语义

除非另有说明，语义定义在有效区域上，目标相关行为标记为实现定义。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_EXPAND(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

## 约束

类型/布局/位置/形状的合法性由后端决定；对于特定后端，请将实现相关说明视为规范性约束。

## 示例

参见 `docs/isa/` 和 `docs/coding/tutorials/` 中的相关示例。
