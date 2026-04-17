# pto.txors

`pto.txors` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述

对 tile 和标量逐元素做按位异或。

## 机制

对目标 tile 的有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \oplus \mathrm{scalar} $$

标量会广播到整个 valid region。它适合做逐元素翻转、切换位标记和统一扰动位模式。

## 语法

同步形式：

```text
%dst = txors %src, %scalar : !pto.tile<...>, i32
```

### AS Level 1（SSA）

```text
%dst = pto.txors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.txors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TXORS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入

- `src`：源 tile
- `scalar`：广播到所有元素的标量位掩码
- `tmp`：某些实现路径需要的临时 tile
- `dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出

- `dst`：逐元素按位异或结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 这条指令面向整数元素类型，不适用于浮点 tile。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- 支持元素类型：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`
- `dst`、`src` 和 `tmp` 必须使用相同元素类型
- 手动模式下，源、目标和临时存储不能互相重叠

### A5

- 支持元素类型：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`
- `dst` 与 `src` 的元素类型必须一致
- `src.GetValidRow()/GetValidCol()` 必须与 `dst` 一致

## 性能

当前仓内没有为 `txors` 单列公开 cost bucket。若代码依赖具体延迟，应把它视为目标 profile 相关的 tile 按位逻辑路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileSrc = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileTmp = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileDst dst;
  TileSrc src;
  TileTmp tmp;
  TXORS(dst, src, 0x1u, tmp);
}
```

## 相关页面

- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.tshrs](./tshrs_zh.md)
- 下一条指令：[pto.tlrelu](./tlrelu_zh.md)
