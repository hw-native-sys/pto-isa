# pto.txor

`pto.txor` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对两个 tile 做逐元素按位异或。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \oplus \mathrm{src1}_{i,j} $$

它适合做逐元素位翻转、差异位检测和位级扰动。

## 语法

### PTO-AS

```text
%dst = txor %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.txor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.txor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TXOR(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入

- `%src0`、`%src1`：两个源 tile
- `%tmp`：A2A3 路径需要的临时 tile
- `%dst`：目标 tile

## 预期输出

- `%dst`：逐元素按位异或结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 这条指令只对整数元素类型有意义。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A5

- `dst`、`src0`、`src1` 的元素类型必须一致
- 支持类型：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`
- 三者都必须是行主序
- `src0` 与 `src1` 的 valid shape 必须与 `dst` 一致

### A2A3

- `dst`、`src0`、`src1`、`tmp` 的元素类型必须一致
- 支持类型：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`
- `dst`、`src0`、`src1`、`tmp` 都必须是行主序
- `src0`、`src1`、`tmp` 的 valid shape 必须与 `dst` 一致
- 手动模式下，`dst`、`src0`、`src1`、`tmp` 之间不能重叠

## 性能

### A2A3

英文页当前把 `TXOR` 归到与二元算术同一类吞吐模型：

| 指标 | FP | INT |
| --- | --- | --- |
| 启动时延 | 14 | 14 |
| 完成时延 | 19 | 17 |
| 每次 repeat 吞吐 | 2 | 2 |
| 流水间隔 | 18 | 18 |

实际语义主要面向整数 tile。

### A5

当前手册未单列 `txor` 的独立周期表，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileSrc0 = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileSrc1 = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileTmp = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileDst dst;
  TileSrc0 src0;
  TileSrc1 src1;
  TileTmp tmp;
  TXOR(dst, src0, src1, tmp);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tshr](./tshr_zh.md)
- 下一条指令：[pto.tlog](./tlog_zh.md)
