# pto.tor

`pto.tor` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对两个 tile 做逐元素按位或。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \;|\; \mathrm{src1}_{i,j} $$

它常用于逐元素置位、标志聚合和位图合并。

## 语法

### PTO-AS

```text
%dst = tor %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TOR(TileData &dst, TileData &src0, TileData &src1, WaitEvents &... events);
```

## 输入

- `%src0`、`%src1`：两个源 tile
- `%dst`：目标 tile

## 预期输出

- `%dst`：逐元素按位或结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 这条指令只对整数元素类型有意义。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- 支持 1 字节或 2 字节整数类型。
- `dst`、`src0`、`src1` 必须使用相同元素类型。
- 三者都必须是行主序。
- 运行时要求：`src0`、`src1` 的 valid shape 必须与 `dst` 一致。

### A5

- 支持 `uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`
- `dst`、`src0`、`src1` 必须使用相同元素类型。
- 三者都必须是行主序。
- 运行时要求：`src0`、`src1` 的 valid shape 必须与 `dst` 一致。

## 性能

### A2A3

英文页当前把 `TOR` 归到与二元算术同一类吞吐模型：

| 指标 | FP | INT |
| --- | --- | --- |
| 启动时延 | 14 | 14 |
| 完成时延 | 19 | 17 |
| 每次 repeat 吞吐 | 2 | 2 |
| 流水间隔 | 18 | 18 |

这主要是统一模型口径，不表示浮点 tile 是推荐使用场景。

### A5

当前手册未单列 `tor` 的独立周期表，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT a, b, out;
  TOR(out, a, b);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tand](./tand_zh.md)
- 下一条指令：[pto.tsub](./tsub_zh.md)
