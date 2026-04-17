# pto.tnot

`pto.tnot` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对 tile 做逐元素按位取反。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \sim\mathrm{src}_{i,j} $$

这是 tile 版的一元位逻辑操作，适合做位翻转、补码掩码构造和位图预处理。

## 语法

### PTO-AS

```text
%dst = tnot %src : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tnot %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tnot ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TNOT(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## 输入

- `%src`：源 tile
- `%dst`：目标 tile

## 预期输出

- `%dst`：逐元素按位取反结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 这条指令只对整数元素类型有意义。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- 支持类型：`int16_t`、`uint16_t`
- tile 必须是行主序向量 tile
- 静态 valid 边界必须合法
- 运行时通常要求 `src` 与 `dst` 的 `validRow/validCol` 一致

### A5

- 支持类型：`uint32_t`、`int32_t`、`uint16_t`、`int16_t`、`uint8_t`、`int8_t`
- tile 必须是行主序向量 tile
- 静态 valid 边界必须合法
- 运行时通常要求 `src` 与 `dst` 的 `validRow/validCol` 一致

## 性能

### A2A3

英文页当前把 `TNOT` 归到一元 tile 运算桶：

| 指标 | 数值 |
| --- | --- |
| 启动时延 | 13 |
| 完成时延 | 26 |
| 每次 repeat 吞吐 | 1 |
| 流水间隔 | 18 |

### A5

当前手册未单列 `tnot` 的独立周期表，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT x, out;
  TNOT(out, x);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.texp](./texp_zh.md)
- 下一条指令：[pto.trelu](./trelu_zh.md)
