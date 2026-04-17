# pto.trelu

`pto.trelu` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对 tile 做逐元素 ReLU。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \max(\mathrm{src}_{i,j}, 0) $$

它是 tile 版的逐元素 ReLU，用来把负值裁成 0。

## 语法

### PTO-AS

```text
%dst = trelu %src : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trelu %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trelu ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TRELU(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## 输入

- `%src`：源 tile
- `%dst`：目标 tile

## 预期输出

- `%dst`：逐元素 ReLU 结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- `src` 与 `dst` 的 validRow / validCol 通常应兼容。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- 支持类型：`half`、`float`、`int32_t`
- tile 必须是行主序向量 tile
- 静态 valid 边界必须合法
- 运行时通常要求 `src` 与 `dst` 的 `validRow/validCol` 一致

### A5

- 支持类型：`half`、`float`、`int32_t`
- tile 必须是行主序向量 tile
- 静态 valid 边界必须合法
- 运行时通常要求 `src` 与 `dst` 的 `validRow/validCol` 一致

## 性能

### A2A3

英文页把 `TRELU` 归入一元 tile 运算桶：

| 指标 | 数值 |
| --- | --- |
| 启动时延 | 13 |
| 完成时延 | 26（沿用一元 / 超越函数桶） |
| 每次 repeat 吞吐 | 1 |
| 流水间隔 | 18 |

### A5

当前手册未单列 `trelu` 的独立周期表，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TRELU(out, x);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tnot](./tnot_zh.md)
- 下一条指令：[pto.tneg](./tneg_zh.md)
