# pto.taddc

`pto.taddc` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

三输入逐元素加法：`src0 + src1 + src2`。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{src1}_{i,j} + \mathrm{src2}_{i,j} $$

它是一条三输入 tile 算术指令，重点在于把三路源 tile 直接合并进一次逐元素运算。

## 语法

### PTO-AS

```text
%dst = taddc %src0, %src1, %src2 : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.taddc %src0, %src1, %src2 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.taddc ins(%src0, %src1, %src2 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TADDC(TileData &dst, TileData &src0, TileData &src1, TileData &src2, WaitEvents &... events);
```

## 输入

- `%src0`、`%src1`、`%src2`：三个源 tile
- `%dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出

- `%dst`：逐元素三输入加法结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 三个源 tile 与目标 tile 应在 shape、layout 和 valid region 上兼容。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

- `pto.taddc` 在 CPU 仿真、A2/A3 和 A5 上保留一致的 PTO 可见语义，但具体支持子集仍取决于 profile。

## 性能

### A2A3

英文页当前把 `TADDC` 归到与二元算术同类的估算口径：

| 指标 | FP | INT |
| --- | --- | --- |
| 启动时延 | 14 | 14 |
| 完成时延 | 19 | 17 |
| 每次 repeat 吞吐 | 2 | 2 |
| 流水间隔 | 18 | 18 |

### A5

当前手册未单列 `taddc` 的独立周期表，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c, out;
  TADDC(out, a, b, c);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tprelu](./tprelu_zh.md)
- 下一条指令：[pto.tsubc](./tsubc_zh.md)
