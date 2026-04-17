# pto.tfmod

`pto.tfmod` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对两个 tile 做逐元素 `fmod` 运算。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{fmod}(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) $$

它表示浮点余数语义，而不是整数模。常用于周期折返、相位归一化和浮点余数路径。

## 语法

### PTO-AS

```text
%dst = tfmod %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tfmod %src0, %src1 : !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tfmod ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TFMOD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## 输入

- `%src0`：被除数 tile
- `%src1`：除数 tile
- `%dst`：目标 tile

## 预期输出

- `%dst`：逐元素 `fmod` 结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 除零行为由目标 profile 定义；CPU 模拟器在调试构建下会断言。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

- `pto.tfmod` 在 CPU 仿真、A2/A3 和 A5 上保留相同的 PTO 可见语义，但具体支持子集仍取决于 profile。

## 性能

### A2A3

英文页当前把 `TFMOD` 归到和二元算术同一类估算口径：

| 指标 | FP | INT |
| --- | --- | --- |
| 启动时延 | 14 | 14 |
| 完成时延 | 19 | 17 |
| 每次 repeat 吞吐 | 2 | 2 |
| 流水间隔 | 18 | 18 |

### A5

当前手册未单列 `tfmod` 的独立周期表，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT out, a, b;
  TFMOD(out, a, b);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.trem](./trem_zh.md)
