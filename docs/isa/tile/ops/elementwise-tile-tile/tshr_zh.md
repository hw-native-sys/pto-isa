# pto.tshr

`pto.tshr` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对两个 tile 做逐元素右移，其中第二个 tile 给出逐元素移位量。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \gg \mathrm{src1}_{i,j} $$

这同样是 tile/tile 的逐元素移位，而不是统一移位量的标量变体。

## 语法

### PTO-AS

```text
%dst = tshr %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tshr %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tshr ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSHR(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## 输入

- `%src0`：被右移的值 tile
- `%src1`：逐元素移位量 tile
- `%dst`：目标 tile

## 预期输出

- `%dst`：逐元素右移结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 只对整数元素类型有意义。
- `dst`、`src0`、`src1` 必须使用相同元素类型。
- 三者都必须是行主序。
- 运行时要求：`src0`、`src1` 的 valid shape 与 `dst` 一致。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- 支持类型：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`

### A5

- 支持类型：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`

## 性能

### A2A3

英文页当前把 `TSHR` 归到与二元算术同一类吞吐模型：

| 指标 | FP | INT |
| --- | --- | --- |
| 启动时延 | 14 | 14 |
| 完成时延 | 19 | 17 |
| 每次 repeat 吞吐 | 2 | 2 |
| 流水间隔 | 18 | 18 |

实际语义主要面向整数 tile。

### A5

当前手册未单列 `tshr` 的独立周期表，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileT x, sh, out;
  TSHR(out, x, sh);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tshl](./tshl_zh.md)
- 下一条指令：[pto.txor](./txor_zh.md)
