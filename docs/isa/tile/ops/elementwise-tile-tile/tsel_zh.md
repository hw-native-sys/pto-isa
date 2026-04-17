# pto.tsel

`pto.tsel` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

根据掩码 tile，在两个源 tile 之间逐元素选择。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src0}_{i,j} & \text{当 } \mathrm{mask}_{i,j}\ \text{为真} \\
\mathrm{src1}_{i,j} & \text{否则}
\end{cases}
$$

掩码 tile 使用目标定义的 packed predicate 编码，`tmp` 用作谓词展开时的临时缓冲。

## 语法

### PTO-AS

```text
%dst = tsel %mask, %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileData, typename MaskTile, typename TmpTile, typename... WaitEvents>
PTO_INST RecordEvent TSEL(TileData &dst, MaskTile &selMask, TileData &src0,
                          TileData &src1, TmpTile &tmp, WaitEvents &... events);
```

## 输入

- `%mask`：掩码 tile
- `%src0`：掩码为真时选择的 tile
- `%src1`：掩码为假时选择的 tile
- `%tmp`：谓词展开所需临时 tile
- `%dst`：目标 tile

## 预期输出

- `%dst`：逐元素选择后的结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `sizeof(TileData::DType)` 必须是 2 或 4 字节。
- `dst`、`src0`、`src1` 必须使用相同元素类型。
- `dst`、`src0`、`src1` 必须是行主序。
- 选择域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- `tmp` 必须有足够容量承载谓词展开过程。

## 不允许的情形

- `dst`、`src0`、`src1` 使用不同 shape。
- 使用非行主序 tile。
- 假设掩码 tile 的具体位打包格式是跨目标固定的。

## Target-Profile 限制

### A2A3

- `dtype` 必须是 2 或 4 字节
- 支持类型：`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`

### A5

- `dtype` 必须是 2 或 4 字节
- 支持类型：`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`

## 性能

### A2A3

英文页当前把 `TSEL` 归到与二元算术同一类模型：

| 指标 | FP | INT |
| --- | --- | --- |
| 启动时延 | 14 | 14 |
| 完成时延 | 19 | 17 |
| 每次 repeat 吞吐 | 2 | 2 |
| 流水间隔 | 18 | 18 |

### A5

当前手册未单列 `tsel` 的独立周期表，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, uint32_t, 1, 16>;
  TileT src0, src1, dst;
  MaskT mask(16, 2);
  TmpT tmp;
  TSEL(dst, mask, src0, src1, tmp);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tcvt](./tcvt_zh.md)
- 下一条指令：[pto.trsqrt](./trsqrt_zh.md)
