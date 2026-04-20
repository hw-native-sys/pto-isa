# pto.tshrs

`pto.tshrs` 属于[逐元素 Tile-标量](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对 Tile 按标量做逐元素右移，结果写入目标 tile。

## 机制

对有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \gg \mathrm{scalar} $$

迭代域由目标 tile 的 valid region 决定。标量仅支持零和正值。

## 语法

### PTO-AS

```text
%dst = tshrs %src, %scalar : !pto.tile<...>, i32
```

### AS Level 1（SSA）

```mlir
%dst = pto.tshrs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tshrs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSHRS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar, WaitEvents & ... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | 被移位的值 |
| `%scalar` | 标量 | 移位量（仅零和正值） |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | valid region 内每个元素等于 `src >> scalar` |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 和 `src` 必须使用相同的元素类型。
- `dst` 和 `src` 必须是向量 Tile。
- 运行时：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`。
- 标量仅支持零和正值。
- A2/A3 支持的元素类型：`int32_t`、`int`、`int16_t`、`uint32_t`、`unsigned int` 和 `uint16_t`。
- A5 支持的元素类型：`int32_t`、`int16_t`、`int8_t`、`uint32_t`、`uint16_t` 和 `uint8_t`。
- 两个 Tile 的静态有效边界都必须满足 `ValidRow <= Rows` 且 `ValidCol <= Cols`。
- 迭代域总是 `dst.GetValidRow() × dst.GetValidCol()`。

## 异常与非法情形

- 源/目标类型不匹配会被 verifier 拒绝。
- 所选 target profile 不支持的元素类型会被后端拒绝。
- 负数移位量会被拒绝。
- 程序不能依赖 `dst` valid region 之外的值。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `i32` | Simulated | Supported | Supported |
| `i16` | Simulated | Supported | Supported |
| `i8` | Simulated | No | Supported |
| `u32` | Simulated | Supported | Supported |
| `u16` | Simulated | Supported | Supported |
| `u8` | Simulated | No | Supported |

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint16_t, 16, 16>;
  using TileSrc = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileDst dst;
  TileSrc src;
  TSHRS(dst, src, 0x2);
}
```

### PTO-AS

```text
%dst = tshrs %src, %scalar : !pto.tile<...>, i32
```

### AS Level 2（DPS）

```text
pto.tshrs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```
