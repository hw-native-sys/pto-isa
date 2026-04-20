# pto.txors

`pto.txors` 属于[逐元素 Tile-标量](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对 Tile 与标量做逐元素按位异或，结果写入目标 tile。

## 机制

对有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \oplus \mathrm{scalar} $$

迭代域由目标 tile 的 valid region 决定。

## 语法

### PTO-AS

```text
%dst = txors %src, %scalar : !pto.tile<...>, i32
```

### AS Level 1（SSA）

```mlir
%dst = pto.txors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.txors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TXORS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | 逐元素异或的左操作数 |
| `%scalar` | 标量 | 逐元素异或的右操作数 |
| `%tmp` | 临时 tile | 临时缓冲区 |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | valid region 内每个元素等于 `src ⊕ scalar` |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst`、`src` 和 `tmp` 必须使用相同的元素类型。
- 布局必须彼此兼容。
- 迭代域总是 `dst.GetValidRow() × dst.GetValidCol()`。
- A2/A3 支持的元素类型为 `uint8_t`、`int8_t`、`uint16_t` 和 `int16_t`。
- A5 支持的元素类型为 `uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t` 和 `int32_t`。
- A5 要求 `src.GetValidRow()/GetValidCol()` 与 `dst` 一致。
- 在手动模式下，源、目标和临时存储的内存区域不得重叠。

## 异常与非法情形

- 源/目标类型不匹配会被 verifier 拒绝。
- 所选 target profile 不支持的元素类型会被后端拒绝。
- 程序不能依赖 `dst` valid region 之外的值。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `i8 / u8` | Simulated | Supported | Supported |
| `i16 / u16` | Simulated | Supported | Supported |
| `i32 / u32` | Simulated | No | Supported |
| 布局 | Any | RowMajor | RowMajor |

## 示例

### C++ 自动模式

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

### PTO-AS

```text
%dst = txors %src, %scalar : !pto.tile<...>, i32
```

### AS Level 2（DPS）

```text
pto.txors ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```
