# pto.tsubs

`pto.tsubs` 属于[逐元素 Tile-to-Tile 指令](../../elementwise-tile-tile_zh.md)集。

## 概述

`TSUBS` 从 Tile 中逐元素减去一个标量，结果写入目标 Tile。对每个元素 `(i, j)` 在有效区域内，执行 `dst[i,j] = src[i,j] - scalar`。

## 机制

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} - \mathrm{scalar} $$

该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域。

## 语法

### PTO-AS

```text
%dst = tsubs %src, %scalar : !pto.tile<...>, f32
```

### AS Level 1（SSA）

```mlir
%dst = pto.tsubs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tsubs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSUBS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `dst` | 输出 | 目标 Tile |
| `src` | 输入 | 源 Tile |
| `scalar` | 标量 | 要减去的标量值 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 逐元素相减结果 |

## 副作用

该指令在执行向量流水线操作前可能隐式插入同步屏障。

## 约束

- `TileData::DType` 必须是以下之一：`int32_t`、`int`、`int16_t`、`half`、`float16_t`、`float`、`float32_t`。
- Tile 位置必须是向量（`TileData::Loc == TileType::Vec`）。
- 运行时：`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`。
- `dst` 和 `src0` 必须使用相同的元素类型。
- 标量类型必须与 `TileDataSrc::DType` 一致。
- 静态有效边界检查（A5）：`TileDataDst::ValidRow <= TileDataDst::Rows`、`TileDataDst::ValidCol <= TileDataDst::Cols`、`TileDataSrc::ValidRow <= TileDataSrc::Rows`，且 `TileDataSrc::ValidCol <= TileDataSrc::Cols`。

## 异常与非法情形

- 当 `TileData::DType` 不属于支持的数据类型列表时，行为未定义。
- 当 `src` 和 `dst` 的有效区域不匹配时，行为未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 支持的数据类型 | 是 | 是 | 是 |
| 向量位置要求 | - | 是 | 是 |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TSUBS(out, x, 1.0f);
}
```

### PTO-AS

```text
%dst = tsubs %src, %scalar : !pto.tile<...>, f32
```

### AS Level 2 (DPS)

```mlir
pto.tsubs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[逐元素 Tile-to-Tile](../../elementwise-tile-tile_zh.md)
- [TSUBS](../tile-scalar-and-immediate/tsubs_zh.md)
