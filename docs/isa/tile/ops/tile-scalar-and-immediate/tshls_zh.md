# pto.tshls

`pto.tshls` 属于[Tile 标量与立即数指令](../../tile-scalar-and-immediate_zh.md)集。

## 概述

Tile 按标量逐元素进行有符号左移操作。对有效区域内的每个元素，将其值左移指定的立即数位数。

## 机制

### 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \ll \mathrm{scalar} $$

其中 `scalar` 为非负整数移位量。

## 语法

### PTO-AS

```text
%dst = tshls %src, %scalar : !pto.tile<...>, i32
```

### AS Level 1（SSA）

```mlir
%dst = pto.tshls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tshls ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSHLS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源 Tile |
| `scalar` | 立即数 | 非负移位量 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 左移结果，有效区域内有效 |

## 副作用

无。

## 约束

- A2/A3 实现检查：
    - 支持的元素类型为 `int32_t`、`int`、`int16_t`、`uint32_t`、`unsigned int` 和 `uint16_t`
    - `dst` 和 `src` 必须使用相同的元素类型
    - `dst` 和 `src` 必须是向量 Tile
    - 运行时：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`
    - 标量仅支持零和正值
- A5 实现检查：
    - 支持的元素类型为 `int32_t`、`int16_t`、`int8_t`、`uint32_t`、`uint16_t` 和 `uint8_t`
    - `dst` 和 `src` 必须使用相同的元素类型
    - `dst` 和 `src` 必须是向量 Tile
    - 两个 Tile 的静态有效边界都必须满足 `ValidRow <= Rows` 且 `ValidCol <= Cols`
    - 运行时：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`
    - 标量仅支持零和正值
- 有效区域：
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域

## 异常与非法情形

- 若 `src` 和 `dst` 的元素类型不匹配，编译失败
- 若 `scalar` 为负值，行为未定义
- 若 valid region 不匹配，行为未定义

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| int32_t | 是 | 是 | 是 |
| int16_t | 是 | 是 | 是 |
| int8_t | 是 | 否 | 是 |
| uint32_t | 是 | 是 | 是 |
| uint16_t | 是 | 是 | 是 |
| uint8_t | 是 | 否 | 是 |

## 示例

### C++

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint16_t, 16, 16>;
  using TileSrc = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileDst dst;
  TileSrc src;
  TSHLS(dst, src, 0x2);
}
```

### PTO-AS

```text
# 自动模式：由编译器/运行时负责资源放置与调度
%dst = pto.tshls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>

# 手动模式：先显式绑定资源，再发射指令
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tshls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

## 相关页面

- 指令集总览：[Tile 标量与立即数指令](../../tile-scalar-and-immediate_zh.md)
