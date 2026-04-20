# pto.tadds

`pto.tadds` 属于[逐元素 Tile-标量](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对 Tile 与标量做逐元素加法，结果写入目标 tile。

## 机制

对有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} + \mathrm{scalar} $$

迭代域由目标 tile 的 valid region 决定。

## 语法

### PTO-AS

```text
%dst = tadds %src, %scalar : !pto.tile<...>, f32
```

### AS Level 1（SSA）

```mlir
%dst = pto.tadds %src, %scalar : (!pto.tile<...>,dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tadds ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TADDS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | 逐元素加法的左操作数 |
| `%scalar` | 标量 | 逐元素加法的右操作数 |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | valid region 内每个元素等于 `src + scalar` |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 和 `src` 必须使用相同的元素类型。
- Tile 位置必须是向量。
- 静态有效边界：`TileData::ValidRow <= TileData::Rows` 且 `TileData::ValidCol <= TileData::Cols`。
- 运行时：`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`（A2/A3）。
- 运行时：`src0.GetValidCol() == dst.GetValidCol()`（A5）。
- 布局必须彼此兼容。
- 迭代域总是 `dst.GetValidRow() × dst.GetValidCol()`。

## 异常与非法情形

- 源/目标类型不匹配会被 verifier 拒绝。
- 所选 target profile 不支持的元素类型会被后端拒绝。
- 程序不能依赖 `dst` valid region 之外的值。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `f32` | Simulated | Supported | Supported |
| `f16` | Simulated | Supported | Supported |
| `bf16` | Simulated | No | Supported |
| `i32` | Simulated | Supported | Supported |
| `i16` | Simulated | Supported | Supported |
| `i8 / u8` | Simulated | No | Supported |
| 布局 | Any | RowMajor | RowMajor |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TADDS(dst, src, 1.0f);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TADDS(dst, src, 1.0f);
}
```

### PTO-AS

```text
%dst = tadds %src, %scalar : !pto.tile<...>, f32
```

### AS Level 2（DPS）

```text
pto.tadds ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```
