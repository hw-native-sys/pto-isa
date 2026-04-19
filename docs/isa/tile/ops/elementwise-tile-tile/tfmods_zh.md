# pto.tfmods

`pto.tfmods` 属于[逐元素 Tile-标量](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对浮点 Tile 与标量取逐元素浮点余数，结果写入目标 tile。

## 机制

对有效区域内每个元素 `(i, j)`：

$$\mathrm{dst}_{i,j} = \mathrm{fmod}(\mathrm{src}_{i,j}, \mathrm{scalar})$$

迭代域由目标 tile 的 valid region 决定。除零行为由目标定义，CPU 模拟器在调试构建中会断言。

## 语法

### PTO-AS

```text
%dst = tfmods %src, %scalar : !pto.tile<...>, f32
```

### AS Level 1（SSA）

```mlir
%dst = pto.tfmods %src, %scalar : !pto.tile<...>, f32
```

### AS Level 2（DPS）

```mlir
pto.tfmods ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TFMODS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar, WaitEvents & ... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | 被除数 |
| `%scalar` | 标量 | 除数 |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | valid region 内每个元素等于 `fmod(src, scalar)` |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 和 `src` 必须使用相同的元素类型。
- A2/A3 支持的元素类型：`float` 和 `float32_t`。
- A5 支持的元素类型：目标实现支持的 2 字节或 4 字节浮点类型。
- `dst` 和 `src` 必须是向量 Tile。
- `dst` 和 `src` 必须是行主序。
- 运行时：`dst.GetValidRow() == src.GetValidRow()` 且 `dst.GetValidCol() == src.GetValidCol()`。
- 迭代域总是 `dst.GetValidRow() × dst.GetValidCol()`。

## 异常与非法情形

- 除零行为由目标定义。
- 源/目标类型不匹配会被 verifier 拒绝。
- 所选 target profile 不支持的元素类型会被后端拒绝。
- 程序不能依赖 `dst` valid region 之外的值。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `f32` | Simulated | Supported | Supported |
| `f16` | Simulated | No | Supported |
| 布局 | Any | RowMajor | RowMajor |

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TFMODS(out, x, 3.0f);
}
```

### PTO-AS

```text
%dst = tfmods %src, %scalar : !pto.tile<...>, f32
```

### AS Level 2（DPS）

```text
pto.tfmods ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
```
