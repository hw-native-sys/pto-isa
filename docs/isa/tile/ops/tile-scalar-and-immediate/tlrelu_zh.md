# pto.tlrelu

`pto.tlrelu` 属于[逐元素 Tile-标量](../../elementwise-tile-tile_zh.md)指令集。

## 概述

带标量斜率的 Leaky ReLU 激活，对 Tile 逐元素执行，结果写入目标 tile。

## 机制

对有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = (\mathrm{src}_{i,j} > 0) ? \mathrm{src}_{i,j} : (\mathrm{src}_{i,j} \cdot \mathrm{slope}) $$

迭代域由目标 tile 的 valid region 决定。

## 语法

### PTO-AS

```text
%dst = tlrelu %src, %slope : !pto.tile<...>, f32
```

### AS Level 1（SSA）

```mlir
%dst = pto.tlrelu %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tlrelu ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TLRELU(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, WaitEvents&... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | Leaky ReLU 的输入 |
| `%slope` | 标量 | 负值的乘数 |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | valid region 内每个元素为 Leaky ReLU 结果 |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 和 `src` 必须使用相同的元素类型。
- 标量类型必须与 Tile 数据类型一致。
- Tile 位置必须是向量。
- 静态有效边界：`TileData::ValidRow <= TileData::Rows` 且 `TileData::ValidCol <= TileData::Cols`。
- 运行时：`dst` 和 `src` 的有效行列数必须相同。
- 布局必须是行主序。
- 迭代域总是 `dst.GetValidRow() × dst.GetValidCol()`。
- A2/A3 支持的浮点类型：`half`、`float16_t`、`float`、`float32_t`。
- A5 支持的浮点类型：`half`、`float`。

## 异常与非法情形

- 源/目标类型不匹配会被 verifier 拒绝。
- 所选 target profile 不支持的元素类型会被后端拒绝。
- 程序不能依赖 `dst` valid region 之外的值。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `f32` | Simulated | Supported | Supported |
| `f16` | Simulated | Supported | Supported |
| 布局 | Any | RowMajor | RowMajor |

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TLRELU(out, x, 0.1f);
}
```

### PTO-AS

```text
%dst = tlrelu %src, %slope : !pto.tile<...>, f32
```

### AS Level 2（DPS）

```text
pto.tlrelu ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```
