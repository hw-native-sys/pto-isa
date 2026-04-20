# pto.trems

`pto.trems` 属于[逐元素 Tile-标量](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对 Tile 与标量取逐元素余数，结果写入目标 tile。

## 机制

对有效区域内每个元素 `(i, j)`：

$$\mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \bmod \mathrm{scalar}$$

迭代域由目标 tile 的 valid region 决定。除零行为由目标定义，CPU 模拟器在调试构建中会断言。

## 语法

### PTO-AS

```text
%dst = trems %src, %scalar : !pto.tile<...>, f32
```

### AS Level 1（SSA）

```mlir
%dst = pto.trems %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.trems ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TREMS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                           TileDataTmp &tmp, WaitEvents & ... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | 被除数 |
| `%scalar` | 标量 | 除数 |
| `%tmp` | 临时 tile | 临时缓冲区 |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | valid region 内每个元素等于 `src % scalar` |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 和 `src` 必须使用相同的元素类型。
- A2/A3 支持的元素类型：`float` 和 `int32_t`。
- A5 支持的元素类型：`float`、`int32_t`、`uint32_t`、`half`、`int16_t` 和 `uint16_t`。
- `dst` 和 `src` 必须是向量 Tile。
- `dst` 和 `src` 必须是行主序。
- 运行时：`dst.GetValidRow() == src.GetValidRow()` 且 `dst.GetValidCol() == src.GetValidCol()`。
- 迭代域总是 `dst.GetValidRow() × dst.GetValidCol()`。
- A2/A3 的 tmp 缓冲区要求：`tmp.GetValidCol() >= dst.GetValidCol()`，`tmp.GetValidRow() >= 1`。
- A5 的 tmp 参数被接受但不进行验证或使用。
- 对于 `int32_t` 输入（A2/A3）：`src` 的元素和 `scalar` 必须在 `[-2^24, 2^24]` 范围内。

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
| `i32 / u32` | Simulated | Supported | Supported |
| `i16 / u16` | Simulated | No | Supported |
| 布局 | Any | RowMajor | RowMajor |

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  Tile<TileType::Vec, float, 16, 16> tmp;
  TREMS(out, x, 3.0f, tmp);
}
```

### PTO-AS

```text
%dst = trems %src, %scalar : !pto.tile<...>, f32
```

### AS Level 2（DPS）

```text
pto.trems ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```
