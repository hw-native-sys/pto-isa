# pto.tdivs

`pto.tdivs` 属于[逐元素 Tile-标量](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对 Tile 与标量做逐元素除法（Tile/标量 或 标量/Tile），结果写入目标 tile。

## 机制

对有效区域内的每个元素 `(i, j)`：

- Tile/标量形式：$\mathrm{dst}_{i,j} = \frac{\mathrm{src}_{i,j}}{\mathrm{scalar}}$
- 标量/Tile 形式：$\mathrm{dst}_{i,j} = \frac{\mathrm{scalar}}{\mathrm{src}_{i,j}}$

迭代域由目标 tile 的 valid region 决定。除零行为由目标定义；在 A5 上，Tile/标量形式映射到乘以倒数，并对 `scalar == 0` 使用 `1/0 -> +inf`。

## 语法

### PTO-AS

Tile/标量形式：
```text
%dst = tdivs %src, %scalar : !pto.tile<...>, f32
```

标量/Tile 形式：
```text
%dst = tdivs %scalar, %src : f32, !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tdivs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst = pto.tdivs %scalar, %src : (dtype, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
pto.tdivs ins(%scalar, %src : dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents & ... events);

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileDataDst &dst, typename TileDataDst::DType scalar, TileDataSrc &src0,
                           WaitEvents & ... events)
```

`PrecisionType` 可指定以下值：
- `DivAlgorithm::DEFAULT`：普通算法，速度快但精度较低。
- `DivAlgorithm::HIGH_PRECISION`：高精度算法，速度较慢，仅在 A5 上有效。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | 被除数或除数 |
| `%scalar` | 标量 | 除数或被除数 |
| `PrecisionType` | 算法选项 | DEFAULT 或 HIGH_PRECISION |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | valid region 内每个元素等于 `src / scalar` 或 `scalar / src` |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 和 `src` 必须使用相同的元素类型。
- Tile 位置必须是向量。
- 静态有效边界：`TileData::ValidRow <= TileData::Rows` 且 `TileData::ValidCol <= TileData::Cols`。
- 运行时：`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`。
- 布局必须彼此兼容。
- 迭代域总是 `dst.GetValidRow() × dst.GetValidCol()`。
- A2/A3 支持的元素类型：`int32_t`、`int16_t`、`half`、`float`。
- A5 支持的元素类型：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`、`half`、`float`。
- `HIGH_PRECISION` 算法选项仅在 A5 上有效，在 A3 上将被忽略。

## 异常与非法情形

- 除零行为由目标定义。
- 源/目标类型不匹配会被 verifier 拒绝。
- 所选 target profile 不支持的元素类型会被后端拒绝。
- 程序不能依赖 `dst` valid region 之外的值。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `f32` | Simulated | Supported | Supported |
| `f16` | Simulated | Supported | Supported |
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
  TDIVS(dst, src, 2.0f);
  TDIVS<DivAlgorithm::HIGH_PRECISION>(dst, src, 2.0f);
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
  TDIVS(dst, 2.0f, src);
  TDIVS<DivAlgorithm::HIGH_PRECISION>(dst, 2.0f, src);
}
```

### PTO-AS

```text
%dst = tdivs %src, %scalar : !pto.tile<...>, f32
%dst = tdivs %scalar, %src : f32, !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
pto.tdivs ins(%scalar, %src : dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
