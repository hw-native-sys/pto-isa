# pto.tsels

`pto.tsels` 属于[逐元素 Tile-标量](../../elementwise-tile-tile_zh.md)指令集。

## 概述

使用 mask tile 在源 tile 和标量之间做逐元素选择，结果写入目标 tile。

## 机制

对有效区域内每个元素 `(i, j)`：

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src}_{i,j} & \text{if } \mathrm{mask}_{i,j}\ \text{为真} \\
\mathrm{scalar} & \text{否则}
\end{cases}
$$

迭代域由目标 tile 的 valid region 决定。掩码 tile 被解释为目标定义布局中的打包谓词位。

## 语法

### PTO-AS

```text
%dst = tsels %mask, %src, %scalar : !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tsels %mask, %src, %scalar : (!pto.tile<...>, !pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tsels ins(%mask, %src, %scalar : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TSELS(TileDataDst &dst, TileDataMask &mask, TileDataSrc &src, TileDataTmp &tmp, typename TileDataSrc::DType scalar, WaitEvents & ... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%mask` | 掩码 tile | 控制每个位置选择源还是标量 |
| `%src` | 源 tile | 条件为真时的输出值 |
| `%scalar` | 标量 | 条件为假时的输出值 |
| `%tmp` | 临时 tile | 临时缓冲区 |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | valid region 内每个元素由 mask 决定选择 src 或 scalar |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 和 `src` 必须使用相同的元素类型。
- A2/A3：`sizeof(TileDataDst::DType)` 必须是 `2` 或 `4` 字节，支持 `half`、`float16_t`、`float` 和 `float32_t`。
- A5：`sizeof(TileDataDst::DType)` 可以是 `1`、`2` 或 `4` 字节，支持 `int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half` 和 `float`。
- `dst`、`mask` 和 `src` 必须是行主序。
- 运行时：`src.GetValidRow()/GetValidCol()` 必须与 `dst.GetValidRow()/GetValidCol()` 一致。
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
| `i32 / u32` | Simulated | No | Supported |
| `i16 / u16` | Simulated | No | Supported |
| `i8 / u8` | Simulated | No | Supported |
| 布局 | Any | RowMajor | RowMajor |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileDst = Tile<TileType::Vec, float, 16, 16>;
  using TileSrc = Tile<TileType::Vec, float, 16, 16>;
  using TileTmp = Tile<TileType::Vec, float, 16, 16>;
  using TileMask = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  TileDst dst;
  TileSrc src;
  TileTmp tmp;
  TileMask mask(16, 2);
  float scalar = 0.0f;
  TSELS(dst, mask, src, tmp, scalar);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileDst = Tile<TileType::Vec, float, 16, 16>;
  using TileSrc = Tile<TileType::Vec, float, 16, 16>;
  using TileTmp = Tile<TileType::Vec, float, 16, 16>;
  using TileMask = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  TileDst dst;
  TileSrc src;
  TileTmp tmp;
  TileMask mask(16, 2);
  float scalar = 0.0f;
  TASSIGN(src, 0x1000);
  TASSIGN(tmp, 0x2000);
  TASSIGN(dst, 0x3000);
  TASSIGN(mask, 0x4000);
  TSELS(dst, mask, src, tmp, scalar);
}
```

### PTO-AS

```text
%dst = tsels %mask, %src, %scalar : !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tsels ins(%mask, %src, %scalar : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```
