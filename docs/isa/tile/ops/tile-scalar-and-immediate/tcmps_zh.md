# pto.tcmps

`pto.tcmps` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述

把 tile 和标量做逐元素比较，并把比较结果写入目标 tile。

## 机制

对目标 tile 的有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \left(\mathrm{src}_{i,j}\ \mathrm{cmpMode}\ \mathrm{scalar}\right) $$

`cmpMode` 选择比较谓词。目标 tile 的精确编码由实现决定，但通常表现为掩码式结果。

## 语法

同步形式：

```text
%dst = tcmps %src, %scalar {cmpMode = #pto.cmp<EQ>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tcmps %src, %scalar {cmpMode = #pto<cmp xx>} : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tcmps ins(%src, %scalar{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename T, typename... WaitEvents>
PTO_INST RecordEvent TCMPS(TileDataDst& dst, TileDataSrc0& src0, T src1, CmpMode cmpMode, WaitEvents&... events);
```

## 输入

- `src`：源 tile
- `scalar`：广播到所有元素的标量比较值
- `cmpMode`：比较模式
- `dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出

- `dst`：逐元素比较结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- tile 位置必须是向量 tile。
- 静态 valid 边界必须合法。
- 运行时要求：`src0` 与 `dst` 的 valid row / valid col 一致。
- 支持的比较模式包括 `EQ`、`NE`、`LT`、`GT`、`LE`、`GE`。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- `TileData::DType` 必须属于：`int32_t`、`float`、`half`、`uint16_t`、`int16_t`
- tile 布局必须是行主序

### A5

- `TileData::DType` 必须属于：`int32_t`、`float`、`half`、`uint16_t`、`int16_t`
- tile 布局必须是行主序

## 性能

当前仓内没有把 `tcmps` 单独落成公开 cost bucket。若代码依赖具体延迟，应把它视为目标 profile 相关的 tile-比较路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src;
  DstT dst(16, 2);
  TCMPS(dst, src, 0.0f, CmpMode::GT);
}
```

## 相关页面

- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.texpands](./texpands_zh.md)
- 下一条指令：[pto.tsels](./tsels_zh.md)
