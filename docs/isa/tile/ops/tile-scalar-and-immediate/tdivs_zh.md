# pto.tdivs

`pto.tdivs` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述

带标量的逐元素除法，既支持 tile / scalar，也支持 scalar / tile 两种方向。

## 机制

对目标 tile 的有效区域内每个元素 `(i, j)`：

- tile / scalar 形式：

  $$ \mathrm{dst}_{i,j} = \frac{\mathrm{src}_{i,j}}{\mathrm{scalar}} $$

- scalar / tile 形式：

  $$ \mathrm{dst}_{i,j} = \frac{\mathrm{scalar}}{\mathrm{src}_{i,j}} $$

这条指令仍然作用在 tile payload 上；标量只是广播来源。

## 语法

tile / scalar 形式：

```text
%dst = tdivs %src, %scalar : !pto.tile<...>, f32
```

scalar / tile 形式：

```text
%dst = tdivs %scalar, %src : f32, !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tdivs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst = pto.tdivs %scalar, %src : (dtype, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
pto.tdivs ins(%scalar, %src : dtype, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &... events);

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileDataDst &dst, typename TileDataDst::DType scalar, TileDataSrc &src0,
                           WaitEvents &... events);
```

`PrecisionType` 可取：

- `DivAlgorithm::DEFAULT`
- `DivAlgorithm::HIGH_PRECISION`

## 输入

- `src`：源 tile
- `scalar`：广播到所有元素的标量
- `dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出

- `dst`：逐元素除法后的结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 除零行为由目标 profile 定义。
- `HIGH_PRECISION` 只在 A5 可用，A3 上该选项会被忽略。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- 数据类型必须属于：`int32_t`、`int`、`int16_t`、`half`、`float16_t`、`float`、`float32_t`
- tile 位置必须是向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`
- tile 布局必须是行主序

### A5

- 数据类型必须属于：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`、`half`、`float`
- tile 位置必须是向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`
- tile 布局必须是行主序
- tile / scalar 形式在 A5 backend 上通常会映射到“乘以倒数”的实现路径，`scalar == 0` 的行为遵循目标浮点异常约定

## 性能

### A2A3

仓内 `include/pto/costmodel/pto_isa_costmodel.hpp` 直接给出 `TDIVS -> vdivs` 对应桶：

| 指标 | INT16 / INT32 | FP16 / FP32 |
| --- | --- | --- |
| 启动时延 | 14 (`A2A3_STARTUP_BINARY`) | 14 (`A2A3_STARTUP_BINARY`) |
| 完成时延 | 18 (`A2A3_COMPL_INT_MUL`) | 20 (`A2A3_COMPL_FP_MUL`) |
| 每次 repeat 吞吐 | 1 (`A2A3_RPT_1`) | 1 (`A2A3_RPT_1`) |
| 流水间隔 | 18 (`A2A3_INTERVAL`) | 18 (`A2A3_INTERVAL`) |

### A5

当前手册未单列 `tdivs` 的独立周期条目，应视为目标 profile 相关。

## 示例

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

## 相关页面

- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.tsubs](./tsubs_zh.md)
- 下一条指令：[pto.tmuls](./tmuls_zh.md)
