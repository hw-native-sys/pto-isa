# pto.tdivs

![TDIVS tile operation](../figures/isa/TDIVS.svg)

`pto.tdivs` 属于[Tile-标量与立即数](./tile/tile-scalar-and-immediate_zh.md)指令集。

## 概述

带标量的逐元素除法，支持 tile / scalar 和 scalar / tile 两种方向，标量广播到 tile 有效区域的所有元素。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

- tile / scalar 形式：

  $$ \mathrm{dst}_{i,j} = \frac{\mathrm{src}_{i,j}}{\mathrm{scalar}} $$

- scalar / tile 形式：

  $$ \mathrm{dst}_{i,j} = \frac{\mathrm{scalar}}{\mathrm{src}_{i,j}} $$

除零行为由目标 profile 定义。在 A5 上，tile / scalar 形式通常映射到"乘以倒数"的实现路径。

## 语法

### PTO-AS

tile / scalar 形式：

```text
%dst = tdivs %src, %scalar : !pto.tile<...>, f32
```

scalar / tile 形式：

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
                           WaitEvents &... events);

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileDataDst &dst, typename TileDataDst::DType scalar, TileDataSrc &src0,
                           WaitEvents &... events);
```

`PrecisionType` 可选：

- `DivAlgorithm::DEFAULT`：普通算法，速度快但精度较低。
- `DivAlgorithm::HIGH_PRECISION`：高精度算法，速度较慢。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | tile / scalar 形式中被除数 |
| `%scalar` | 标量 | 广播到所有元素的标量值 |
| `%dst` | 目标 tile | 接收逐元素除法结果 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | `dst` valid region 内的每个元素都等于对应形式的除法结果 |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 除零行为由目标 profile 定义。
- `HIGH_PRECISION` 只在 A5 可用，A3 上该选项会被忽略。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `int32_t` / `uint32_t` | Simulated | Supported | Supported |
| `int16_t` / `uint16_t` | Simulated | Supported | Supported |
| `float` | Simulated | Supported | Supported |
| `half` | Simulated | Supported | Supported |
| `int8_t` / `uint8_t` | Simulated | No | Supported |
| 布局 | Any | RowMajor only | RowMajor only |

A2/A3 支持：`int32_t`、`int16_t`、`half`、`float`；A5 额外支持 `uint32_t`、`uint16_t`、`int8_t`、`uint8_t`。

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
# tile / scalar 自动模式
%dst = pto.tdivs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>

# scalar / tile 自动模式
%dst = pto.tdivs %scalar, %src : (dtype, !pto.tile<...>) -> !pto.tile<...>

# 手动模式
pto.tassign %arg0, @tile(0x1000)
pto.tassign %arg1, @tile(0x2000)
%dst = pto.tdivs %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>

# PTO 汇编形式
%dst = tdivs %src, %scalar : !pto.tile<...>, f32
%dst = tdivs %scalar, %src : f32, !pto.tile<...>
# AS Level 2 (DPS)
pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[Tile-标量与立即数](./tile/tile-scalar-and-immediate_zh.md)
- 规范页：[pto.tdivs](./tile/ops/tile-scalar-and-immediate/tdivs_zh.md)
