# pto.tdiv

![TDIV tile operation](../figures/isa/TDIV.svg)

`pto.tdiv` 属于[逐元素 Tile-Tile](./tile/elementwise-tile-tile_zh.md)指令集。

## 概述

对两个 tile 做逐元素除法，结果写入目标 tile。迭代域由目标 tile 的 valid region 决定。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \frac{\mathrm{src0}_{i,j}}{\mathrm{src1}_{i,j}} $$

除零行为由目标 profile 定义。

## 语法

### PTO-AS

```text
%dst = tdiv %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tdiv %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tdiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

`PrecisionType` 可选：

- `DivAlgorithm::DEFAULT`：普通算法，速度快但精度较低。
- `DivAlgorithm::HIGH_PRECISION`：高精度算法，速度较慢。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src0` | 左 tile | 被除数 tile，在 `dst` valid region 上逐坐标读取 |
| `%src1` | 右 tile | 除数 tile，在 `dst` valid region 上逐坐标读取 |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | `dst` valid region 内的每个元素都等于 `src0 / src1` |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 除零行为由目标 profile 定义。
- `HIGH_PRECISION` 选项只在 A5 有效，A3 上会被忽略。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `float` | Simulated | Supported | Supported |
| `half` | Simulated | Supported | Supported |
| `int32_t` | Simulated | No | Supported |
| `uint32_t` | Simulated | No | Supported |
| `int16_t` | Simulated | No | Supported |
| `uint16_t` | Simulated | No | Supported |
| 布局 | Any | RowMajor only | RowMajor only |

A2/A3 当前要求行主序布局；A5 支持更多整数类型。

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TDIV(dst, src0, src1);
  TDIV<DivAlgorithm::HIGH_PRECISION>(dst, src0, src1);  // A5 Only
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TDIV(dst, src0, src1);
  TDIV<DivAlgorithm::HIGH_PRECISION>(dst, src0, src1);  // A5 Only
}
```

### PTO-AS

```text
# 自动模式
%dst = pto.tdiv %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>

# 手动模式
pto.tassign %arg0, @tile(0x1000)
pto.tassign %arg1, @tile(0x2000)
%dst = pto.tdiv %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>

# PTO 汇编形式
%dst = tdiv %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.tdiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](./tile/elementwise-tile-tile_zh.md)
- 规范页：[pto.tdiv](./tile/ops/elementwise-tile-tile/tdiv_zh.md)
