# pto.tcvt

`pto.tcvt` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

按指定舍入模式，对 tile 做逐元素类型转换；部分形式还允许显式指定饱和模式。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{cast}_{\mathrm{rmode},\mathrm{satmode}}\!\left(\mathrm{src}_{i,j}\right) $$

其中：

- `rmode` 控制舍入规则
- `satmode`（若显式给出）控制溢出时是否饱和

这条指令既覆盖 tile 内部的数值类型变化，也把“舍入 / 饱和是否显式暴露”做成了接口的一部分。

## 舍入模式

| 模式 | 行为 |
| --- | --- |
| `CAST_RINT` | 就近舍入，ties to even |
| `CAST_RZ` | 向 0 舍入 |
| `CAST_RP` | 向正无穷舍入 |
| `CAST_RM` | 向负无穷舍入 |
| `CAST_RN` | 就近舍入，ties away from zero |

## 饱和模式

| 模式 | 行为 |
| --- | --- |
| `NONE` | 不饱和，溢出时 wrap 或走目标定义行为 |
| `SAT` | 饱和到目标类型可表示范围 |

## 语法

### PTO-AS

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tcvt ins(%src {rmode = #pto.round_mode<CAST_RINT>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, WaitEvents &... events);
```

CPU 模拟器当前只实现了不显式传入 `SaturationMode` 的形式。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | 在 `dst` valid region 上逐坐标读取 |
| `%dst` | 目标 tile | 保存转换后的元素值 |
| `mode` | 舍入模式 | `CAST_RINT` / `CAST_RZ` / `CAST_RP` / `CAST_RM` / `CAST_RN` |
| `satMode` | 可选饱和模式 | `NONE` / `SAT` |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | 逐元素转换后的结果 tile |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `src` 与 `dst` 必须在 shape 和 valid region 上兼容。
- 源 / 目标类型对必须属于目标 profile 支持的集合。
- 给定类型对必须支持所选 rounding mode。
- `dst` 与 `src` 必须是不同元素类型。

## 不允许的情形

- 使用目标 profile 不支持的类型对。
- 使用该类型对不支持的 rounding mode。

## Target-Profile 限制

| 特性 | CPU Simulator | A2A3 | A5 |
| --- | :---: | :---: | :---: |
| 默认饱和形式 | Yes | Yes | Yes |
| 显式 `SaturationMode` | No | Yes | Yes |
| `f32 -> f16` | Yes | Yes | Yes |
| `f16 -> f32` | Yes | Yes | Yes |
| `f32 -> bf16` | Yes | Yes | Yes |
| `bf16 -> f32` | Yes | Yes | Yes |
| `f32 -> int32_t` | Yes | Yes | Yes |
| `int32_t -> f32` | Yes | Yes | Yes |
| `f16 -> bf16` | No | Yes | Yes |
| FP8 类型 | No | No | Yes |

## 性能

当前仓内没有把 `tcvt` 作为 tile 路径单独落成公开 cost bucket。若代码依赖具体延迟，应把它视为目标 profile 相关的 tile 类型转换路径。

## 示例

### 自动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  SrcT src;
  DstT dst;
  TCVT(dst, src, RoundMode::CAST_RINT);
}
```

### 显式饱和

```cpp
TCVT(dst, src, RoundMode::CAST_RINT, SaturationMode::SAT);
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tsubc](./tsubc_zh.md)
- 下一条指令：[pto.tsel](./tsel_zh.md)
