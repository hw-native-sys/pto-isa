# pto.trecip

![TRECIP tile operation](../figures/isa/TRECIP.svg)

`pto.trecip` 属于[逐元素 Tile-Tile](./tile/elementwise-tile-tile_zh.md)指令集。

## 概述

对 tile 做逐元素倒数，结果写入目标 tile。迭代域由目标 tile 的 valid region 决定。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \frac{1}{\mathrm{src}_{i,j}} $$

它适合在后续仍要与其他 tile 做乘法组合时，替代显式除法。除零行为由目标定义。

## 语法

### PTO-AS

```text
%dst = trecip %src : !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.trecip %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.trecip ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <auto PrecisionType = RecipAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TRECIP(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

`PrecisionType` 可选：

- `RecipAlgorithm::DEFAULT`：普通算法，速度快但精度较低。
- `RecipAlgorithm::HIGH_PRECISION`：高精度算法，速度较慢。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | 输入 tile |
| `%dst` | 目标 tile | 接收逐元素倒数结果 |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | `dst` valid region 内的每个元素都等于 `1 / src` |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 除零行为由目标定义；CPU 模拟器在调试构建下会断言。
- 高精度算法只在 A5 有效。
- A3 不支持源 tile 与目标 tile 绑定到同一片内存。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `float` | Simulated | Supported | Supported |
| `half` | Simulated | Supported | Supported |
| 布局 | Any | RowMajor only | RowMajor only |
| 同址操作 | No | No | No |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TRECIP(out, x);
  TRECIP<RecipAlgorithm::HIGH_PRECISION>(out, x);
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
  TRECIP(dst, src);
}
```

### PTO-AS

```text
# 自动模式
%dst = pto.trecip %src : !pto.tile<...> -> !pto.tile<...>

# 手动模式
pto.tassign %arg0, @tile(0x1000)
pto.tassign %arg1, @tile(0x2000)
%dst = pto.trecip %src : !pto.tile<...> -> !pto.tile<...>

# PTO 汇编形式
%dst = trecip %src : !pto.tile<...>
# AS Level 2 (DPS)
pto.trecip ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](./tile/elementwise-tile-tile_zh.md)
- 规范页：[pto.trecip](./tile/ops/elementwise-tile-tile/trecip_zh.md)
