# pto.tlog

![TLOG tile operation](../figures/isa/TLOG.svg)

`pto.tlog` 属于[逐元素 Tile-Tile](./tile/elementwise-tile-tile_zh.md)指令集。

## 概述

对 tile 做逐元素自然对数，结果写入目标 tile。迭代域由目标 tile 的 valid region 决定。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \log(\mathrm{src}_{i,j}) $$

它是 tile 路径上的一元对数操作，用于归一化、损失计算前处理和指数域反变换。对 `log(<=0)` 的域外情况，行为由目标 profile 定义。

## 语法

### PTO-AS

```text
%dst = tlog %src : !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tlog %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tlog ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <auto PrecisionType = LogAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TLOG(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

`PrecisionType` 可选：

- `LogAlgorithm::DEFAULT`：普通算法，速度快但精度较低。
- `LogAlgorithm::HIGH_PRECISION`：高精度算法，速度较慢。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | 输入 tile |
| `%dst` | 目标 tile | 接收逐元素对数结果 |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | `dst` valid region 内的每个元素都等于 `log(src)` |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 支持类型当前是 `float` / `half`。
- tile 必须是行主序向量 tile。
- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 对 `log(<=0)` 的域外情况，行为由目标 profile 定义。
- 高精度算法只在 A5 有效。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `float` | Simulated | Supported | Supported |
| `half` | Simulated | Supported | Supported |
| 布局 | Any | RowMajor only | RowMajor only |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TLOG(out, x);
  TLOG<LogAlgorithm::HIGH_PRECISION>(out, x);  // A5 Only
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
  TLOG(dst, src);
}
```

### PTO-AS

```text
# 自动模式
%dst = pto.tlog %src : !pto.tile<...> -> !pto.tile<...>

# 手动模式
pto.tassign %arg0, @tile(0x1000)
pto.tassign %arg1, @tile(0x2000)
%dst = pto.tlog %src : !pto.tile<...> -> !pto.tile<...>

# PTO 汇编形式
%dst = tlog %src : !pto.tile<...>
# AS Level 2 (DPS)
pto.tlog ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](./tile/elementwise-tile-tile_zh.md)
- 规范页：[pto.tlog](./tile/ops/elementwise-tile-tile/tlog_zh.md)
