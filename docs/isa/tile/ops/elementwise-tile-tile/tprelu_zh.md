# pto.tprelu

`pto.tprelu` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

带逐元素斜率 tile 的参数化 ReLU。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = (\mathrm{src0}_{i,j} > 0) ? \mathrm{src0}_{i,j} : (\mathrm{src0}_{i,j} \cdot \mathrm{src1}_{i,j}) $$

这里 `src0` 是值 tile，`src1` 是每个元素自己的斜率 tile，因此它比 `tlrelu` 更细粒度。

## 语法

### PTO-AS

```text
%dst = tprelu %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tprelu ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TPRELU(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入

- `%src0`：值 tile
- `%src1`：逐元素斜率 tile
- `%tmp`：A3 路径所需的临时 tile
- `%dst`：目标 tile

## 预期输出

- `%dst`：逐元素 PReLU 结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- A3 需要临时空间，而 A5 不使用 `tmp`。
- A3 上，两个源 tile、目标 tile 和临时空间不能互相重叠。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

- A3 需要 `tmp` 参与实现。
- A5 保留相同的 PTO 可见语义，但不要求 `tmp` 真正参与硬件路径。

## 性能

### A2A3

英文页当前把 `TPRELU` 归到与二元算术同一类模型：

| 指标 | FP | INT |
| --- | --- | --- |
| 启动时延 | 14 | 14 |
| 完成时延 | 19 | 17 |
| 每次 repeat 吞吐 | 2 | 2 |
| 流水间隔 | 18 | 18 |

### A5

当前手册未单列 `tprelu` 的独立周期表，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, slope, out, tmp;
  TPRELU(out, x, slope, tmp);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.trecip](./trecip_zh.md)
- 下一条指令：[pto.taddc](./taddc_zh.md)
