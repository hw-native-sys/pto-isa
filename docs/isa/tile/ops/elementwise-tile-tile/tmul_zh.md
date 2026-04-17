# pto.tmul

`pto.tmul` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对两个 tile 做逐元素乘法。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot \mathrm{src1}_{i,j} $$

和其他二元逐元素 tile 指令一样，迭代域由目标 tile 决定，源 tile 域外值不应被依赖。

## 语法

### PTO-AS

```text
%dst = tmul %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tmul %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src0` | 左 tile | 第一个源 tile |
| `%src1` | 右 tile | 第二个源 tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | `dst` valid region 内的每个元素都等于 `src0 * src1` |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- A2A3 与 A5 当前都要求行主序向量 tile。
- 静态 valid 边界必须合法。
- 运行时通常要求 `src0`、`src1` 与 `dst` 的 `validRow/validCol` 一致。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- 支持类型：`int32_t`、`int16_t`、`half`、`float`
- tile 必须是行主序向量 tile

### A5

- 支持类型：`int32_t`、`uint32_t`、`float`、`int16_t`、`uint16_t`、`half`
- tile 必须是行主序向量 tile

## 性能

### A2A3

`TMUL` 的吞吐模型是：

| 指标 | FP | INT |
| --- | --- | --- |
| 启动时延 | 14 | 14 |
| 完成时延 | 20 | 18 |
| 每次 repeat 吞吐 | 2 | 2 |
| 流水间隔 | 18 | 18 |

连续路径下，`R = validRow × validCol / 8` 可作为常见估算。

### A5

当前手册未单列 `tmul` 的独立周期表，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TMUL(dst, src0, src1);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tsub](./tsub_zh.md)
- 下一条指令：[pto.tdiv](./tdiv_zh.md)
