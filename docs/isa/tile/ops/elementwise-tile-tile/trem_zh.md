# pto.trem

`pto.trem` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对两个 tile 做逐元素余数运算。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$\mathrm{dst}_{i,j} = \mathrm{remainder}(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) = \mathrm{src0}_{i,j} - \mathrm{floor}(\frac{\mathrm{src0}_{i,j}}{\mathrm{src1}_{i,j}}) \times \mathrm{src1}_{i,j}$$

结果符号会被修正为与除数（`src1`）的符号相同。

!!! note "注意"
    这与 `TFMOD` 不同，`TFMOD` 的结果符号与被除数（`src0`）相同。

## 语法

### PTO-AS

```text
%dst = trem %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trem ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <auto PrecisionType = RemAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TREM(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入

- `%src0`：被除数 tile
- `%src1`：除数 tile
- `%tmp`：A2A3 路径需要的临时 tile
- `%dst`：目标 tile

## 预期输出

- `%dst`：逐元素余数结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

!!! warning "约束"
    - 除零行为由目标 profile 定义；CPU 模拟器在调试构建下会断言。
    - 迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
    - A2A3 对 `tmp` 有容量和类型要求。

## 异常与非法情形

!!! danger "异常与非法情形"
    - 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- `TileData::DType` 必须是以下之一：`float`、`float32_t`、`int32_t`
- `dst`、`src0`、`src1` 必须使用相同元素类型
- 三者都必须是行主序向量 tile
- 运行时要求：`src0`、`src1` 和 `dst` 具有相同的 `validRow` 和 `validCol`
- `tmp` 需要至少 2 行且列数不少于 `dst`；第 0 行用于中间结果，第 1 行用于比较掩码
- 对 `int32_t` 输入，还要求数值落在 `[-2^24, 2^24]`
- `PrecisionType` 高精度选项在 A2A3 上会被忽略

### A5

- `dst`、`src0`、`src1` 必须使用相同元素类型
- 支持类型：`float`、`int32_t`、`uint32_t`、`half`、`int16_t`、`uint16_t`
- 三者必须是向量 tile
- 静态 valid 边界必须合法
- `tmp` 在 A5 形参存在，但不参与约束和实现
- 高精度算法仅在 A5 上对 `float` 类型有效

## 性能

### A2A3

英文页当前把 `TREM` 归到和二元算术同一类估算口径：

| 指标 | FP | INT |
| --- | --- | --- |
| 启动时延 | 14 | 14 |
| 完成时延 | 19 | 17 |
| 每次 repeat 吞吐 | 2 | 2 |
| 流水间隔 | 18 | 18 |

### A5

当前手册未单列 `trem` 的独立周期表，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 2, 16>;
  TileT dst, src0, src1;
  TmpT tmp;
  TREM(dst, src0, src1, tmp);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tneg](./tneg_zh.md)
- 下一条指令：[pto.tfmod](./tfmod_zh.md)
