# pto.trem

`pto.trem` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对两个 tile 做逐元素余数运算。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \bmod \mathrm{src1}_{i,j} $$

它和 `tfmod` 的差别在于这里强调整数式 remainder 语义，而不是浮点 `fmod`。

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
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
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

- 除零行为由目标 profile 定义；CPU 模拟器在调试构建下会断言。
- 迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- A2A3 对 `tmp` 有容量和类型要求。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- `dst`、`src0`、`src1` 必须使用相同元素类型
- 支持类型：`float`、`int32_t`
- 三者都必须是行主序向量 tile
- 运行时要求：`dst.GetValidRow() == src0.GetValidRow() == src1.GetValidRow() > 0`，且列数也一致
- `tmp` 需要至少 1 行，且列数不少于 `dst`
- 对 `int32_t` 输入，还要求数值落在 `[-2^24, 2^24]`

### A5

- `dst`、`src0`、`src1` 必须使用相同元素类型
- 支持类型：`float`、`int32_t`、`uint32_t`、`half`、`int16_t`、`uint16_t`
- 三者必须是向量 tile
- 静态 valid 边界必须合法
- `tmp` 在 A5 形参存在，但不参与约束和实现

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
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT out, a, b;
  Tile<TileType::Vec, int32_t, 16, 16> tmp;
  TREM(out, a, b, tmp);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tneg](./tneg_zh.md)
- 下一条指令：[pto.tfmod](./tfmod_zh.md)
