# pto.tpartmin

`pto.tpartmin` 属于[不规则与复杂指令](../../irregular-and-complex_zh.md)集。

## 概述

在目标有效区域内执行逐元素最小值选择。若某个位置上 `src0` 和 `src1` 都有效，则结果为 `min(src0, src1)`；若只有一个输入在该位置有效，则结果直接取该输入的值。其余有效区域不匹配的情况由具体实现定义。

对目标有效区域内的每个元素 `(i, j)`：若两个输入都有定义，则 $\mathrm{dst}_{i,j} = \min(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j})$；若仅 src0 有定义，则 $\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j}$；若仅 src1 有定义，则 $\mathrm{dst}_{i,j} = \mathrm{src1}_{i,j}$。

## 语法

### PTO-AS

```text
%dst = tpartmin %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tpartmin %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tpartmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `dst` | 输出 | 目标 Tile |
| `src0` | 输入 | 第一个源 Tile |
| `src1` | 输入 | 第二个源 Tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 逐元素最小值结果 |

## 副作用

该指令可能会写入 Tile 的有效区域标记。

## 约束

- `dst`、`src0` 和 `src1` 的元素类型必须一致。
- 目标有效区域定义结果的计算范围。
- 对目标有效区域内的每个元素：若两个输入都有效，则执行逐元素最小值运算；若只有一个输入有效，则结果直接取该输入的值。
- 若 `dst` 的有效区域为零，指令直接返回。
- 支持的部分有效区域模式要求至少有一个源 Tile 的有效区域与 `dst` 完全一致，另一个源 Tile 的有效区域在两个维度上都不能超过 `dst`。
- 上述范围之外的有效区域组合，其行为均由具体实现定义。
- A2A3：支持 `int32_t`、`int16_t`、`half`、`float`，且 `dst`/`src0`/`src1` 必须全部为行主序。
- A5：支持 `int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。

## 异常与非法情形

- 输入 Tile 类型不一致时行为未定义。
- 超出支持的有效区域模式组合时行为由实现定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TPARTMIN(dst, src0, src1);
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
  TPARTMIN(dst, src0, src1);
}
```

### PTO-AS

```text
%dst = pto.tpartmin %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tpartmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[不规则与复杂指令](../../irregular-and-complex_zh.md)
