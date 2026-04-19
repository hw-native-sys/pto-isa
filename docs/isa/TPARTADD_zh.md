# pto.tpartadd

`pto.tpartadd` 属于[不规则及复杂运算](./tile/ops/irregular-and-complex/tpartadd_zh.md)指令集。

## 概述

在目标有效区域内执行逐元素加法。若某个位置上 `src0` 和 `src1` 都有效，则结果为两者之和；若只有一个输入在该位置有效，则结果直接取该输入的值。

## 机制

对目标有效区域内的每个元素 `(i, j)`：

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src0}_{i,j} + \mathrm{src1}_{i,j} & \text{若两个输入在 } (i,j) \text{ 处均有定义} \\
\mathrm{src0}_{i,j} & \text{若仅 src0 在 } (i,j) \text{ 处有定义} \\
\mathrm{src1}_{i,j} & \text{若仅 src1 在 } (i,j) \text{ 处有定义}
\end{cases}
$$

## 语法

### PTO-AS

```text
%dst = tpartadd %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tpartadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tpartadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src0` | 输入 | 源 tile 0 |
| `src1` | 输入 | 源 tile 1 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | 输出 | 部分加法结果，类型与输入一致 |

## 副作用

`dst` 的有效区域定义结果的计算范围。若 `dst` 的有效区域为零，指令直接返回。

## 约束

- `dst`、`src0` 和 `src1` 的元素类型必须一致
- 目标有效区域定义结果的计算范围
- 对目标有效区域内的每个元素：
    - 若两个输入都有效，则执行逐元素加法
    - 若只有一个输入有效，则结果直接取该输入的值
- 若 `dst` 的有效区域为零，指令直接返回
- 支持的部分有效区域模式要求至少有一个源 Tile 的有效区域与 `dst` 完全一致，另一个源 Tile 的有效区域在两个维度上都不能超过 `dst`
- 上述范围之外的有效区域组合，其行为均由具体实现定义
- A2A3：`dst`、`src0` 和 `src1` 必须全部为行主序（`isRowMajor`）

## 异常与非法情形

- 运行时检查失败时，行为由具体实现定义

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 支持的元素类型 | 全部 | `int32_t`、`int16_t`、`half`、`float` | `uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`、`half`、`float`、`bfloat16_t` |
| 行主序要求 | 无 | 是 | 无 |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TPARTADD(dst, src0, src1);
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
  TPARTADD(dst, src0, src1);
}
```

### PTO-AS

```text
# 自动模式
%dst = pto.tpartadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

## 相关页面

- 指令集总览：[不规则及复杂运算](./tile/ops/irregular-and-complex/tpartadd_zh.md)
