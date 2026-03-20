# TPARTADD

## 指令示意图

![TPARTADD tile operation](../figures/isa/TPARTADD.svg)

## 简介

部分逐元素加法，对不匹配的有效区域具有实现定义的处理方式。

## 数学语义

对每个元素 `(i, j)` 在目标有效区域中：

$$
\mathrm{dst}_{i,j} =
\b\begin{cases}
\mathrm{src0}_{i,j} + \mathrm{src1}_{i,j} & \text{如果两个输入在 } (i,j) \text{ 处有定义} \text{ 处都有定义} \\
\mathrm{src0}_{i,j} & \text{如果只有 src0 在 } (i,j) \text{ 处有定义} \text{ 处都有定义} \\
\mathrm{src1}_{i,j} & \text{如果只有 src1 在 } (i,j) \text{ 处有定义}
\end{cases}
$$

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = tpartadd %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tpartadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tpartadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## 约束

- **实现检查 (A2A3)**:
    - `dst`/`src0`/`src1` 元素类型必须相同，且必须是以下之一：`int32_t`、`int16_t`、`half`、`float`。
    - 三个 Tile 都必须是行主序（`isRowMajor`）。
    - 运行时：若 `dst.GetValidRow() == 0` 或 `dst.GetValidCol() == 0`，操作提前返回。
    - 运行时：实现要求至少一个输入的有效区域与 `dst` 的有效区域匹配，另一个输入的有效区域不大于 `dst` 的有效区域（否则断言失败）。
- **实现检查 (A5)**:
    - `dst`/`src0`/`src1` 元素类型必须相同，且必须是以下之一：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`、`half`、`float`、`bfloat16_t`。
    - 运行时：若 `dst` 的有效区域为零，操作提前返回。
    - 仅处理特定的部分有效性模式（例如，一个源与 `dst` 相等，另一个源在有效行或有效列上更小）；其他模式不受支持（由目标定义行为）。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TPARTADD(dst, src0, src1);
}
```

### 手动（Manual）

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

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tpartadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tpartadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tpartadd %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tpartadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

