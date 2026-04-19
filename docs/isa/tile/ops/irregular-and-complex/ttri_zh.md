# pto.ttri

`pto.ttri` 属于[不规则与复杂指令](../../irregular-and-complex_zh.md)集。

## 概述

TTRI 生成一个三角掩码 Tile。它不读取源 Tile，而是根据目标 Tile 的有效形状和 `diagonal` 参数直接在 `dst` 里写出上三角或下三角的 0/1 模式，常用于注意力 mask、三角区域约束或后续按位/乘法掩码场景。

设 `R = dst.GetValidRow()`、`C = dst.GetValidCol()`，`d = diagonal`。当 `isUpperOrLower = 0`（下三角）时，若 `j ≤ i + d` 则输出 1，否则输出 0；当 `isUpperOrLower = 1`（上三角）时，若 `j < i + d` 则输出 0，否则输出 1。`diagonal = 0` 表示主对角线；正值向右扩展保留区域，负值则收缩。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.ttri %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.ttri ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, int isUpperOrLower, typename... WaitEvents>
PTO_INST RecordEvent TTRI(TileData &dst, int diagonal, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `dst` | 输出 | 目标 Tile，接收生成的三角掩码 |
| `diagonal` | 标量输入 | 对角线偏移量 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 包含上三角或下三角掩码的 Tile |

## 副作用

该指令可能会写入 Tile 的有效区域标记。

## 约束

- `isUpperOrLower` 只能是 0（下三角）或 1（上三角）。
- `dst` 必须是 row-major Tile。
- CPU/A2A3 支持的数据类型：`int32_t`、`int16_t`、`uint32_t`、`uint16_t`、`half`、`float` 等。
- A5 额外支持 `int8_t`、`uint8_t`、`bfloat16_t`。

## 异常与非法情形

- `isUpperOrLower` 值不为 0 或 1 时行为未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |

## 示例

### C++ 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using MaskT = Tile<TileType::Vec, float, 16, 16>;
  MaskT mask;
  TTRI<MaskT, 0>(mask, 0);  // lower triangular
}
```

## 相关页面

- 指令集总览：[不规则与复杂指令](../../irregular-and-complex_zh.md)
