# TTRI

## 指令示意图

![TTRI tile operation](../figures/isa/TTRI.svg)

## 简介

生成三角（下/上）掩码 Tile。

## 数学语义

设 `R = dst.GetValidRow()`，`C = dst.GetValidCol()`，`d = diagonal`。

下三角（`isUpperOrLower=0`）概念上产生：

$$
\mathrm{dst}_{i,j} = \begin{cases}1 & j \le i + d \\\\ 0 & \text{否则}\end{cases}
$$

上三角（`isUpperOrLower=1`）概念上产生：

$$
\mathrm{dst}_{i,j} = \begin{cases}0 & j < i + d \\\\ 1 & \text{否则}\end{cases}
$$

## 汇编语法

### AS Level 1（SSA）

```text
%dst = pto.ttri %diag : i32 -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.ttri ins(%diag : i32) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileData, int isUpperOrLower, typename... WaitEvents>
PTO_INST RecordEvent TTRI(TileData &dst, int diagonal, WaitEvents &... events);
```

## 约束

- `isUpperOrLower` 必须是 `0`（下三角）或 `1`（上三角）。
- **实现检查 (A2A3)**:
    - 目标 Tile 必须是行主序（`isRowMajor`），由 `static_assert` 强制执行。
    - 支持的元素类型：`int32_t`、`int`、`int16_t`、`uint32_t`、`uint16_t`、`half`、`float16_t`、`float`、`float32_t`。
- **实现检查 (A5)**:
    - 支持的元素类型：`int32_t`、`int16_t`、`int8_t`、`uint32_t`、`uint16_t`、`uint8_t`、`half`、`float16_t`、`float32_t`、`bfloat16_t`。
    - 下三角（`upperOrLower == 0`）和上三角（`upperOrLower == 1`）由 `if constexpr` 分支区分。
- 有效区域通过 `dst.GetValidRow()` / `dst.GetValidCol()` 获取。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_lower() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT dst;
  TASSIGN(dst, 0x1000);
  TTRI<0>(dst, /*diagonal=*/0);   // 下三角
}

void example_upper() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT dst;
  TASSIGN(dst, 0x1000);
  TTRI<1>(dst, /*diagonal=*/-1);  // 上三角
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.ttri {isUpperOrLower = 0} : i32 -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# pto.tassign %arg0, @tile(0x1000)
%dst = pto.ttri %diag : i32 -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = pto.ttri %diag : i32 -> !pto.tile<...>
# AS Level 2 (DPS)
pto.ttri ins(%diag : i32) outs(%dst : !pto.tile_buf<...>)
```
