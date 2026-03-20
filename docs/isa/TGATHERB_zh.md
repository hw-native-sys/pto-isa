# TGATHERB

## 指令示意图

![TGATHERB tile operation](../figures/isa/TGATHERB.svg)

## 简介

使用字节偏移量收集元素。

## 数学语义

对有效区域内的每个元素：

$$ \mathrm{dst}_{i,j} = *\left(\mathrm{srcBase} + \mathrm{offset}_{i,j}\right) $$

精确的边界行为由实现定义。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = tgatherb %src, %offsets : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tgatherb %src, %offsets : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tgatherb ins(%src, %offsets : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset, typename... WaitEvents>
PTO_INST RecordEvent TGATHERB(TileDataDst &dst, TileDataSrc &src, TileDataOffset &offset, WaitEvents &... events);
```

## 约束

- **实现检查 (A2A3)**:
    - 目标布局必须是行主序（`TileDataDst::isRowMajor`）。
    - 目标元素大小必须是 `1`、`2` 或 `4` 字节（通过辅助函数中的 `static_assert` 强制执行）。
    - `SrcTileData::DType`/`DstTileData::DType` 必须是 `int8_t` 或 `uint8_t` 或 `int16_t` 或 `uint16_t` 或 `int32_t` 或 `uint32_t` 或 `half` 或 `bfloat16_t` 或 `float`。
- **实现检查 (A5)**:
    - 目标元素大小必须是 `1`、`2` 或 `4` 字节。
    - `SrcTileData::DType`/`DstTileData::DType` 必须是 `int8_t` 或 `uint8_t` 或 `int16_t` 或 `uint16_t` 或 `int32_t` 或 `uint32_t` 或 `half` 或 `bfloat16_t` 或 `float`。
- **偏移量解释**:
    - 偏移量被实现解释为 `uint32_t` 值（字节偏移量）。
    - 偏移量边界不通过显式运行时断言进行验证；超出范围的偏移量由目标定义。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, uint8_t, 1, 256>;
  using OffT = Tile<TileType::Vec, uint32_t, 1, 256>;
  using DstT = Tile<TileType::Vec, uint8_t, 1, 256>;
  SrcT src;
  OffT off;
  DstT dst;
  TGATHERB(dst, src, off);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, uint8_t, 1, 256>;
  using OffT = Tile<TileType::Vec, uint32_t, 1, 256>;
  using DstT = Tile<TileType::Vec, uint8_t, 1, 256>;
  SrcT src;
  OffT off;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(off, 0x2000);
  TASSIGN(dst, 0x3000);
  TGATHERB(dst, src, off);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tgatherb %src, %offsets : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tgatherb %src, %offsets : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tgatherb %src, %offsets : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tgatherb ins(%src, %offsets : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

