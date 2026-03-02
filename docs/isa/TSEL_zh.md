# TSEL

## 指令示意图

![TSEL tile operation](../figures/isa/TSEL.svg)

## 简介

使用掩码 Tile 在两个 Tile 之间进行选择（逐元素选择）。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src0}_{i,j} & \text{if } \mathrm{mask}_{i,j}\ \text{is true} \\
\mathrm{src1}_{i,j} & \text{otherwise}
\end{cases}
$$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = tsel %mask, %src0, %src1 : !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename MaskTile, typename... WaitEvents>
PTO_INST RecordEvent TSEL(TileData& dst, MaskTile& selMask, TileData& src0, TileData& src1, WaitEvents&... events);
```

## 约束

- **实现检查 (A2A3)**:
  - `sizeof(TileData::DType)` must be `2` or `4` bytes.
  - `TileData::DType` must be `int16_t` or `uint16_t` or `int32_t` or `uint32_t` or `half` or `bfloat16_t` or `float`.
  - No explicit assertions are enforced on the mask tile type/shape; mask encoding is target-defined.
  - The implementation uses `dst.GetValidRow()` / `dst.GetValidCol()` for the selection domain.
- **实现检查 (A5)**:
  - `sizeof(TileData::DType)` must be `2` or `4` bytes.
  - `TileData::DType` must be `int16_t` or `uint16_t` or `int32_t` or `uint32_t` or `half` or `bfloat16_t` or `float`.
  - No explicit `static_assert`/`PTO_ASSERT` checks are enforced by `TSEL_IMPL`.
  - The implementation uses `dst.GetValidRow()` / `dst.GetValidCol()` for the selection domain.
- **Mask encoding**:
  - The mask tile is interpreted as packed predicate bits in a target-defined layout.

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  TileT src0, src1, dst;
  MaskT mask(16, 2);
  TSEL(dst, mask, src0, src1);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  TileT src0, src1, dst;
  MaskT mask(16, 2);
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TASSIGN(mask, 0x4000);
  TSEL(dst, mask, src0, src1);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tsel %mask, %src0, %src1 : !pto.tile<...>
# IR Level 2 (DPS)
pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

