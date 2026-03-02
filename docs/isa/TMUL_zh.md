# TMUL

## 指令示意图

![TMUL tile operation](../figures/isa/TMUL.svg)

## 简介

两个 Tile 的逐元素乘法。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot \mathrm{src1}_{i,j} $$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = tmul %src0, %src1 : !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tmul %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TMUL(TileData& dst, TileData& src0, TileData& src1, WaitEvents&... events);
```

## 约束

- **实现检查 (A2A3)**:
  - `TileData::DType` must be one of: `int32_t`, `int16_t`, `half`, `float`.
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
  - Tile 布局 must be row-major (`TileData::isRowMajor`).
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **实现检查 (A5)**:
  - `TileData::DType` must be one of: `int32_t`, `uint32_t`, `float`, `int16_t`, `uint16_t`, `half`.
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
  - Tile 布局 must be row-major (`TileData::isRowMajor`).
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **有效区域**:
  - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain; .

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TMUL(dst, src0, src1);
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
  TMUL(dst, src0, src1);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tmul %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tmul %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tmul %src0, %src1 : !pto.tile<...>
# IR Level 2 (DPS)
pto.tmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

