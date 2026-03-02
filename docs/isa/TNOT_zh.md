# TNOT

## 指令示意图

![TNOT tile operation](../figures/isa/TNOT.svg)

## 简介

Tile 的逐元素按位取反。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \sim\mathrm{src}_{i,j} $$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = tnot %src : !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tnot %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tnot ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TNOT(TileData& dst, TileData& src, WaitEvents&... events);
```

## 约束

- **实现检查 (A2A3)**:
  - `TileData::DType` must be one of: `int16_t`, `uint16_t`.
  - Tile 布局 must be row-major (`TileData::isRowMajor`).
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
  - Runtime: `src` and `dst` tiles should have the same `validRow/validCol`.
- **实现检查 (A5)**:
  - `TileData::DType` must be one of: `uint32_t`, `int32_t`, `uint16_t`, `int16_t`, `uint8_t`,  `int8_t`.
  - Tile 布局 must be row-major (`TileData::isRowMajor`).
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
  - Runtime: `src` and `dst` tiles should have the same `validRow/validCol`.
- **有效区域**:
  - The op uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain; `src/dst` are assumed to be compatible (not validated by explicit runtime checks in this op).

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT x, out;
  TNOT(out, x);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tnot %src : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tnot %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tnot %src : !pto.tile<...>
# IR Level 2 (DPS)
pto.tnot ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

