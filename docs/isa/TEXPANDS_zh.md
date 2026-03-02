# TEXPANDS

## 指令示意图

![TEXPANDS tile operation](../figures/isa/TEXPANDS.svg)

## 简介

将标量广播到目标 Tile 中。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{scalar} $$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = texpands %scalar : f32, !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.texpands ins(%scalar : dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TEXPANDS(TileData& dst, typename TileData::DType scalar, WaitEvents&... events);
```

## 约束

- **实现检查 (A2A3)**:
  - `TileData::DType` must be one of: `int32_t`, `int16_t`, `half`, `float`.
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Tile 布局 must be row-major (`TileData::isRowMajor`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
- **实现检查 (A5)**:
  - `TileData::DType` must be one of: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `half`, `float`.
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Tile 布局 must be row-major (`TileData::isRowMajor`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
- **有效区域**:
  - The op fills `dst` over `dst.GetValidRow()` / `dst.GetValidCol()`.

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT dst;
  TEXPANDS(dst, 0.0f);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT dst;
  TASSIGN(dst, 0x1000);
  TEXPANDS(dst, 0.0f);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = texpands %scalar : f32, !pto.tile<...>
# IR Level 2 (DPS)
pto.texpands ins(%scalar : dtype) outs(%dst : !pto.tile_buf<...>)
```

