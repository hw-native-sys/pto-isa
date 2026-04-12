<!-- Generated from `docs/isa/tile/ops/elementwise-tile-tile/texp_zh.md` -->

# TEXP

## 指令示意图

![TEXP tile operation](../figures/isa/TEXP.svg)

## 简介

逐元素指数运算。

## 数学语义

对有效区域内的每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \exp(\mathrm{src}_{i,j}) $$

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = texp %src : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.texp %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.texp ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1（SSA）

```text
%dst = pto.texp %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.texp ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <auto PrecisionType = ExpAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TEXP(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

`PrecisionType` 可指定以下取值：

- `ExpAlgorithm::DEFAULT`：普通算法，速度更快但精度较低。
- `ExpAlgorithm::HIGH_PRECISION`：高精度算法，速度较慢。

## 约束

- **实现检查（NPU）**：
    - `TileData::DType` 必须是 `float` 或 `half`；
    - Tile 位置必须是向量（`TileData::Loc == TileType::Vec`）；
    - 静态有效边界：`TileData::ValidRow <= TileData::Rows` 且 `TileData::ValidCol <= TileData::Cols`；
    - 运行时：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`；
    - Tile 布局必须是行主序（`TileData::isRowMajor`）。
- **有效区域**：
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域。
- **高精度算法**：
    - 仅 A5 支持，A3 会忽略 `PrecisionType` 选项。

## 示例

### 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TEXP(dst, src);
  TEXP<ExpAlgorithm::HIGH_PRECISION>(dst, src);  // 仅 A5
}
```

### 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TEXP(dst, src);
}
```

## 汇编示例

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.texp %src : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.texp %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = texp %src : !pto.tile<...>
# AS Level 2 (DPS)
pto.texp ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
