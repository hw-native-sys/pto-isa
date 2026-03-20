# TRECIP

## 指令示意图

![TRECIP tile operation](../figures/isa/TRECIP.svg)

## 简介

Tile 的逐元素倒数。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \frac{1}{\mathrm{src}_{i,j}} $$

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = trecip %src : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trecip %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trecip ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TRECIP(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## 约束

- **实现检查 (NPU)**:
    - `TileData::DType` 必须是以下之一：`float` 或 `half`。
    - Tile 位置必须是向量（`TileData::Loc == TileType::Vec`);
    - 静态有效边界：`TileData::ValidRow <= TileData::Rows` 且 `TileData::ValidCol <= TileData::Cols`。
    - 运行时：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`。
    - Tile 布局必须是行主序（`TileData::isRowMajor`）。
    - A3 的 TRECIP 指令不支持将源 Tile 和目标 Tile 设置为相同的内存。
- **有效区域**:
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域。
- **域 / NaN**:
    - 除零行为由目标定义；CPU 模拟器在调试构建中会断言。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TRECIP(out, x);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.trecip %src : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trecip %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = trecip %src : !pto.tile<...>
# AS Level 2 (DPS)
pto.trecip ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

