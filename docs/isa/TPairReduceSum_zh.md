# TPAIRREDUCESUM

## 指令示意图

![TPAIRREDUCESUM](../figures/isa/TPairReduceSum.svg)

## 简介

对源 Tile 中每两个相邻元素求和（Pair-Reduction），并将结果写入目标 Tile 的下半部分。该指令使用 `vcpadd` 向量对加法原语，对每个对索引 `k` 计算 `src0[i, 2k] + src0[i, 2k+1]`，将归约后的值打包存储到每行的位置 `0 … ⌈validCols/2⌉−1`。目标行上半部分（位置 `⌈validCols/2⌉ … validCols−1`）的元素填充为 **0**。无效元素（有效区域之外的位置）也视为 **0**。

## 数学语义

给定源 Tile `src0` 和目标 Tile `dst`，两者具有相同的有效形状 `(validRows, validCols)`，对于每行 `i` 和对索引 `k`：

$$ \mathrm{dst}_{i,k} = \mathrm{src0}_{i, 2k} + \mathrm{src0}_{i, 2k+1}, \quad 0 \le k < \left\lceil \frac{\mathrm{validCols}}{2} \right\rceil $$

`dst` 每行中位置 `⌈validCols/2⌉ … validCols−1` 的元素填充为 0。无效元素（有效区域之外的位置）也视为 0。

其中 `validRows = dst.GetValidRow()` 且 `validCols = dst.GetValidCol()`。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = tpairreducesum %src : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tpairreducesum %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tpairreducesum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename... WaitEvents>
PTO_INST RecordEvent TPAIRREDUCESUM(TileDataDst &dst, TileDataSrc0 &src0, WaitEvents &...events);
```

## 约束

- **实现检查 (A2A3 & A5)**:
    - `TileData::DType` 必须为 `float` 或 `half`。
    - Tile 布局必须是行主序（`TileData::isRowMajor`）。
    - `dst` 和 `src0` 必须具有相同的 `DType`。
    - `src0` 必须与 `dst` 具有相同的有效形状（`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`）。
- **有效区域**:
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域。
    - 每个目标行的下半部分（位置 `0 … ⌈validCols/2⌉−1`）存储对归约结果；上半部分（位置 `⌈validCols/2⌉ … validCols−1`）填充为 0。有效区域之外的无效元素也视为 0。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
    using TileDst = Tile<TileType::Vec, float, 16, 32>;
    using TileSrc = Tile<TileType::Vec, float, 16, 64>;
    TileDst dst(16, 32);
    TileSrc src0(16, 32);

    TPAIRREDUCESUM(dst, src0);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
    using TileDst = Tile<TileType::Vec, half, 16, 64, BLayout::RowMajor, 16, 64>;
    using TileSrc = Tile<TileType::Vec, half, 16, 128, BLayout::RowMajor, 16, 128>;
    TileDst dst;
    TileSrc src0;

    TASSIGN(src0, 0x1000);
    TASSIGN(dst, 0x2000);

    TPAIRREDUCESUM(dst, src0);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tpairreducesum %src : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %src,  @tile(0x1000)
# pto.tassign %dst,  @tile(0x2000)
%dst = pto.tpairreducesum %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tpairreducesum %src : !pto.tile<...>
# AS Level 2 (DPS)
pto.tpairreducesum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关指令

- [TDeInterleave](TDEINTERLEAVE_zh.md) - 反交织将偶/奇位置拆分；TPAIRREDUCESUM 将相邻对求和。
- [TInterleave](TINTERLEAVE_zh.md) - 交织将两个源的元素交替组合为统一流。
