# TMRGSORT

## 指令示意图

![TMRGSORT tile operation](../figures/isa/TMRGSORT.svg)

## 简介

用于多个已排序列表的归并排序（实现定义的元素格式和布局）。

## 数学语义

将多个已排序的输入列表归并到 `dst` 中。排序规则、元素格式（如值/索引对）以及已执行计数的含义取决于具体实现。

$$ \mathrm{dst} = \mathrm{merge}(\mathrm{src}_0, \mathrm{src}_1, \ldots) $$

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式（概念性）：

```text
%dst, %executed = tmrgsort %src0, %src1 {exhausted = false}
    : !pto.tile<...>, !pto.tile<...> -> (!pto.tile<...>, vector<4xi16>)
```

### AS Level 1（SSA）

```text
%dst = pto.tmrgsort %src, %blockLen : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst, %executed = pto.tmrgsort %src0, %src1, %src2, %src3 {exhausted = false}
 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, vector<4xi16>)
```

### AS Level 2（DPS）

```text
pto.tmrgsort ins(%src, %blockLen : !pto.tile_buf<...>, dtype)  outs(%dst : !pto.tile_buf<...>)
pto.tmrgsort ins(%src0, %src1, %src2, %src3 {exhausted = false} : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
outs(%dst, %executed : !pto.tile_buf<...>, vector<4xi16>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, typename Src3TileData, bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp, Src0TileData &src0, Src1TileData &src1, Src2TileData &src2, Src3TileData &src3, WaitEvents &... events);

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp, Src0TileData &src0, Src1TileData &src1, Src2TileData &src2, WaitEvents &... events);

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData, bool exhausted,
          typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp, Src0TileData &src0, Src1TileData &src1, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, SrcTileData &src, uint32_t blockLen, WaitEvents &... events);
```

## 约束

- **实现检查 (A2A3/A5)**:
    - 元素类型必须是 `half` 或 `float`，且在 `dst`/`tmp`/`src*` Tile 间必须一致。
    - 所有 Tile 必须是 `TileType::Vec`、行主序，且 `Rows == 1`（列表存储在单行中）。
    - 针对目标限制，编译时和运行时均会检查 UB 内存使用量（包括各输入、`tmp` 和 `dst` 的 `Cols`）。
- **单列表变体（`TMRGSORT(dst, src, blockLen)`）**：
    - `blockLen` 必须是 64 的倍数（由实现检查）。
    - `src.GetValidCol()` 必须是 `blockLen * 4` 的整数倍。
    - `repeatTimes = src.GetValidCol() / (blockLen * 4)` 必须在 `[1, 255]` 范围内。
- **多列表变体**：
    - 需要 `tmp`，`executedNumList` 由实现写入；支持的列表数量及确切语义由目标定义。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 1, 256>;
  using DstT = Tile<TileType::Vec, float, 1, 256>;
  SrcT src;
  DstT dst;
  TMRGSORT(dst, src, /*blockLen=*/64);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 1, 256>;
  using DstT = Tile<TileType::Vec, float, 1, 256>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TMRGSORT(dst, src, /*blockLen=*/64);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tmrgsort %src, %blockLen : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tmrgsort %src, %blockLen : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = pto.tmrgsort %src, %blockLen : (!pto.tile<...>, dtype) -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tmrgsort ins(%src, %blockLen : !pto.tile_buf<...>, dtype)  outs(%dst : !pto.tile_buf<...>)
```

