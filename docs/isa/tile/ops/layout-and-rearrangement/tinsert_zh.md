# pto.tinsert

`pto.tinsert` 属于[布局与重排](../../layout-and-rearrangement_zh.md)指令集。

## 概述

`TINSERT` 在 `(indexRow, indexCol)` 偏移处将子 Tile 插入到目标 Tile 中。

## 机制

设 `R = src.GetValidRow()` 和 `C = src.GetValidCol()`。概念上，对于 `0 <= i < R` 和 `0 <= j < C`：

$$\mathrm{dst}_{\mathrm{indexRow}+i,\;\mathrm{indexCol}+j} = \mathrm{src}_{i,j}$$

## 语法

### PTO-AS

```text
%dst = tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tinsert ins(%src[%r0, %r1] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

#ifdef PTO_NPU_ARCH_A5
template <TInsertMode mode, typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint32_t indexRow = 0, uint32_t indexCol = 0, WaitEvents &... events);
#endif
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src` | 输入 Tile | 源 Tile |
| `indexRow` | 起始行索引 | 插入到目标 Tile 的起始行 |
| `indexCol` | 起始列索引 | 插入到目标 Tile 的起始列 |
| `dst` | 输出 Tile | 目标 Tile，会被部分覆盖 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 源 Tile 插入后的目标 Tile |

## 副作用

源 Tile 的有效区域会被写入到 `dst` 的指定位置，`dst` 其他位置的内容保持不变。

## 约束

- 运行时边界必须满足 `indexRow + src.Rows <= dst.Rows` 且 `indexCol + src.Cols <= dst.Cols`
- A2/A3 的重载对应 `Acc -> Mat` 插入路径，包括普通形式、`reluMode` 形式、标量预量化形式以及向量预量化（`TINSERT_FP`）形式
- A5 除了上面的 `Acc -> Mat` 插入路径外，还额外提供 `template <TInsertMode mode, ...> TINSERT(...)`，用于 `Vec -> Mat` 与 `Vec -> Vec` 插入变体
- `mode == TInsertMode::ND` 要求源向量 tile 为行优先，并以 ND 布局插入到矩阵 tile
- `mode == TInsertMode::ND_VEC` 要求源和目的都为行优先向量 tile
- NZ 系列模式（`NZ`、`NZ_PLUS_1`、`SPLIT2_NZ_PLUS_1`、`SPLIT4_NZ_PLUS_1`）要求源向量 tile 为 NZ 格式，目的为矩阵 tile

## 异常与非法情形

- 如果插入区域超出目标 Tile 范围，行为未定义
- 如果源/目标类型组合不被支持，backend 可能报错

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using SrcT = TileAcc<float, 16, 16>;
  using DstT = Tile<TileType::Mat, float, 32, 32>;
  SrcT src;
  DstT dst;
  TINSERT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

### PTO-AS

```text
%dst = tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

## 相关页面

- [TINSERT_FP](./tinsert-fp_zh.md)
- 指令集总览：[布局与重排](../../layout-and-rearrangement_zh.md)
