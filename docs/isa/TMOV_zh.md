# TMOV

## 指令示意图

![TMOV tile operation](../figures/isa/TMOV.svg)

## 简介

在 Tile 之间移动/复制，可选应用实现定义的转换模式。

Move/copy between tiles, optionally applying implementation-defined conversion modes selected by template parameters and overloads.

`TMOV` is used for:

- Vec -> Vec moves
- Mat -> Left/Right/Bias/Scaling/Scale(Microscaling) moves (target-dependent)
- Acc -> Vec moves (target-dependent)

## 数学语义

Conceptually copies or transforms elements from `src` into `dst` over the valid region. Exact transformation depends on the selected mode and target.

For the pure copy case:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} $$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

The PTO IR design recommends splitting `TMOV` into a family of ops:

```text
%left  = tmov.m2l %mat  : !pto.tile<...> -> !pto.tile<...>
%right = tmov.m2r %mat  : !pto.tile<...> -> !pto.tile<...>
%bias  = tmov.m2b %mat  : !pto.tile<...> -> !pto.tile<...>
%scale = tmov.m2s %mat  : !pto.tile<...> -> !pto.tile<...>
%vec   = tmov.a2v %acc  : !pto.tile<...> -> !pto.tile<...>
%v1    = tmov.v2v %v0   : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tmov ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events);

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, WaitEvents&... events);

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, WaitEvents&... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV_FP(DstTileData& dst, SrcTileData& src, FpTileData& fp, WaitEvents&... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, FpTileData& fp, WaitEvents&... events);
```

## 约束

- **实现检查 (A2A3)**:
  - Shapes must match: `SrcTileData::Rows == DstTileData::Rows` and `SrcTileData::Cols == DstTileData::Cols`.
  - Supported location pairs (compile-time checked):
    - `Mat -> Left/Right/Bias/Scaling`
    - `Vec -> Vec`
    - `Acc -> Mat` (including optional pre-quant / relu / fp variants via overloads)
  - For `Acc -> Mat`, additional fractal/type constraints are enforced (e.g., `Acc` uses NZ-like fractal, `Mat` uses 512B fractal, and only specific dtype conversions are allowed).
- **实现检查 (A5)**:
  - For `Mat -> *`, shapes must match; for some `Vec` moves, the effective copy size is the min of src/dst valid rows/cols.
  - Supported location pairs include (target-dependent):
    - `Mat -> Left/Right/Bias/Scaling/Scale`
    - `Vec -> Vec` and `Vec -> Mat`
    - `Acc -> Vec` and `Acc -> Mat` (including optional pre-quant / relu / fp variants via overloads)
  - For `Mat -> Left/Right`, additional fractal and dtype constraints are enforced via `CommonCheck` (source fractal must be compatible and element types must match).
  - For `Acc -> Vec/Mat`, additional fractal/type/alignment constraints are enforced via `CheckTMovAccValid`.
  - For `Mat -> Scale`, additional fractal and dtype constraints are enforced via `CommonCheckMX` (source fractal must be compatible and element types must match).

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TMOV(dst, src);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT mat;
  DstT left;
  TASSIGN(mat, 0x1000);
  TASSIGN(left, 0x2000);
  TMOV(left, mat);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tmov ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

