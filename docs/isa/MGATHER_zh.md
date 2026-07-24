# MGATHER

## 指令示意图

![MGATHER tile operation](../figures/isa/MGATHER.svg)

## 简介

`MGATHER` 通过UB索引Tile从GM `GlobalTensor` 中读取数据到UB目标Tile。操作模式通过 `Coalesce` 模板参数显式选择：

- **`Coalesce::Row`**（默认）——从 `table[idx[r], :]` 收集整行到 `dst[r, :]`。索引tile是1-D形式（`[1, R]` 行主序；Ascend 950PR/Ascend 950DT上也支持 `[R, 1]` 列主序）。允许 `R = 1`。
- **`Coalesce::Elem`** ——通过 `idx[R, C]` 从一维化 `table` 收集元素到 `dst[R, C]`。索引tile必须与目标具有相同的有效形状。允许退化的 `(1, 1)` 情况。

越界处理通过 `GatherOOB` 模板参数选择。`MGATHER` 没有原子或冲突策略：每个目标槽有唯一的源索引，因此不会发生冲突。

目标也可以是 **L1 / cube `TileType::Mat` tile（NZ布局）**（索引以GM tensor形式提供）。此GM → L1路径（支持 `Coalesce::Row` 和 `Coalesce::Elem`，适用于Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品和Ascend 950PR/Ascend 950DT）见下方 [GM → L1 Gather（TileType::Mat目标）](#gm--l1-gathertiletypemat-) 章节。

按目标分发摘要：

- **CPU模拟器** ——纯C++参考实现。遍历 `validRow * validCol` 并读取 `table[idx[i, j]]`（Elem语义）；CPU sim没有单独的Row coalesce路径。
- **Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品VEC-CORE** ——标量pipe驱动的单线程 / MTE2遍历。Row模式每行通过 `tablePtr + safeIdx * tableRowStride` 发出一次宽 `copy_gm_to_ubuf_align_b*` DMA；Elem模式每元素执行标量GM→UB拷贝。支持ND-GM与ND-UB以及NZ-GM与NZ-UB tile配对。
- **Ascend 950PR/Ascend 950DT SIMT** ——通过 `cce::async_invoke` 以 `dim3{32, 32}`（1024线程）进行SIMT启动。Row模式使用warp并行通道读取；Elem模式将每个通道映射到一个元素。Row内核将GM table视为打包ND（行步长 = `validCols`）；`MGatherCheck` 在编译时强制 `GlobalTable::staticShape[4] == TileDst::ValidCol`。Ascend 950PR/Ascend 950DT不支持NZ块步长布局。退化的Elem `(1, 1)` 情况绕过SIMT启动，在AIV向量核心上运行 `MGatherScalarImpl`。

## 数学语义

### Row Coalesce（`Coalesce::Row`）

目标 `dst[R, C]`，索引 `idx[1, R]`（Ascend 950PR/Ascend 950DT上也为 `idx[R, 1]`），表 `table[TableRows, C]`：

$$ \mathrm{dst}_{r, j} = \mathrm{table}_{\mathrm{idx}_{r},\; j} \quad\text{for } 0 \le r < R,\; 0 \le j < C $$

### Element Coalesce（`Coalesce::Elem`）

目标 `dst[R, C]`，索引 `idx[R, C]`（与 `dst` 相同有效形状），一维表长度为 `TableSize`：

$$ \mathrm{dst}_{r, c} = \mathrm{table}[\mathrm{idx}_{r, c}] \quad\text{for } 0 \le r < R,\; 0 \le c < C $$

### 越界行为

```cpp
enum class GatherOOB : uint8_t {
    Undefined = 0,  // 不检查边界；调用者保证索引有效
    Clamp     = 1,  // 将索引钳制到 capacity - 1
    Wrap      = 2,  // 索引模 capacity
    Zero      = 3   // OOB 返回零；界内索引正常加载
};
```

## 汇编语法

同步形式：

```text
%dst = mgather %mem, %idx : !pto.memref<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
-> !pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

### AS Level 2（DPS）

```text
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

### CPU参考形式

```cpp
template <typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src, TileInd &indexes, WaitEvents &... events);
```

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品形式

```cpp
template <Coalesce  CMode = Coalesce::Row,
          GatherOOB Oob   = GatherOOB::Undefined,
          typename TileDst, typename GlobalTable, typename TileIdx,
          typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalTable& table, TileIdx& idx,
                             WaitEvents&... events);
```

### Ascend 950PR/Ascend 950DT形式

```cpp
template <Coalesce  CMode = Coalesce::Row,
          GatherOOB Mode  = GatherOOB::Undefined,
          typename TileDst, typename GlobalData, typename TileInd,
          typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalData& table, TileInd& idx,
                             WaitEvents&... events);
```

### 参数（NPU形式）

- `dst` —UB目标tile（`TileType::Vec`）；形状 `[R, C]`。
- `table` —源GM `GlobalTensor`。`GlobalTensor::DType` 必须是 `__gm__ T`，与目标元素类型匹配。
- `idx` —UB索引tile（`TileType::Vec`）。
- `CMode` — `Coalesce` 值（`Row` 或 `Elem`）。作为第一个模板参数，操作模式在调用点始终显式指定。
- `Oob` / `Mode` — `GatherOOB` 值，用于越界处理。

### 枚举

```cpp
enum class Coalesce : uint8_t {
    Row  = 0,  // dst[r, :] = table[idx[r], :]   （长度为 R 的 1-D 索引）
    Elem = 1   // dst[i, j] = table[idx[i, j]]   （idx 形状 == dst 形状）
};

enum class GatherOOB : uint8_t {
    Undefined = 0,
    Clamp     = 1,
    Wrap      = 2,
    Zero      = 3
};
```

## 约束

### Tile约束（CPU）

**支持的数据类型：**
- `dst` / `src` 元素类型必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。
- 在AICore目标上（CPU模拟器以 `__CCE_AICORE__` 编译时），还支持 `float8_e4m3_t` 和 `float8_e5m2_t`。
- `indexes` 元素类型必须是 `int32_t` 或 `uint32_t`。

**Tile与内存类型：**
- `dst` 必须是向量Tile（`TileType::Vec`）。
- `indexes` 必须是向量Tile（`TileType::Vec`）。
- `dst` 和 `indexes` 必须使用行主序布局（`BLayout::RowMajor + SLayout::NoneBox`）。
- `src` 必须是位于GM内存中的 `GlobalTensor`。
- `src` 必须使用 `Layout::ND`。

**形状约束：**
- `dst.Rows == indexes.Rows`。
- `indexes` 的形状必须为 `[1, N]`（按行gather）或 `[N, M]`（按元素gather）。
- `dst` 行宽必须满足32字节对齐，即 `dst.Cols * sizeof(T)` 必须是32的倍数。
- `src` 的静态shape必须满足 `Shape<1, 1, 1, TableRows, RowWidth>`。

### Tile约束（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）

**支持的数据类型：**
- `dst` / `src` 元素类型必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`。Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品不支持 `float8_e4m3_t`、`float8_e5m2_t`、`hifloat8_t`。
- `indexes` 元素类型必须是 `int32_t` 或 `uint32_t`。

**Tile与内存类型：**
- `dst` 必须是向量Tile（`TileDst::Loc == TileType::Vec`）。
- `indexes` 必须是向量Tile（`TileIdx::Loc == TileType::Vec`）。
- 索引tile **始终** 是 `BLayout::RowMajor + SLayout::NoneBox`（ND），无论表布局如何。
- `src` 必须是位于GM内存中的 `GlobalTensor`；`GlobalTable::DType == __gm__ T`。
- 目标tile的bulk + sub布局必须与表布局精确配对：
    - `GlobalTable::layout == Layout::ND` ⇒ `TileDst` 是 `BLayout::RowMajor + SLayout::NoneBox`。
    - `GlobalTable::layout == Layout::NZ` ⇒ `TileDst` 是 `BLayout::ColMajor + SLayout::RowMajor + SFractalSize == TileConfig::fractalABSize`（= 512Byte）。

**形状约束：**
- 填充后的 `TileDst::Cols * sizeof(T)` 必须在两种布局中均为32字节对齐。`ValidRow` / `ValidCol` 不受此规则约束。
- 对于 `Coalesce::Row`：`TileIdx::ValidRow == 1` 且 `TileIdx::ValidCol == TileDst::ValidRow`。
- 对于 `Coalesce::Elem`：`TileIdx::ValidRow == TileDst::ValidRow` 且 `TileIdx::ValidCol == TileDst::ValidCol`。
- 两种模式均要求 `TileDst::ValidRow >= 1` 且 `TileDst::ValidCol >= 1`。
- NZ表额外要求 `GlobalTable::staticShape[3] == FRACTAL_NZ_ROW`（= 16）、`GlobalTable::staticShape[4] == C0_SIZE_BYTE / sizeof(T)`（= 32Byte / 元素宽度）、`TileDst::Cols % (C0_SIZE_BYTE / sizeof(T)) == 0` 且 `TileDst::Rows % FRACTAL_NZ_ROW == 0`。

### Tile约束（Ascend 950PR/Ascend 950DT）

**支持的数据类型：**
- `dst` / `src` 元素类型必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。在 `__CCE_AICORE__` 构建中还包含 `hifloat8_t`、`float8_e4m3_t`、`float8_e5m2_t`。
- `indexes` 元素类型必须是 `int32_t` 或 `uint32_t`。

**Tile与内存类型：**
- `dst` 必须是向量Tile（`TileDst::Loc == TileType::Vec`）。
- `indexes` 必须是向量Tile（`TileIdx::Loc == TileType::Vec`）。
- SIMT内核对UB tile的布局无感知：每次UB读写均通过 `tile_offset_2d<TileX>(r, c)`，因此 `TileDst` 可以为 `BLayout::RowMajor` 或 `BLayout::ColMajor`（搭配 `SLayout::NoneBox`）。
- `src` 必须是位于GM内存中的 `GlobalTensor`；`GlobalTable::DType == __gm__ T`。
- **GM表布局：仅 `Layout::ND`。** Ascend 950PR/Ascend 950DT SIMT内核将GM寻址为扁平行主序缓冲区，行步长硬编码为 `validCols`；`MGatherCheck` 强制 `GlobalTable::staticShape[4] == TileDst::ValidCol`，因此表不能有任何行间填充。

**形状约束：**
- 填充后的 `TileDst::Cols * sizeof(T)`（RowMajor）或 `TileDst::Rows * sizeof(T)`（ColMajor）必须32字节对齐。
- 对于 `Coalesce::Row`：索引tile的有效形状为 `[1, R]`（`BLayout::RowMajor`）**或** `[R, 1]`（`BLayout::ColMajor`）。
- 对于 `Coalesce::Elem`：`TileIdx::ValidRow == TileDst::ValidRow` 且 `TileIdx::ValidCol == TileDst::ValidCol`。`TileIdx` 的 `BLayout` 与 `TileDst` 无关。
- 两种模式均要求 `TileDst::ValidRow >= 1` 且 `TileDst::ValidCol >= 1`。Elem模式中退化的 `(1, 1)` 形状绕过SIMT启动。

### 动态运行时形状（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品和Ascend 950PR/Ascend 950DT）

`MGATHER` 同时接受编译时和运行时动态形状：

- `Tile<…, RowMask, ColMask>` 中 `RowMask == -1` 及/或 `ColMask == -1` 将运行时有效范围存储在tile对象中；实现通过 `dst.GetValidRow()` / `dst.GetValidCol()` 读取。
- `Shape<S0, S1, S2, S3, S4>` / `Stride<…>` 中一个或多个 `-1` 条目使用运行时尺寸构造；实现通过 `table.GetShape(GlobalTensorDim::DIM_*)` 读取。

`MGatherCheck` 中的静态断言以 `if constexpr (DIM > 0)` 为门控，仅对编译时已知维度生效。

示例：

```cpp
constexpr auto kPadCols = 16;
using DstTileT    = Tile<TileType::Vec, float,    1, kPadCols, BLayout::RowMajor, -1, -1>;
using IdxTileT    = Tile<TileType::Vec, int32_t,  1, kPadCols, BLayout::RowMajor, -1, -1>;
using TableShape  = Shape<1, 1, 1, -1, -1>;
using TableStride = Stride<1, 1, 1, -1, -1>;

int64_t validCols = 9, d3 = 3, d4 = 10, srcStride3 = 10;
TableShape  tableShape(d3, d4);
TableStride tableStride(srcStride3, (int64_t)1);
GlobalTensor<float, TableShape, TableStride> tableGM(srcGm, tableShape, tableStride);

DstTileT dstTile(1, validCols);
IdxTileT idxTile(1, validCols);
TASSIGN(dstTile, dstUbOffsetBytes);
TASSIGN(idxTile, idxUbOffsetBytes);

MGATHER<Coalesce::Elem, GatherOOB::Undefined>(dstTile, tableGM, idxTile);
```

## 模式匹配

Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品和Ascend 950PR/Ascend 950DT上模式是**显式的**，不会自动检测。`MGatherCheck` 中的静态断言根据所选的 `Coalesce` 值验证提供的tile形状：

```text
A2/A3:
  Coalesce::Row  : Idx.ValidRow == 1 && Idx.ValidCol == Dst.ValidRow
  Coalesce::Elem : Idx.ValidRow == Dst.ValidRow && Idx.ValidCol == Dst.ValidCol

A5:
  Coalesce::Row  : (Idx.ValidRow == 1 && Idx.ValidCol == Dst.ValidRow) ||
                   (Idx.ValidRow == Dst.ValidRow && Idx.ValidCol == 1)
  Coalesce::Elem : (Idx.ValidRow == Dst.ValidRow) && (Idx.ValidCol == Dst.ValidCol)
```

## 布局支持

| Tile / Tensor | CPU | Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品 | Ascend 950PR/Ascend 950DT |
|---------------|-----|-------|----|
| `TileDst` (UB) —ND | `BLayout::RowMajor + SLayout::NoneBox` only | `BLayout::RowMajor + SLayout::NoneBox` | `BLayout::RowMajor` 或 `ColMajor`，`SLayout::NoneBox` |
| `TileDst` (UB) —NZ | 不支持 | `BLayout::ColMajor + SLayout::RowMajor + SFractalSize == 512` | **不支持** |
| `TileIdx` (UB) —Row | 行主序（Cols必须等于1） | `[1, R]` `BLayout::RowMajor + SLayout::NoneBox` | `[1, R]` `RowMajor` **或** `[R, 1]` `ColMajor` |
| `TileIdx` (UB) —Elem | `BLayout::RowMajor + SLayout::NoneBox` | `[R, C]` `BLayout::RowMajor + SLayout::NoneBox` | 任意 `BLayout`，与 `TileDst` 无关 |
| `GlobalTable` (GM) —ND | `Layout::ND` only | `Layout::ND`（线性连续寻址） | `Layout::ND` only；Row模式硬编码 `tableRowStride = validCols` |
| `GlobalTable` (GM) —NZ | 不支持 | `Layout::NZ`；5-D `Shape<B, BCols, BRows, 16, C0>` | **不支持** |

### NZ布局（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）

当 `GlobalTable::layout == Layout::NZ` 且 `TileDst` 是匹配的 `BLayout::ColMajor + SLayout::RowMajor + SFractalSize = 512` tile时，`MGATHER`（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）运行专用的NZ路径（`MGatherRowNzImpl`、`MGatherElemNzImpl`）。

- **常量。** `kC0 = C0_SIZE_BYTE / sizeof(T)`；`kFRow = FRACTAL_NZ_ROW = 16`。每个分形块为 `kFRow × kC0` 元素（= 512Byte）。
- **逻辑形状。** 逻辑行 = `gShape2 * kFRow`。逻辑列 = `gShape0 * gShape1 * kC0`。Row模式的OOB以逻辑行数进行夹制/取模；Elem模式以总元素数进行。
- **Row模式。** 对每个逻辑行 `r`，内核将 `idx[r]` 映射到 `(srcBlockRow, srcRowInBlock)`，将 `r` 映射到 `(dstBlockRow, dstRowInBlock)`，然后对每个外层批次发出 **一次多burst MTE2传输**。当 `Oob == GatherOOB::Zero` 时，内核在DMA循环前对整个tile预填 `T(0)` 并跳过OOB行的DMA。
- **Elem模式。** 对每个 `(r, c)`，内核将 `idx` 映射到 `(logicalRow, logicalCol) = (idx / nLogicalCols, idx % nLogicalCols)`，然后通过 `MGatherNZGmOffset` 转换为NZ物理偏移。遍历顺序为**块列 → 行 → 块内列**，确保连续写入目标连续32Byte UB块。

## Pipe / 同步模型

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品—显式pipe握手

调用者**不需要插入任何额外屏障**，只需在 `MGATHER` 之前使用标准的 `TLOAD` 后置 `set_flag(PIPE_MTE2, PIPE_V)` / `wait_flag(PIPE_MTE2, PIPE_V)` 对，将索引tile带到向量pipe上的干净状态。

| 阶段 | Pipe转换 | 保护内容 |
|-------|-----------------|----------------|
| 前导 (Row ND + Row NZ) | `V→S`、`MTE3→S` flag链 | 使索引tile对标量读取可见，并冲刷可能在标量循环开始前与UB重叠的任何待处理MTE3写入 |
| 前导 (Elem ND + Elem NZ) | `V→S`、`MTE3→S`、`MTE2→S` flag链 | 与Row相同，额外的 `MTE2→S` flag冲刷标量循环读取 `idxPtr` 前的任何飞行中的MTE2 burst |
| Row后置 (ND + NZ) | `S→MTE2`、`MTE2→V`、`MTE2→MTE3`、`S→V`、`S→MTE3` flag链 | 确保标量→MTE2竞争解决，然后在下游消费者触摸目标tile前排空MTE2 DMA |
| Elem后置 (ND + NZ) | `S→V`、`S→MTE2`、`S→MTE3` flag链 | 使标量UB写入对V、MTE2和MTE3可见 |

### Ascend 950PR/Ascend 950DT—SIMT启动与V↔S握手

Ascend 950PR/Ascend 950DT实现将几乎整个pipe模型隐藏在 `cce::async_invoke` 之后。唯一显式的内核端握手在标量回退路径（`MGatherScalarImpl`，用于Elem `(1, 1)`）：

| 阶段 | Pipe转换 | 保护内容 |
|-------|-----------------|----------------|
| 前导（标量回退） | `set_flag(PIPE_V, PIPE_S)` / `wait_flag(PIPE_V, PIPE_S)` | 在单元素gather前使索引tile对标量pipe可见 |
| 后置（标量回退） | `set_flag(PIPE_S, PIPE_V)` / `wait_flag(PIPE_S, PIPE_V)` | 将对标量UB的写入释放给下游向量操作 |

## UB内存预算

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品

AIV向量核心具有标准的CANN 192KB UB布局。`MGATHER` 不从内核内部分配任何UB scratch—唯一的UB消费者是调用者分配的目标tile和索引tile。

### Ascend 950PR/Ascend 950DT

Ascend 950PR/Ascend 950DT SIMT内核在AIV向量核心上运行。所有用户tile必须适应AIV的256KB Unified Buffer，加上两个固定的运行时预留：8KB保留区域和Data Cache（最少32KB）。因此最大可用为：

```text
max dynUBufSize = 256 KB - 8 KB - 32 KB - static_memory
                = 216 KB - static_memory
```

当使用 `TASSIGN` 手动放置tile时，编译器看到 `static_memory ≈ 0`，全部 **216KB** 可作为 `dynUBufSize` 使用。当工作集在默认预算下超过128KB时，通过 `kernel_name<<<numBlocks, dynUBufSize, stream>>>(args...)` 显式声明。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, int32_t, 16, 16>;
  DstT dst;
  IdxT idx;
  // src 是 GM 中的 GlobalTensor
  MGATHER(dst, src, idx);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, int32_t, 16, 16>;
  DstT dst;
  IdxT idx;
  // src 是 GM 中的 GlobalTensor
  TASSIGN(dst, 0x1000);
  TASSIGN(idx, 0x2000);
  MGATHER(dst, src, idx);
}
```

### Row Coalesce—Embedding Lookup（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品或Ascend 950PR/Ascend 950DT）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T, int R, int C, int TableRows>
AICORE void example_embedding_lookup(__gm__ T* tablePtr, __gm__ int32_t* idxPtr, __gm__ T* outPtr)
{
    using DstTile     = Tile<TileType::Vec, T,       R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, 1, R, BLayout::RowMajor, 1, R>;
    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<T, TableShape, TableStride>;
    using IdxShape    = Shape<1, 1, 1, 1, R>;
    using IdxStride   = Stride<1, 1, 1, R, 1>;
    using IdxTensor   = GlobalTensor<int32_t, IdxShape, IdxStride>;

    TableTensor tableGM(tablePtr);
    IdxTensor   idxGM(idxPtr);
    DstTile dst; TASSIGN(dst, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x1000);

    TLOAD(idx, idxGM);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    MGATHER<Coalesce::Row, GatherOOB::Clamp>(dst, tableGM, idx);
}
```

### Element Coalesce—2-D随机访问

```cpp
AICORE void example_elem_2d(__gm__ float* tablePtr, __gm__ int32_t* idxPtr)
{
    constexpr int R = 8, C = 32, TableSize = 256;

    using DstTile     = Tile<TileType::Vec, float,   R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, R, C, BLayout::RowMajor, R, C>;
    using TableShape  = Shape<1, 1, 1, 1, TableSize>;
    using TableStride = Stride<1, 1, 1, TableSize, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    DstTile dst; TASSIGN(dst, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x0800);

    MGATHER<Coalesce::Elem, GatherOOB::Wrap>(dst, tableGM, idx);
}
```

### Row Coalesce— `[R, 1]` ColMajor索引（Ascend 950PR/Ascend 950DT only）

```cpp
AICORE void example_row_colidx(__gm__ half* tablePtr, __gm__ int32_t* idxPtr)
{
    constexpr int R = 8, C = 64, TableRows = 64;

    using DstTile     = Tile<TileType::Vec, half,    R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, R, 1, BLayout::ColMajor, R, 1>;
    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<half, TableShape, TableStride>;
    using IdxShape    = Shape<1, 1, 1, R, 1>;
    using IdxStride   = Stride<1, 1, 1, 1, 1>;
    using IdxTensor   = GlobalTensor<int32_t, IdxShape, IdxStride, Layout::DN>;

    TableTensor tableGM(tablePtr);
    IdxTensor   idxGM(idxPtr);
    DstTile dst; TASSIGN(dst, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x1000);

    TLOAD(idx, idxGM);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    MGATHER<Coalesce::Row, GatherOOB::Undefined>(dst, tableGM, idx);
}
```

### Element Coalesce— `(1, 1)` 退化情况

```cpp
AICORE void example_scalar(__gm__ float* tablePtr, __gm__ int32_t* idxPtr)
{
    constexpr int TableSize = 32;

    using DstTile     = Tile<TileType::Vec, float,   1, 8, BLayout::RowMajor, 1, 1>;
    using IdxTile     = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 1>;
    using TableShape  = Shape<1, 1, 1, 1, TableSize>;
    using TableStride = Stride<1, 1, 1, TableSize, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    DstTile dst; TASSIGN(dst, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x0080);

    MGATHER<Coalesce::Elem>(dst, tableGM, idx);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
```

### PTO汇编形式

```text
%dst = mgather %mem, %idx : !pto.memref<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## GM → L1 Gather（TileType::Mat目标）

除上述GM → UB gather外，`MGATHER` 还支持直接收集索引选择的数据到 **L1 / cube `TileType::Mat` tile（NZ fractal布局）** 中。当 `TileDst::Loc == TileType::Mat` 时，自动选择GM → L1路径。

### 索引来源—GM

GM → L1变体将索引作为GM `GlobalTensor`（`int32_t` / `uint32_t`）提供。Row模式复用三参数形式；Elem模式需要第四个参数 `scratch`（GM workspace）：

```cpp
// Row — 三参数形式
template <Coalesce CMode = Coalesce::Row, GatherOOB Oob = GatherOOB::Undefined,
          typename MatTileDst, typename GlobalTable, typename GlobalIdx>
PTO_INST RecordEvent MGATHER(MatTileDst& dst, GlobalTable& table, GlobalIdx& idx);

// Elem — 四参数形式（含 GM scratch）
template <Coalesce CMode = Coalesce::Elem, GatherOOB Oob = GatherOOB::Undefined,
          typename MatTileDst, typename GlobalTable, typename GlobalIdx, typename GlobalScratch>
PTO_INST RecordEvent MGATHER(MatTileDst& dst, GlobalTable& table, GlobalIdx& idx,
                             GlobalScratch& scratch);
```

### 约束（`MGatherCheckGm2L1`，Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品和Ascend 950PR/Ascend 950DT）

- **Dtypes。** Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品：`int8/uint8/int16/uint16/int32/uint32/half/bfloat16/float`。Ascend 950PR/Ascend 950DT额外支持 `hifloat8/float8_e4m3/float8_e5m2`。Row模式进一步要求 `sizeof(T) <= 4`。
- **索引。** `idx` 必须是GM `GlobalTensor`，类型为 `int32_t` 或 `uint32_t`。
- **表。** `GlobalTable::DType == __gm__ T`，`GlobalTable::layout == Layout::ND`。
- **Scratch（仅Elem）。** `GlobalScratch::DType == __gm__ T`。
- **目标。** `TileDst::Loc == TileType::Mat`，NZ形式（`!isRowMajor && SFractal == SLayout::RowMajor && SFractalSize == 512`），`TileDst::Cols % (C0_SIZE_BYTE / sizeof(T)) == 0`，`TileDst::Rows % FRACTAL_NZ_ROW (16) == 0`。

### 示例—Row gather到L1 NZ tile

```cpp
template <typename T, int R, int C, int TableRows>
AICORE void example_gm2l1_row(__gm__ T* tablePtr, __gm__ int32_t* idxPtr)
{
    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using IdxShape    = Shape<1, 1, 1, 1, R>;
    using IdxStride   = Stride<1, 1, 1, R, 1>;
    GlobalTensor<T, TableShape, TableStride, Layout::ND> tableGM(tablePtr);
    GlobalTensor<int32_t, IdxShape, IdxStride, Layout::ND> idxGM(idxPtr);

    using DstTile = Tile<TileType::Mat, T, R, C, BLayout::ColMajor, R, C, SLayout::RowMajor, 512>;
    DstTile dst; TASSIGN(dst, 0x0);

    MGATHER<Coalesce::Row, GatherOOB::Clamp>(dst, tableGM, idxGM);   // GM (ND) -> L1 (NZ)
}
```

### 示例—Elem gather到L1 NZ tile（含GM scratch）

```cpp
template <typename T, int R, int C, int TableSize>
AICORE void example_gm2l1_elem(__gm__ T* tablePtr, __gm__ int32_t* idxPtr, __gm__ T* scratchPtr)
{
    using TableShape   = Shape<1, 1, 1, 1, TableSize>;
    using TableStride  = Stride<1, 1, 1, TableSize, 1>;
    using IdxShape     = Shape<1, 1, 1, R, C>;
    using IdxStride    = Stride<1, 1, 1, C, 1>;
    using ScratchShape = Shape<1, 1, 1, 1, R * C>;
    using ScratchStride= Stride<1, 1, 1, R * C, 1>;
    GlobalTensor<T, TableShape, TableStride, Layout::ND> tableGM(tablePtr);
    GlobalTensor<int32_t, IdxShape, IdxStride, Layout::ND> idxGM(idxPtr);
    GlobalTensor<T, ScratchShape, ScratchStride, Layout::ND> scratchGM(scratchPtr);

    using DstTile = Tile<TileType::Mat, T, R, C, BLayout::ColMajor, R, C, SLayout::RowMajor, 512>;
    DstTile dst; TASSIGN(dst, 0x0);

    MGATHER<Coalesce::Elem, GatherOOB::Zero>(dst, tableGM, idxGM, scratchGM);
}
```

### Ascend 950PR/Ascend 950DT only—SIMT executor for Elem GM → L1 (`GatherExec::Simt`)

Ascend 950PR/Ascend 950DT上Elem GM → L1路径有两个executor，通过第三个模板参数 `GatherExec` 选择：

```cpp
enum class GatherExec : uint8_t { Scalar = 0, Simt = 1 };
```

- **`GatherExec::Scalar`**（四参数形式的默认值）——cube核心通过标量加载遍历索引。
- **`GatherExec::Simt`** —— **AIV向量核心**通过SIMT内核并行化gather。

```cpp
template <Coalesce CMode, GatherOOB Oob, GatherExec Exec, typename MatTileDst,
          typename GlobalTable, typename GlobalIdx, typename GlobalScratch>
PTO_INST RecordEvent MGATHER(MatTileDst& dst, GlobalTable& table, GlobalIdx& idx,
                             GlobalScratch& scratch);
```

```cpp
template <typename T, int R, int C, int TableSize>
AICORE void example_gm2l1_elem_simt(__gm__ T* tablePtr, __gm__ int32_t* idxPtr, __gm__ T* scratchPtr)
{
    using TableShape   = Shape<1, 1, 1, 1, TableSize>;
    using TableStride  = Stride<1, 1, 1, TableSize, 1>;
    using IdxShape     = Shape<1, 1, 1, R, C>;
    using IdxStride    = Stride<1, 1, 1, C, 1>;
    using ScratchShape = Shape<1, 1, 1, 1, R * C>;
    using ScratchStride= Stride<1, 1, 1, R * C, 1>;
    GlobalTensor<T, TableShape, TableStride, Layout::ND> tableGM(tablePtr);
    GlobalTensor<int32_t, IdxShape, IdxStride, Layout::ND> idxGM(idxPtr);
    GlobalTensor<T, ScratchShape, ScratchStride, Layout::ND> scratchGM(scratchPtr);

    using DstTile = Tile<TileType::Mat, T, R, C, BLayout::ColMajor, R, C, SLayout::RowMajor, 512>;
    DstTile dst; TASSIGN(dst, 0x0);

    MGATHER<Coalesce::Elem, GatherOOB::Zero, GatherExec::Simt>(dst, tableGM, idxGM, scratchGM);
}
```

## 性能考量

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品

1. **Row vs. Elem。** Row coalesce实现最佳聚合带宽。Elem coalesce每个活跃通道发出一条标量GM读取 + UB写入。
2. **顺序标量循环（Elem）。** Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品以单线程顺序遍历 `validRow * validCol` 个通道。
3. **DMA成本（Row）。** ND：每行一次 `copy_gm_to_ubuf_align_b*` 调用。NZ：每个（逻辑行，批次）对一次调用。
4. **OOB成本。** `Undefined` 免费；`Clamp`/`Wrap` 每通道增加一次算术remap；`Zero` 写入 `T(0)` 内联（Elem）或预零tile并跳过OOB行的DMA（Row）。

### Ascend 950PR/Ascend 950DT

1. **形状自适应启动。** SIMT网格大小根据解析的 `validRows`/`validCols` 确定。
2. **OOB策略成本。** `Undefined`：零开销。`Clamp`/`Wrap`：每通道一次算术remap。`Zero`：每通道一次比较和选择。
3. **模式/OOB无线程分支。** 所有决策均为 `if constexpr`。
4. **Row vs. Elem带宽。** Row coalesce实现最佳聚合带宽；Elem coalesce每通道执行一次标量GM加载。

## 相关指令

- [`TLOAD`](TLOAD.md)：连续块传输GM → Tile。
- [`MSCATTER`](MSCATTER.md)：索引散射Tile → GM（逆操作）。
- [`TGATHER`](TGATHER.md)：基于索引的Tile内部gather（同vec-core上的UB-to-UB）。
