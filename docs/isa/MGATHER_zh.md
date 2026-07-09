# MGATHER

## 指令示意图

![MGATHER tile operation](../figures/isa/MGATHER.svg)

## 简介

`MGATHER` 通过 UB 索引 Tile 从 GM `GlobalTensor` 中读取数据到 UB 目标 Tile。操作模式通过 `Coalesce` 模板参数显式选择：

- **`Coalesce::Row`**（默认）—— 从 `table[idx[r], :]` 收集整行到 `dst[r, :]`。索引 tile 是 1-D 形式（`[1, R]` 行主序；A5 上也支持 `[R, 1]` 列主序）。允许 `R = 1`。
- **`Coalesce::Elem`** —— 通过 `idx[R, C]` 从一维化 `table` 收集元素到 `dst[R, C]`。索引 tile 必须与目标具有相同的有效形状。允许退化的 `(1, 1)` 情况。

越界处理通过 `GatherOOB` 模板参数选择。`MGATHER` 没有原子或冲突策略：每个目标槽有唯一的源索引，因此不会发生冲突。

目标也可以是 **L1 / cube `TileType::Mat` tile（NZ 布局）**（索引以 GM tensor 形式提供）。此 GM → L1 路径（支持 `Coalesce::Row` 和 `Coalesce::Elem`，适用于 A2/A3 和 A5）见下方 [GM → L1 Gather（TileType::Mat 目标）](#gm--l1-gathertiletypemat-) 章节。

按目标分发摘要：

- **CPU 模拟器** —— 纯 C++ 参考实现。遍历 `validRow * validCol` 并读取 `table[idx[i, j]]`（Elem 语义）；CPU sim 没有单独的 Row coalesce 路径。
- **A2/A3 VEC-CORE** —— 标量 pipe 驱动的单线程 / MTE2 遍历。Row 模式每行通过 `tablePtr + safeIdx * tableRowStride` 发出一次宽 `copy_gm_to_ubuf_align_b*` DMA；Elem 模式每元素执行标量 GM→UB 拷贝。支持 ND-GM 与 ND-UB 以及 NZ-GM 与 NZ-UB tile 配对。
- **A5 SIMT** —— 通过 `cce::async_invoke` 以 `dim3{32, 32}`（1024 线程）进行 SIMT 启动。Row 模式使用 warp 并行通道读取；Elem 模式将每个通道映射到一个元素。Row 内核将 GM table 视为打包 ND（行步长 = `validCols`）；`MGatherCheck` 在编译时强制 `GlobalTable::staticShape[4] == TileDst::ValidCol`。A5 不支持 NZ 块步长布局。退化的 Elem `(1, 1)` 情况绕过 SIMT 启动，在 AIV 向量核心上运行 `MGatherScalarImpl`。

## 数学语义

### Row Coalesce（`Coalesce::Row`）

目标 `dst[R, C]`，索引 `idx[1, R]`（A5 上也为 `idx[R, 1]`），表 `table[TableRows, C]`：

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

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

### CPU 参考形式

```cpp
template <typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src, TileInd &indexes, WaitEvents &... events);
```

### A2/A3 形式

```cpp
template <Coalesce  CMode = Coalesce::Row,
          GatherOOB Oob   = GatherOOB::Undefined,
          typename TileDst, typename GlobalTable, typename TileIdx,
          typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalTable& table, TileIdx& idx,
                             WaitEvents&... events);
```

### A5 形式

```cpp
template <Coalesce  CMode = Coalesce::Row,
          GatherOOB Mode  = GatherOOB::Undefined,
          typename TileDst, typename GlobalData, typename TileInd,
          typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalData& table, TileInd& idx,
                             WaitEvents&... events);
```

### 参数（NPU 形式）

- `dst` — UB 目标 tile（`TileType::Vec`）；形状 `[R, C]`。
- `table` — 源 GM `GlobalTensor`。`GlobalTensor::DType` 必须是 `__gm__ T`，与目标元素类型匹配。
- `idx` — UB 索引 tile（`TileType::Vec`）。
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

### Tile 约束（CPU）

**支持的数据类型：**
- `dst` / `src` 元素类型必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。
- 在 AICore 目标上（CPU 模拟器以 `__CCE_AICORE__` 编译时），还支持 `float8_e4m3_t` 和 `float8_e5m2_t`。
- `indexes` 元素类型必须是 `int32_t` 或 `uint32_t`。

**Tile 与内存类型：**
- `dst` 必须是向量 Tile（`TileType::Vec`）。
- `indexes` 必须是向量 Tile（`TileType::Vec`）。
- `dst` 和 `indexes` 必须使用行主序布局（`BLayout::RowMajor + SLayout::NoneBox`）。
- `src` 必须是位于 GM 内存中的 `GlobalTensor`。
- `src` 必须使用 `Layout::ND`。

**形状约束：**
- `dst.Rows == indexes.Rows`。
- `indexes` 的形状必须为 `[N, 1]`（按行 gather）或 `[N, M]`（按元素 gather）。
- `dst` 行宽必须满足 32 字节对齐，即 `dst.Cols * sizeof(T)` 必须是 32 的倍数。
- `src` 的静态 shape 必须满足 `Shape<1, 1, 1, TableRows, RowWidth>`。

### Tile 约束（A2/A3）

**支持的数据类型：**
- `dst` / `src` 元素类型必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`。A2/A3 不支持 `float8_e4m3_t`、`float8_e5m2_t`、`hifloat8_t`。
- `indexes` 元素类型必须是 `int32_t` 或 `uint32_t`。

**Tile 与内存类型：**
- `dst` 必须是向量 Tile（`TileDst::Loc == TileType::Vec`）。
- `indexes` 必须是向量 Tile（`TileIdx::Loc == TileType::Vec`）。
- 索引 tile **始终** 是 `BLayout::RowMajor + SLayout::NoneBox`（ND），无论表布局如何。
- `src` 必须是位于 GM 内存中的 `GlobalTensor`；`GlobalTable::DType == __gm__ T`。
- 目标 tile 的 bulk + sub 布局必须与表布局精确配对：
    - `GlobalTable::layout == Layout::ND` ⇒ `TileDst` 是 `BLayout::RowMajor + SLayout::NoneBox`。
    - `GlobalTable::layout == Layout::NZ` ⇒ `TileDst` 是 `BLayout::ColMajor + SLayout::RowMajor + SFractalSize == TileConfig::fractalABSize`（= 512 B）。

**形状约束：**
- 填充后的 `TileDst::Cols * sizeof(T)` 必须在两种布局中均为 32 字节对齐。`ValidRow` / `ValidCol` 不受此规则约束。
- 对于 `Coalesce::Row`：`TileIdx::ValidRow == 1` 且 `TileIdx::ValidCol == TileDst::ValidRow`。
- 对于 `Coalesce::Elem`：`TileIdx::ValidRow == TileDst::ValidRow` 且 `TileIdx::ValidCol == TileDst::ValidCol`。
- 两种模式均要求 `TileDst::ValidRow >= 1` 且 `TileDst::ValidCol >= 1`。
- NZ 表额外要求 `GlobalTable::staticShape[3] == FRACTAL_NZ_ROW`（= 16）、`GlobalTable::staticShape[4] == C0_SIZE_BYTE / sizeof(T)`（= 32 B / 元素宽度）、`TileDst::Cols % (C0_SIZE_BYTE / sizeof(T)) == 0` 且 `TileDst::Rows % FRACTAL_NZ_ROW == 0`。

### Tile 约束（A5）

**支持的数据类型：**
- `dst` / `src` 元素类型必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。在 `__CCE_AICORE__` 构建中还包含 `hifloat8_t`、`float8_e4m3_t`、`float8_e5m2_t`。
- `indexes` 元素类型必须是 `int32_t` 或 `uint32_t`。

**Tile 与内存类型：**
- `dst` 必须是向量 Tile（`TileDst::Loc == TileType::Vec`）。
- `indexes` 必须是向量 Tile（`TileIdx::Loc == TileType::Vec`）。
- SIMT 内核对 UB tile 的布局无感知：每次 UB 读写均通过 `tile_offset_2d<TileX>(r, c)`，因此 `TileDst` 可以为 `BLayout::RowMajor` 或 `BLayout::ColMajor`（搭配 `SLayout::NoneBox`）。
- `src` 必须是位于 GM 内存中的 `GlobalTensor`；`GlobalTable::DType == __gm__ T`。
- **GM 表布局：仅 `Layout::ND`。** A5 SIMT 内核将 GM 寻址为扁平行主序缓冲区，行步长硬编码为 `validCols`；`MGatherCheck` 强制 `GlobalTable::staticShape[4] == TileDst::ValidCol`，因此表不能有任何行间填充。

**形状约束：**
- 填充后的 `TileDst::Cols * sizeof(T)`（RowMajor）或 `TileDst::Rows * sizeof(T)`（ColMajor）必须 32 字节对齐。
- 对于 `Coalesce::Row`：索引 tile 的有效形状为 `[1, R]`（`BLayout::RowMajor`）**或** `[R, 1]`（`BLayout::ColMajor`）。
- 对于 `Coalesce::Elem`：`TileIdx::ValidRow == TileDst::ValidRow` 且 `TileIdx::ValidCol == TileDst::ValidCol`。`TileIdx` 的 `BLayout` 与 `TileDst` 无关。
- 两种模式均要求 `TileDst::ValidRow >= 1` 且 `TileDst::ValidCol >= 1`。Elem 模式中退化的 `(1, 1)` 形状绕过 SIMT 启动。

### 动态运行时形状（A2/A3 和 A5）

`MGATHER` 同时接受编译时和运行时动态形状：

- `Tile<…, RowMask, ColMask>` 中 `RowMask == -1` 及/或 `ColMask == -1` 将运行时有效范围存储在 tile 对象中；实现通过 `dst.GetValidRow()` / `dst.GetValidCol()` 读取。
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

A2/A3 和 A5 上模式是**显式的**，不会自动检测。`MGatherCheck` 中的静态断言根据所选的 `Coalesce` 值验证提供的 tile 形状：

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

| Tile / Tensor | CPU | A2/A3 | A5 |
|---------------|-----|-------|----|
| `TileDst` (UB) — ND | `BLayout::RowMajor + SLayout::NoneBox` only | `BLayout::RowMajor + SLayout::NoneBox` | `BLayout::RowMajor` 或 `ColMajor`，`SLayout::NoneBox` |
| `TileDst` (UB) — NZ | 不支持 | `BLayout::ColMajor + SLayout::RowMajor + SFractalSize == 512` | **不支持** |
| `TileIdx` (UB) — Row | 行主序（Cols 必须等于 1） | `[1, R]` `BLayout::RowMajor + SLayout::NoneBox` | `[1, R]` `RowMajor` **或** `[R, 1]` `ColMajor` |
| `TileIdx` (UB) — Elem | `BLayout::RowMajor + SLayout::NoneBox` | `[R, C]` `BLayout::RowMajor + SLayout::NoneBox` | 任意 `BLayout`，与 `TileDst` 无关 |
| `GlobalTable` (GM) — ND | `Layout::ND` only | `Layout::ND`（线性连续寻址） | `Layout::ND` only；Row 模式硬编码 `tableRowStride = validCols` |
| `GlobalTable` (GM) — NZ | 不支持 | `Layout::NZ`；5-D `Shape<B, BCols, BRows, 16, C0>` | **不支持** |

### NZ 布局（A2/A3）

当 `GlobalTable::layout == Layout::NZ` 且 `TileDst` 是匹配的 `BLayout::ColMajor + SLayout::RowMajor + SFractalSize = 512` tile 时，`MGATHER`（A2/A3）运行专用的 NZ 路径（`MGatherRowNzImpl`、`MGatherElemNzImpl`）。

- **常量。** `kC0 = C0_SIZE_BYTE / sizeof(T)`；`kFRow = FRACTAL_NZ_ROW = 16`。每个分形块为 `kFRow × kC0` 元素（= 512 B）。
- **逻辑形状。** 逻辑行 = `gShape2 * kFRow`。逻辑列 = `gShape0 * gShape1 * kC0`。Row 模式的 OOB 以逻辑行数进行夹制/取模；Elem 模式以总元素数进行。
- **Row 模式。** 对每个逻辑行 `r`，内核将 `idx[r]` 映射到 `(srcBlockRow, srcRowInBlock)`，将 `r` 映射到 `(dstBlockRow, dstRowInBlock)`，然后对每个外层批次发出 **一次多 burst MTE2 传输**。当 `Oob == GatherOOB::Zero` 时，内核在 DMA 循环前对整个 tile 预填 `T(0)` 并跳过 OOB 行的 DMA。
- **Elem 模式。** 对每个 `(r, c)`，内核将 `idx` 映射到 `(logicalRow, logicalCol) = (idx / nLogicalCols, idx % nLogicalCols)`，然后通过 `MGatherNZGmOffset` 转换为 NZ 物理偏移。遍历顺序为**块列 → 行 → 块内列**，确保连续写入目标连续 32 B UB 块。

## Pipe / 同步模型

### A2/A3 — 显式 pipe 握手

调用者**不需要插入任何额外屏障**，只需在 `MGATHER` 之前使用标准的 `TLOAD` 后置 `set_flag(PIPE_MTE2, PIPE_V)` / `wait_flag(PIPE_MTE2, PIPE_V)` 对，将索引 tile 带到向量 pipe 上的干净状态。

| 阶段 | Pipe 转换 | 保护内容 |
|-------|-----------------|----------------|
| 前导 (Row ND + Row NZ) | `V→S`、`MTE3→S` flag 链 | 使索引 tile 对标量读取可见，并冲刷可能在标量循环开始前与 UB 重叠的任何待处理 MTE3 写入 |
| 前导 (Elem ND + Elem NZ) | `V→S`、`MTE3→S`、`MTE2→S` flag 链 | 与 Row 相同，额外的 `MTE2→S` flag 冲刷标量循环读取 `idxPtr` 前的任何飞行中的 MTE2 burst |
| Row 后置 (ND + NZ) | `S→MTE2`、`MTE2→V`、`MTE2→MTE3`、`S→V`、`S→MTE3` flag 链 | 确保标量→MTE2 竞争解决，然后在下游消费者触摸目标 tile 前排空 MTE2 DMA |
| Elem 后置 (ND + NZ) | `S→V`、`S→MTE2`、`S→MTE3` flag 链 | 使标量 UB 写入对 V、MTE2 和 MTE3 可见 |

### A5 — SIMT 启动与 V↔S 握手

A5 实现将几乎整个 pipe 模型隐藏在 `cce::async_invoke` 之后。唯一显式的内核端握手在标量回退路径（`MGatherScalarImpl`，用于 Elem `(1, 1)`）：

| 阶段 | Pipe 转换 | 保护内容 |
|-------|-----------------|----------------|
| 前导（标量回退） | `set_flag(PIPE_V, PIPE_S)` / `wait_flag(PIPE_V, PIPE_S)` | 在单元素 gather 前使索引 tile 对标量 pipe 可见 |
| 后置（标量回退） | `set_flag(PIPE_S, PIPE_V)` / `wait_flag(PIPE_S, PIPE_V)` | 将对标量 UB 的写入释放给下游向量操作 |

## UB 内存预算

### A2/A3

AIV 向量核心具有标准的 CANN 192 KB UB 布局。`MGATHER` 不从内核内部分配任何 UB scratch — 唯一的 UB 消费者是调用者分配的目标 tile 和索引 tile。

### A5

A5 SIMT 内核在 AIV 向量核心上运行。所有用户 tile 必须适应 AIV 的 256 KB Unified Buffer，加上两个固定的运行时预留：8 KB 保留区域和 Data Cache（最少 32 KB）。因此最大可用为：

```text
max dynUBufSize = 256 KB - 8 KB - 32 KB - static_memory
                = 216 KB - static_memory
```

当使用 `TASSIGN` 手动放置 tile 时，编译器看到 `static_memory ≈ 0`，全部 **216 KB** 可作为 `dynUBufSize` 使用。当工作集在默认预算下超过 128 KB 时，通过 `kernel_name<<<numBlocks, dynUBufSize, stream>>>(args...)` 显式声明。

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
  TASSIGN(dst, 0x1000);
  TASSIGN(idx, 0x2000);
  MGATHER(dst, src, idx);
}
```

### Row Coalesce — Embedding Lookup（A2/A3 或 A5）

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

### Element Coalesce — 2-D 随机访问

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

### Row Coalesce — `[R, 1]` ColMajor 索引（A5 only）

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

### Element Coalesce — `(1, 1)` 退化情况

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

## GM → L1 Gather（TileType::Mat 目标）

除上述 GM → UB gather 外，`MGATHER` 还支持直接收集索引选择的数据到 **L1 / cube `TileType::Mat` tile（NZ fractal 布局）** 中。当 `TileDst::Loc == TileType::Mat` 时，自动选择 GM → L1 路径。

### 索引来源 — GM

GM → L1 变体将索引作为 GM `GlobalTensor`（`int32_t` / `uint32_t`）提供。Row 模式复用三参数形式；Elem 模式需要第四个参数 `scratch`（GM workspace）：

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

### 约束（`MGatherCheckGm2L1`，A2/A3 和 A5）

- **Dtypes。** A2/A3：`int8/uint8/int16/uint16/int32/uint32/half/bfloat16/float`。A5 额外支持 `hifloat8/float8_e4m3/float8_e5m2`。Row 模式进一步要求 `sizeof(T) <= 4`。
- **索引。** `idx` 必须是 GM `GlobalTensor`，类型为 `int32_t` 或 `uint32_t`。
- **表。** `GlobalTable::DType == __gm__ T`，`GlobalTable::layout == Layout::ND`。
- **Scratch（仅 Elem）。** `GlobalScratch::DType == __gm__ T`。
- **目标。** `TileDst::Loc == TileType::Mat`，NZ 形式（`!isRowMajor && SFractal == SLayout::RowMajor && SFractalSize == 512`），`TileDst::Cols % (C0_SIZE_BYTE / sizeof(T)) == 0`，`TileDst::Rows % FRACTAL_NZ_ROW (16) == 0`。

### 示例 — Row gather 到 L1 NZ tile

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

### 示例 — Elem gather 到 L1 NZ tile（含 GM scratch）

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

### A5 only — SIMT executor for Elem GM → L1 (`GatherExec::Simt`)

A5 上 Elem GM → L1 路径有两个 executor，通过第三个模板参数 `GatherExec` 选择：

```cpp
enum class GatherExec : uint8_t { Scalar = 0, Simt = 1 };
```

- **`GatherExec::Scalar`**（四参数形式的默认值）—— cube 核心通过标量加载遍历索引。
- **`GatherExec::Simt`** —— **AIV 向量核心**通过 SIMT 内核并行化 gather。

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

### A2/A3

1. **Row vs. Elem。** Row coalesce 实现最佳聚合带宽。Elem coalesce 每个活跃通道发出一条标量 GM 读取 + UB 写入。
2. **顺序标量循环（Elem）。** A2/A3 以单线程顺序遍历 `validRow * validCol` 个通道。
3. **DMA 成本（Row）。** ND：每行一次 `copy_gm_to_ubuf_align_b*` 调用。NZ：每个（逻辑行，批次）对一次调用。
4. **OOB 成本。** `Undefined` 免费；`Clamp`/`Wrap` 每通道增加一次算术 remap；`Zero` 写入 `T(0)` 内联（Elem）或预零 tile 并跳过 OOB 行的 DMA（Row）。

### A5

1. **形状自适应启动。** SIMT 网格大小根据解析的 `validRows`/`validCols` 确定。
2. **OOB 策略成本。** `Undefined`：零开销。`Clamp`/`Wrap`：每通道一次算术 remap。`Zero`：每通道一次比较和选择。
3. **模式/OOB 无线程分支。** 所有决策均为 `if constexpr`。
4. **Row vs. Elem 带宽。** Row coalesce 实现最佳聚合带宽；Elem coalesce 每通道执行一次标量 GM 加载。

## 相关指令

- [`TLOAD`](TLOAD.md)：连续块传输 GM → Tile。
- [`MSCATTER`](MSCATTER.md)：索引散射 Tile → GM（逆操作）。
- [`TGATHER`](TGATHER.md)：基于索引的 Tile 内部 gather（同 vec-core 上的 UB-to-UB）。
