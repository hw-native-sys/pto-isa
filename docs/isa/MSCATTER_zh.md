# MSCATTER

## 指令示意图

![MSCATTER tile operation](../figures/isa/MSCATTER.svg)

## 简介

`MSCATTER` 通过 UB 索引 Tile 将 UB 源 Tile 的数据写入 GM `GlobalTensor`。操作模式通过 `Coalesce` 模板参数显式选择：

- **`Coalesce::Row`**（默认）—— 将整行 `src[r, :]` 散射到 `table[idx[r], :]`。索引 tile 是 1-D 形式（`[1, R]` 行主序；A5 上也支持 `[R, 1]` 列主序）。允许 `R = 1`。
- **`Coalesce::Elem`** —— 从 `src[R, C]`（或 `src[1, N]`）通过 `idx[R, C]` 逐元素散射到一维化 `table`。索引 tile 必须与源具有相同的有效形状。允许退化的 `(1, 1)` 情况。

写入行为通过正交的模板策略控制：

- `ScatterAtomicOp` — `None`（普通存储）、`Add`、`Max`、`Min`。原子数据类型支持因目标而异（见下表）。
- `ScatterOOB` — `Undefined`、`Skip`、`Clamp`、`Wrap`。没有 `Zero` 选项（操作写入已有表；越界索引没有可写入零值的真实目标槽位）。
- `ScatterConflict`（**仅 A5**） — `Last`（确定性最大索引获胜，仅在 `Atomic == None` 时生效）或 `Default`（warp 调度器依赖）。A2/A3 没有 `ScatterConflict` 参数，因为内核严格顺序执行，碰撞由"最后写入获胜"解决。

按目标分发摘要：

- **CPU 模拟器** —— 纯 C++ 参考实现。以行主序迭代顺序遍历 `validRow * validCol`，写入 `table[idx[i, j]] = src[i, j]`（Elem 语义）。当多个源映射到同一目标时，行主序迭代顺序中最后写入者获胜。
- **A2/A3 VEC-CORE** —— 标量 pipe 驱动的单线程 / MTE3 遍历。Row 模式每行通过 `tablePtr + safeIdx * tableRowStride` 发出一次宽 `copy_ubuf_to_gm_align_b*` DMA；Elem 模式每元素执行标量 UB→GM 存储。支持 ND-GM 与 ND-UB 以及 NZ-GM 与 NZ-UB tile 配对。`ScatterAtomicOp::None` 时始终"最后写入获胜"。
- **A5 SIMT** —— 通过 `cce::async_invoke` 以 `dim3{32, 32}`（1024 线程）进行 SIMT 启动。Row 模式使用 warp 并行通道写入；Elem 模式将每个通道映射到一个元素。Row 内核将 GM table 视为打包 ND（行步长 = `validCols`）；`MScatterCheck` 在编译时强制 `GlobalTable::staticShape[4] == TileSrc::ValidCol`。`Conflict::Last` 实现为以槽位为中心的反向扫描（`last_owner_find_*`），结果确定且无竞争。A5 不支持 NZ 块步长布局。退化的 Elem `(1, 1)` 情况绕过 SIMT 启动。

## 数学语义

### Row Coalesce（`Coalesce::Row`）

源 `src[R, C]`，索引 `idx[1, R]`（A5 上也为 `idx[R, 1]`），表 `table[TableRows, C]`。对每行 `r`：

$$ \mathrm{table}_{\mathrm{idx}_{r},\; j} \;\leftarrow\; \mathrm{atom}\!\left(\mathrm{table}_{\mathrm{idx}_{r},\; j},\; \mathrm{src}_{r, j}\right) \quad\text{for } 0 \le j < C $$

其中 `atom` 对 `ScatterAtomicOp::None` 为恒等替换，否则为对应的原子累加。

### Element Coalesce（`Coalesce::Elem`）

源 `src[R, C]`，索引 `idx[R, C]`（与 `src` 相同有效形状），一维表长度为 `TableSize`：

$$ \mathrm{table}[\mathrm{idx}_{r, c}] \;\leftarrow\; \mathrm{atom}\!\left(\mathrm{table}[\mathrm{idx}_{r, c}],\; \mathrm{src}_{r, c}\right) $$

### 原子累加

当选择 `ScatterAtomicOp::Add` / `Max` / `Min` 时：

$$ \mathrm{table}[\cdot] \mathrel{\oplus}= \mathrm{src}_{\cdot},\quad \oplus \in \{+,\; \max,\; \min\} $$

### 冲突解决

- **A2/A3。** 内核在递增的 `(r)`（Row）或 `(r, c)`（Elem）顺序下严格顺序执行。`ScatterAtomicOp::None` 时，**后续写入始终获胜**（"最后写入获胜"）。
- **A5。** `ScatterAtomicOp::None` 时：
    - **`Conflict::Last`** —— 以给定目标槽位为目标的**最大平坦索引**的源位置的值被存储。实现为以槽位为中心的反向扫描，构造上无竞争。
    - **`Conflict::Default`** —— 幸存写入者依赖 warp 调度器。对于无碰撞的索引集合，结果与 `Last` 相同。
- **CPU 模拟器。** 行主序迭代顺序中最后写入者获胜（与 `Conflict::Last` 匹配）。

原子模式忽略 `ScatterConflict`，因为 GM 原子 R-M-W 自行序列化碰撞写入。

### 越界行为

```cpp
enum class ScatterOOB : uint8_t {
    Undefined = 0,  // 不检查边界；调用者保证索引有效
    Skip      = 1,  // 丢弃写入（保留原始表值）
    Clamp     = 2,  // 将索引钳制到 capacity - 1
    Wrap      = 3   // 索引模 capacity
};
```

- `Undefined`：调用者保证 `idx < capacity`；不应用任何 remap。
- `Skip`：越界行/元素直接不写入。该 GM 地址处的原始表值被保留。
- `Clamp`：`idx = min(idx, capacity - 1)`。
- `Wrap`：`idx = idx % capacity`。

没有 `Zero` 选项 — OOB 索引从未标识真实目标槽位，因此 `Skip` 是自然的"OOB 时无操作"策略。

## 汇编语法

同步形式：

```text
mscatter %src, %idx, %mem : !pto.tile<...>, !pto.tile<...>, !pto.memref<...>
```

### AS Level 1（SSA）

```text
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### AS Level 2（DPS）

```text
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

### CPU 参考形式

```cpp
template <typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &... events);
```

### A2/A3 形式

```cpp
template <Coalesce        CMode  = Coalesce::Row,
          ScatterAtomicOp AtomOp = ScatterAtomicOp::None,
          ScatterOOB      Oob    = ScatterOOB::Undefined,
          typename GlobalTable, typename TileSrc, typename TileIdx,
          typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalTable& table, TileSrc& src, TileIdx& idx,
                              WaitEvents&... events);
```

A2/A3 没有 `ScatterConflict` 模板参数（内核始终顺序执行，碰撞为确定性的"最后写入获胜"）。

### A5 形式

```cpp
template <Coalesce         Mode     = Coalesce::Row,
          ScatterAtomicOp  Atomic   = ScatterAtomicOp::None,
          ScatterOOB       Oob      = ScatterOOB::Undefined,
          ScatterConflict  Conflict = ScatterConflict::Last,
          typename GlobalTable, typename TileSrc, typename TileIdx,
          typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalTable& table, TileSrc& src, TileIdx& idx,
                              WaitEvents&... events);
```

### 参数（NPU 形式）

- `table` — 目标 GM `GlobalTensor`。`GlobalTensor::DType` 必须是 `__gm__ T`，与源元素类型匹配。
- `src` — UB 源 tile（`TileType::Vec`）；形状 `[R, C]`。
- `idx` — UB 索引 tile（`TileType::Vec`）。
- `CMode` / `Mode` — `Coalesce` 值（`Row` 或 `Elem`）。
- `AtomOp` / `Atomic` — `ScatterAtomicOp` 值。A2/A3 不支持 `Max` / `Min`。
- `Oob` — `ScatterOOB` 值。
- `Conflict`（**仅 A5**） — `ScatterConflict` 值。仅在 `Atomic == None` 时生效。

### 枚举

```cpp
enum class Coalesce : uint8_t {
    Row  = 0,  // table[idx[r], :] = src[r, :]
    Elem = 1   // table[idx[i, j]] = src[i, j]
};

enum class ScatterAtomicOp : uint8_t {
    None = 0,  // 普通存储
    Add  = 1,  // 原子加法
    Max  = 2,  // 原子最大值（仅 A5）
    Min  = 3   // 原子最小值（仅 A5）
};

enum class ScatterOOB : uint8_t {
    Undefined = 0,
    Skip      = 1,
    Clamp     = 2,
    Wrap      = 3
};

enum class ScatterConflict : uint8_t {  // 仅 A5
    Last    = 0,  // 确定性：最大源索引获胜
    Default = 1   // warp 调度器依赖
};
```

### 原子类型支持

| Atomic | CPU（ABI 契约 / 模拟器行为） | A2/A3 | A5 |
|--------|----------------------------|-------|----|
| `None` | 所有 dtypes | 所有 dtypes | 所有 dtypes |
| `Add`  | ABI 契约：`int32_t`、`uint32_t`、`float`、`half` | `int8_t`、`int16_t`、`int32_t`、`half`、`bfloat16_t`、`float`（仅带符号整数 — 无 `uint*`） | `int32_t`、`uint32_t`、`float`、`half`、`bfloat16_t` |
| `Max`  | ABI 契约：`int32_t` 或 `float` | 不支持（`MScatterCheck` 编译时拒绝） | `int32_t`、`uint32_t`、`float` |
| `Min`  | ABI 契约：`int32_t` 或 `float` | 不支持（`MScatterCheck` 编译时拒绝） | `int32_t`、`uint32_t`、`float` |

## 约束

### Tile 约束（CPU）

**支持的数据类型：**
- `src` / `dst` 元素类型必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。
- 在 AICore 目标上（CPU 模拟器以 `__CCE_AICORE__` 编译时），还支持 `float8_e4m3_t` 和 `float8_e5m2_t`。
- `indexes` 元素类型必须是 `int32_t` 或 `uint32_t`。

**Tile 与内存类型：**
- `src` 必须是向量 Tile（`TileType::Vec`）。
- `indexes` 必须是向量 Tile（`TileType::Vec`）。
- `src` 和 `indexes` 必须使用行主序布局（`BLayout::RowMajor + SLayout::NoneBox`）。
- `dst` 必须是位于 GM 内存中的 `GlobalTensor`。
- `dst` 必须使用 `Layout::ND`。

**形状约束：**
- `src.Rows == indexes.Rows`。
- `indexes` 形状为 `[N, 1]`（按行）或 `[N, M]`（按元素）。
- `src` 行宽必须 32字节对齐。
- `dst` 静态 shape 满足 `Shape<1, 1, 1, TableRows, RowWidth>`。

### Tile 约束（A2/A3）

**支持的数据类型：**
- `src` / `dst` 元素类型：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。不支持 `float8_e4m3_t`、`float8_e5m2_t`、`hifloat8_t`。
- `indexes` 元素类型必须是 `int32_t` 或 `uint32_t`。

**Tile 与内存类型：**
- `src` 必须是向量 Tile（`TileSrc::Loc == TileType::Vec`）。
- `indexes` 必须是向量 Tile（`TileIdx::Loc == TileType::Vec`）。
- 索引 tile **始终** 是 `BLayout::RowMajor + SLayout::NoneBox`（ND）。
- `dst` 必须是 GM 中的 `GlobalTensor`；`GlobalTable::DType == __gm__ T`。
- 源 tile 的布局必须与表布局精确配对：
    - `GlobalTable::layout == Layout::ND` ⇒ `TileSrc` 为 `BLayout::RowMajor + SLayout::NoneBox`。
    - `GlobalTable::layout == Layout::NZ` ⇒ `TileSrc` 为 `BLayout::ColMajor + SLayout::RowMajor + SFractalSize == 512`。

**原子操作约束：**
- `ScatterAtomicOp::None` 支持所有上述 dtype。
- `ScatterAtomicOp::Add` 要求 `int8_t`、`int16_t`、`int32_t`、`half`、`bfloat16_t` 或 `float`。A2/A3 不支持无符号整数原子加法。
- `ScatterAtomicOp::Max` 和 `Min` **在 A2/A3 上不支持**。

**形状约束：**
- 填充后的 `TileSrc::Cols * sizeof(T)` 必须 32字节对齐。
- `Coalesce::Row`：`TileIdx::ValidRow == 1` 且 `TileIdx::ValidCol == TileSrc::ValidRow`。
- `Coalesce::Elem`：`TileIdx::ValidRow == TileSrc::ValidRow` 且 `TileIdx::ValidCol == TileSrc::ValidCol`。
- NZ 表额外要求：`staticShape[3] == 16`、`staticShape[4] == 32/sizeof(T)`、`Cols % kC0 == 0`、`Rows % 16 == 0`。

### Tile 约束（A5）

**支持的数据类型：**
- `src` / `dst` 元素类型：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。`__CCE_AICORE__` 构建中还包含 `hifloat8_t`、`float8_e4m3_t`、`float8_e5m2_t`。
- `indexes` 元素类型必须是 `int32_t` 或 `uint32_t`。

**Tile 与内存类型：**
- `src` 必须是向量 Tile（`TileSrc::Loc == TileType::Vec`）。
- `indexes` 必须是向量 Tile（`TileIdx::Loc == TileType::Vec`）。
- SIMT 内核对 UB tile 的布局无感知：每次读写均通过 `tile_offset_2d<TileX>(r, c)`。
- **GM 表布局：仅 `Layout::ND`。** A5 SIMT 内核将 GM 寻址为扁平行主序缓冲区，行步长硬编码为 `validCols`；`MScatterCheck` 强制 `GlobalTable::staticShape[4] == TileSrc::ValidCol`。

**原子操作约束：**
- `ScatterAtomicOp::None` 支持所有 dtype。
- `ScatterAtomicOp::Add` 要求 `int32_t`、`uint32_t`、`float`、`half` 或 `bfloat16_t`。
- `ScatterAtomicOp::Max` / `Min` 要求 `int32_t`、`uint32_t` 或 `float`。

**形状约束：**
- 填充后的 `TileSrc::Cols * sizeof(T)`（RowMajor）或 `TileSrc::Rows * sizeof(T)`（ColMajor）必须 32字节对齐。
- `Coalesce::Row`：索引 tile 有效形状为 `[1, R]`（`RowMajor`）**或** `[R, 1]`（`ColMajor`）。
- `Coalesce::Elem`：`TileIdx::ValidRow == TileSrc::ValidRow` 且 `TileIdx::ValidCol == TileSrc::ValidCol`。

### 动态运行时形状（A2/A3 和 A5）

`MSCATTER` 同时接受编译时和运行时动态形状：
- `Tile<…, RowMask, ColMask>` 中 `RowMask == -1` 及/或 `ColMask == -1` 将运行时有效范围存储在 tile 中。
- `Shape<S0, S1, S2, S3, S4>` / `Stride<…>` 中的 `-1` 条目使用运行时尺寸构造。

`MScatterCheck` 中的静态断言以 `if constexpr (DIM > 0)` 为门控。

示例：

```cpp
constexpr auto kPadCols = 16;
using SrcTileT    = Tile<TileType::Vec, float,   1, kPadCols, BLayout::RowMajor, -1, -1>;
using IdxTileT    = Tile<TileType::Vec, int32_t, 1, kPadCols, BLayout::RowMajor, -1, -1>;
using TableShape  = Shape<1, 1, 1, -1, -1>;
using TableStride = Stride<1, 1, 1, -1, -1>;

int64_t validCols = 9, tableR = 3, tableC = 10;
TableShape  tableShape(tableR, tableC);
TableStride tableStride(tableC, (int64_t)1);
GlobalTensor<float, TableShape, TableStride> tableGM(dstGm, tableShape, tableStride);

SrcTileT srcTile(1, validCols);
IdxTileT idxTile(1, validCols);
TASSIGN(srcTile, srcUbOffsetBytes);
TASSIGN(idxTile, idxUbOffsetBytes);

MSCATTER<Coalesce::Elem, ScatterAtomicOp::None, ScatterOOB::Skip>(tableGM, srcTile, idxTile);
```

## 模式匹配

模式在 A2/A3 和 A5 上是**显式的**，不会自动检测。`MScatterCheck` 验证 tile 形状与所选 `Coalesce` 值：

```text
A2/A3:
  Coalesce::Row  : Idx.ValidRow == 1 && Idx.ValidCol == Src.ValidRow
  Coalesce::Elem : Idx.ValidRow == Src.ValidRow && Idx.ValidCol == Src.ValidCol

A5:
  Coalesce::Row  : (Idx.ValidRow == 1 && Idx.ValidCol == Src.ValidRow) ||
                   (Idx.ValidRow == Src.ValidRow && Idx.ValidCol == 1)
  Coalesce::Elem : (Idx.ValidRow == Src.ValidRow) && (Idx.ValidCol == Src.ValidCol)
```

## 布局支持

| Tile / Tensor | CPU | A2/A3 | A5 |
|---------------|-----|-------|----|
| `TileSrc` (UB) — ND | `BLayout::RowMajor + SLayout::NoneBox` only | `BLayout::RowMajor + SLayout::NoneBox` | `BLayout::RowMajor` 或 `ColMajor`，`SLayout::NoneBox` |
| `TileSrc` (UB) — NZ | 不支持 | `BLayout::ColMajor + SLayout::RowMajor + SFractalSize == 512` | **不支持** |
| `TileIdx` (UB) — Row | row-major（Cols 必须等于 1） | `[1, R]` `BLayout::RowMajor + SLayout::NoneBox` | `[1, R]` `RowMajor` **或** `[R, 1]` `ColMajor` |
| `TileIdx` (UB) — Elem | `BLayout::RowMajor + SLayout::NoneBox` | `[R, C]` `BLayout::RowMajor + SLayout::NoneBox` | 任意 `BLayout`，与 `TileSrc` 无关 |
| `GlobalTable` (GM) — ND | `Layout::ND` only | `Layout::ND`（线性连续寻址） | `Layout::ND` only |
| `GlobalTable` (GM) — NZ | 不支持 | `Layout::NZ`；5-D `Shape<B, BCols, BRows, 16, C0>` | **不支持** |

### NZ 布局（A2/A3）

> **NZ 路径仅在 A2/A3 上存在。** A5 SIMT 内核将 GM 寻址为扁平 ND 缓冲区，没有 NZ 块步长转换。

当 `GlobalTable::layout == Layout::NZ` 且 `TileSrc` 是匹配的 NZ tile 时，`MSCATTER`（A2/A3）运行专用的 NZ 路径。

- **常量。** `kC0 = 32 / sizeof(T)`；`kFRow = 16`。每个分形块为 `16 × kC0` 元素（= 512Byte）。
- **Row 模式。** 对每个逻辑源行，内核映射索引和行到 NZ 块地址，每批次发出一次多 burst MTE3 传输。
- **Elem 模式。** 内核映射 `idx` 到 `(logicalRow, logicalCol)` 并通过 `MScatterNZGmOffset` 转为 NZ 物理偏移。遍历顺序为**块列 → 行 → 块内列**。

## Pipe / 同步模型

### A2/A3 — 显式 pipe 握手

调用者**不需要插入任何额外屏障**，只需在 `MSCATTER` 之前使用标准的 `TLOAD` 后置 flag 对。内核从不使用 `pipe_barrier(PIPE_ALL)`。

| 阶段 | Pipe 转换 | 保护内容 |
|-------|-----------------|----------------|
| 前导 (Row) | `V→S`、`MTE2→S`；若 `Atomic == Add` 则 `MScatterAtomicAddSet<T>()`；最后 `S→MTE3` | 使源和索引 tile 对标量读取可见；对于原子加法，在发出 DMA 前将 MTE3 单元切换为原子模式 |
| 前导 (Elem) | `V→S`、`MTE3→S`、`MTE2→S` flag 链 | 对标量循环前的元素级 UB→GM 存储进行保护 |
| Row 后置 | `MTE3→V`、`MTE3→MTE2` flag 链 | 在下游消费者触摸 GM 前排空 MTE3 DMA |
| Elem 后置 | `S→V`、`S→MTE2`、`S→MTE3` flag 链 | 使标量 GM 写入对下游操作可见 |

### A5 — SIMT 启动与 V↔S 握手

A5 实现将几乎整个 pipe 模型隐藏在 `cce::async_invoke` 之后。唯一显式的手握手在标量回退路径（`MScatterScalarImpl`，用于 Elem `(1, 1)`）。

## UB 内存预算

### A2/A3

AIV 向量核心 192KB UB。`MSCATTER` 不从内核内部分配任何 UB scratch — 仅消耗调用者分配的源 tile 和索引 tile。

### A5

A5 SIMT 内核在 AIV 上运行。最大可用：

```text
max dynUBufSize = 256 KB - 8 KB - 32 KB - static_memory
                = 216 KB - static_memory
```

默认安全工作集 ≤ 128KB。超过时需通过 `kernel_name<<<numBlocks, dynUBufSize, stream>>>(args...)` 显式声明。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, int32_t, 16, 16>;
  SrcT src;
  IdxT idx;
  // dst 是 GM 中的 GlobalTensor
  MSCATTER(dst, src, idx);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, int32_t, 16, 16>;
  SrcT src;
  IdxT idx;
  TASSIGN(src, 0x1000);
  TASSIGN(idx, 0x2000);
  MSCATTER(dst, src, idx);
}
```

### Row Coalesce — Embedding Scatter（A2/A3 或 A5）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T, int R, int C, int TableRows>
AICORE void example_embedding_scatter(__gm__ T* tablePtr, __gm__ T* srcPtr, __gm__ int32_t* idxPtr)
{
    using SrcTile     = Tile<TileType::Vec, T,       R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, 1, R, BLayout::RowMajor, 1, R>;
    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<T, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    SrcTile src; TASSIGN(src, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x1000);

    MSCATTER<Coalesce::Row, ScatterAtomicOp::None, ScatterOOB::Clamp>(tableGM, src, idx);
}
```

### Row Coalesce — 原子加法聚合

```cpp
template <typename T, int R, int C, int TableRows>
AICORE void example_row_atomic_add(__gm__ T* tablePtr, __gm__ T* srcPtr, __gm__ int32_t* idxPtr)
{
    using SrcTile     = Tile<TileType::Vec, T,       R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, 1, R, BLayout::RowMajor, 1, R>;
    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<T, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    SrcTile src; TASSIGN(src, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x1000);

    MSCATTER<Coalesce::Row, ScatterAtomicOp::Add, ScatterOOB::Wrap>(tableGM, src, idx);
}
```

### Element Coalesce — 稀疏更新

```cpp
AICORE void example_elem_sparse(__gm__ float* tablePtr, __gm__ float* srcPtr, __gm__ int32_t* idxPtr)
{
    constexpr int R = 8, C = 32, TableSize = 256;

    using SrcTile     = Tile<TileType::Vec, float,   R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, R, C, BLayout::RowMajor, R, C>;
    using TableShape  = Shape<1, 1, 1, 1, TableSize>;
    using TableStride = Stride<1, 1, 1, TableSize, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    SrcTile src; TASSIGN(src, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x0800);

    MSCATTER<Coalesce::Elem, ScatterAtomicOp::None, ScatterOOB::Skip>(tableGM, src, idx);
}
```

### 确定性最后写入获胜（仅 A5）

```cpp
AICORE void example_last_deterministic(__gm__ half* tablePtr)
{
    constexpr int R = 8, C = 64, TableRows = 65536;

    using SrcTile     = Tile<TileType::Vec, half,    R, C, BLayout::RowMajor, R, C>;
    using IdxTile     = Tile<TileType::Vec, int32_t, R, 1, BLayout::ColMajor, R, 1>;
    using TableShape  = Shape<1, 1, 1, TableRows, C>;
    using TableStride = Stride<1, 1, 1, C, 1>;
    using TableTensor = GlobalTensor<half, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    SrcTile src; TASSIGN(src, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x1000);

    MSCATTER<Coalesce::Row, ScatterAtomicOp::None, ScatterOOB::Clamp, ScatterConflict::Last>(
        tableGM, src, idx);
}
```

### Element Coalesce — `(1, 1)` 退化情况

```cpp
AICORE void example_scalar(__gm__ float* tablePtr, __gm__ float* srcPtr, __gm__ int32_t* idxPtr)
{
    constexpr int TableSize = 32;

    using SrcTile     = Tile<TileType::Vec, float,   1, 8, BLayout::RowMajor, 1, 1>;
    using IdxTile     = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 1>;
    using TableShape  = Shape<1, 1, 1, 1, TableSize>;
    using TableStride = Stride<1, 1, 1, TableSize, 1>;
    using TableTensor = GlobalTensor<float, TableShape, TableStride>;

    TableTensor tableGM(tablePtr);
    SrcTile src; TASSIGN(src, 0x0000);
    IdxTile idx; TASSIGN(idx, 0x0080);

    MSCATTER<Coalesce::Elem>(tableGM, src, idx);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### PTO 汇编形式

```text
mscatter %src, %idx, %mem : !pto.tile<...>, !pto.tile<...>, !pto.memref<...>
# AS Level 2 (DPS)
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

## 性能考量

### A2/A3

1. **Row vs. Elem。** Row coalesce 实现最佳聚合带宽。Elem coalesce 每通道发出一次标量 UB 读取 + GM 写入。
2. **原子加法成本（Row）。** 每次调用一次 `set_atomic_add()` / `set_atomic_none()` 对；MTE3 原子加法单元处理每次 burst 的累加。
3. **OOB 成本。** `Undefined` 免费；`Skip` 每行/元素增加一个分支；`Clamp`/`Wrap` 增加一次算术 remap。

### A5

1. **形状自适应启动。** SIMT 网格大小根据解析的 valid 范围确定。
2. **冲突策略成本。** `Last`：per-lane 寄存器内反向扫描，提前终止。`Default`：零额外开销。原子模式：由 GM 原子 R-M-W 自行序列化。
3. **Row vs. Elem 带宽。** Row coalesce 实现最佳 GM 写入带宽；Elem coalesce 每通道一次标量 GM 存储。

## 相关指令

- [`TSTORE`](TSTORE.md)：连续块传输 Tile → GM。
- [`MGATHER`](MGATHER.md)：索引收集 GM → Tile（逆操作）。
- [`TSCATTER`](TSCATTER.md)：基于索引的 Tile 内部散射。
