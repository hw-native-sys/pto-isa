# TCI

## 指令示意图

![TCI tile operation](../figures/isa/TCI.svg)

## 简介

生成连续整数序列到目标 Tile 中。

## 数学语义

对于有效元素上的线性化索引 `k`：

- 升序：

  $$ \mathrm{dst}_{k} = S + k $$

- 降序：

  $$ \mathrm{dst}_{k} = S - k $$

线性化顺序取决于 Tile 布局（实现定义）。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData &dst, T start, WaitEvents &... events);

template <typename TileData, typename TileDataTmp, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData &dst, T start, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束

- **实现检查 (A2A3/A5)**:
    - `TileData::DType` 必须与标量模板参数 `T` 的类型完全相同。
    - `dst`/`scalar` 元素类型必须相同，且必须是以下之一：`int32_t`、`uint32_t`、`int16_t`、`uint16_t`。
    - `TileData::Rows == 1`（此为实现强制执行的条件，序列沿列方向生成）。
- **有效区域**:
    - 实现使用 `dst.GetValidCol()` 作为序列长度，不参考 `dst.GetValidRow()`。
- **临时 Tile**:
    - **A2A3**：C++ API 提供带显式 `tmp` Tile 的重载，用于向量化实现路径。不带 `tmp` 的重载使用标量循环。`TileDataTmp::DType` 必须为 4 字节类型（`float`、`int32_t` 或 `uint32_t`）。实现将 `tmp` 转换为 `float *` 使用；应按字节数来规划 tmp Tile 大小，而不是按 `TileDataTmp::DType` 的类型理解。
    - **b32 元素类型**（`int32_t`、`uint32_t`）：最小 tmp 大小 = 768字节（192 个 float 元素）。
      向量化路径在 `tmp` 内使用两个 float 子缓冲区：`tmp0` 位于偏移 0，`tmp1` 位于偏移 +128 floats。`tmp0` 最多持有 64 个 float 元素（256 字节）用于初始分数序列，`tmp1` 最多持有 64 个 float 元素（256 字节）用于累积结果。最高访问的字节偏移为 128 × 4 + 64 × 4 = **768 字节**（192 个 float 元素）。
    - **b16 元素类型**（`int16_t`、`uint16_t`）：最小 tmp 大小 = 1792字节（448 个 float 元素）。
      向量化路径在 `tmp` 内使用四个子缓冲区：`tmp0/tmp1`（float）位于偏移 0 和 +128，`tmp2/tmp3`（half）位于偏移 +256 和 +384（以 float 索引单位计）。`tmp0/tmp1` 各最多持有 64 个 float（256 字节）用于分数序列生成。`tmp2` 最多持有 16 个 half 元素（32 字节）用于 float 到 half 的转换。`tmp3` 最多持有 128 个 half 元素（256 字节）用于最终的 half 精度累积。最高访问的字节偏移为 384 × 4 + 128 × 2 = **1792 字节**（448 个 float 元素）。
    - 一个方便的形状无关分配大小为 2048 字节（2 KiB），例如 `Tile<TileType::Vec, float, 1, 512>`。
    - **A5**：`tmp` Tile 被接受但不使用。A5 硬件直接使用 `vci` 向量指令，无需临时缓冲区。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, int32_t, 1, 16>;
  TileT dst;
  TCI<TileT, int32_t, /*descending=*/0>(dst, /*S=*/0);
}
```

### 自动（带 tmp）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto_tmp() {
  using TileT = Tile<TileType::Vec, int32_t, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 1, 512>;
  TileT dst;
  TmpT tmp;
  TCI<TileT, TmpT, int32_t, /*descending=*/0>(dst, /*S=*/0, tmp);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, int32_t, 1, 16>;
  TileT dst;
  TASSIGN(dst, 0x1000);
  TCI<TileT, int32_t, /*descending=*/1>(dst, /*S=*/100);
}
```

### 手动（带 tmp）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual_tmp() {
  using TileT = Tile<TileType::Vec, int32_t, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 1, 512>;
  TileT dst;
  TmpT tmp;
  TASSIGN(dst, 0x1000);
  TASSIGN(tmp, 0x2000);
  TCI<TileT, TmpT, int32_t, /*descending=*/1>(dst, /*S=*/100, tmp);
}
```
