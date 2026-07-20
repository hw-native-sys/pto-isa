# TCI

## 指令示意图

![TCI tile operation](../figures/isa/TCI.svg)

## 简介

生成连续整数序列到目标Tile中。

## 数学语义

对于有效元素上的线性化索引 `k`：

- 升序：

  $$ \mathrm{dst}_{k} = S + k $$

- 降序：

  $$ \mathrm{dst}_{k} = S - k $$

线性化顺序取决于Tile布局（实现定义）。

## 汇编语法

同步形式：

```text
%dst = tci %S {descending = false} : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileData, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData &dst, T start, WaitEvents &... events);

template <typename TileData, typename TileDataTmp, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData &dst, T start, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束

- **实现检查 (Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品/Ascend 950PR/Ascend 950DT)**:
    - `TileData::DType` 必须与标量模板参数 `T` 的类型完全相同。
    - `dst`/`scalar` 元素类型必须相同，且必须是以下之一：`int32_t`、`uint32_t`、`int16_t`、`uint16_t`。
    - `TileData::Rows == 1`（此为实现强制执行的条件，序列沿列方向生成）。
- **有效区域**:
    - 实现使用 `dst.GetValidCol()` 作为序列长度，不参考 `dst.GetValidRow()`。
- **临时Tile**:
    - **Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品**：C++ API提供带显式 `tmp` Tile的重载，用于向量化实现路径。不带 `tmp` 的重载使用标量循环。`TileDataTmp::DType` 必须为4字节类型（`float`、`int32_t` 或 `uint32_t`）。实现将 `tmp` 转换为 `float *` 使用；应按字节数来规划tmp Tile大小，而不是按 `TileDataTmp::DType` 的类型理解。
    - **b32元素类型**（`int32_t`、`uint32_t`）：最小tmp大小 = 768字节（192个float元素）。
      向量化路径在 `tmp` 内使用两个float子缓冲区：`tmp0` 位于偏移0，`tmp1` 位于偏移 +128 floats。`tmp0` 最多持有64个float元素（256字节）用于初始分数序列，`tmp1` 最多持有64个float元素（256字节）用于累积结果。最高访问的字节偏移为128 × 4 + 64 × 4 = **768字节**（192个float元素）。
    - **b16元素类型**（`int16_t`、`uint16_t`）：最小tmp大小 = 1792字节（448个float元素）。
      向量化路径在 `tmp` 内使用四个子缓冲区：`tmp0/tmp1`（float）位于偏移0和 +128，`tmp2/tmp3`（half）位于偏移 +256和 +384（以float索引单位计）。`tmp0/tmp1` 各最多持有64个float（256字节）用于分数序列生成。`tmp2` 最多持有16个half元素（32字节）用于float到half的转换。`tmp3` 最多持有128个half元素（256字节）用于最终的half精度累积。最高访问的字节偏移为384 × 4 + 128 × 2 = **1792字节**（448个float元素）。
    - 一个方便的形状无关分配大小为2048字节（2KiB），例如 `Tile<TileType::Vec, float, 1, 512>`。
    - **Ascend 950PR/Ascend 950DT**：`tmp` Tile被接受但不使用。Ascend 950PR/Ascend 950DT硬件直接使用 `vci` 向量指令，无需临时缓冲区。

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

### 自动（带tmp）

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

### 手动（带tmp）

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

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### PTO汇编形式

```text
%dst = tci %S {descending = false} : !pto.tile<...>
# AS Level 2 (DPS)
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```
