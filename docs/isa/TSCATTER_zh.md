# TSCATTER

## 指令示意图

![TSCATTER tile operation](../figures/isa/TSCATTER.svg)

## 简介

TSCATTER提供两种操作模式：

1. **索引散播（Index-based Scatter）**：使用逐元素的扁平元素偏移，将源 Tile 的元素散播到目标 Tile 中。
2. **掩码散播（Mask Scatter）**：按照掩码模式将源元素散播到目标位置，并在元素间交错填充零值。支持按行散播（`SCATTER_ROW`）和按列散播（`SCATTER_COL`）两种模式。

## 数学语义

### 索引散播

设 `R = idx.GetValidRow()`、`C = idx.GetValidCol()`。对 `0 <= i < R`、`0 <= j < C`，写入：

$$ \mathrm{dst.data()}[\mathrm{idx}_{i,j}] = \mathrm{src}_{i,j} $$

`idx[i,j]` 是相对于 `dst.data()` 的元素偏移，不是字节偏移或行号。RowMajor 目标中坐标 `(r,c)` 的索引为 `r * DstTile::Cols + c`；ColMajor 中为 `c * DstTile::Rows + r`。源有效区域须覆盖索引有效区域。调用者须保证索引非负且小于目标物理元素数；实现不做边界检查。

若多个元素映射到同一目标位置，最终值由实现定义，不保证写入顺序。

### 掩码散播

对于掩码模式 `P`，将源元素散播并交错填充零值。散播方向由 `ScatterAxis` 控制：

设扩展倍数为 `f`，选中位置为 `pos_P`，`q` 遍历 `0 <= q < f` 中除 `pos_P` 外的其他位置。`P0101/P1010` 的 `pos_P` 分别为 0/1；`P0001/P0010/P0100/P1000` 分别为 0/1/2/3；`P1111` 为 0。

#### SCATTER_ROW（默认）

沿列方向散播，扩展列维度：

$$ \mathrm{dst}_{i, f \cdot j + \mathrm{pos}_P} = \mathrm{src}_{i,j} $$

$$ \mathrm{dst}_{i, f \cdot j + q} = 0 $$

其中：

- `DstTileData::ValidCol` = `SrcTileData::ValidCol` × 扩展倍数
- `DstTileData::ValidRow` = `SrcTileData::ValidRow`

#### SCATTER_COL

沿行方向散播，扩展行维度：

$$ \mathrm{dst}_{f \cdot i + \mathrm{pos}_P, j} = \mathrm{src}_{i,j} $$

$$ \mathrm{dst}_{f \cdot i + q, j} = 0 $$

其中：

- `DstTileData::ValidRow` = `SrcTileData::ValidRow` × 扩展倍数
- `DstTileData::ValidCol` = `SrcTileData::ValidCol`

#### 扩展倍数

- `P1010` 或 `P0101`：扩展倍数 = 2
- `P0001`、`P0010`、`P0100`、`P1000`：扩展倍数 = 4
- `P1111`：扩展倍数 = 1（等同于 `TMOV`）

## 汇编语法

同步形式：

```text
%dst = tscatter %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

### 索引散播

```cpp
template <typename TileDataD, typename TileDataS, typename TileDataI, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(TileDataD& dst, TileDataS& src, TileDataI& indexes, WaitEvents&... events);
```

### 掩码散播

```cpp
template <MaskPattern maskPattern = MaskPattern::P1111, auto ScatterType = ScatterAxis::SCATTER_ROW,
          typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(DstTileData& dst, SrcTileData& src, WaitEvents&... events);
```

### MaskPattern枚举

定义于 `include/pto/common/type.hpp`：

| 值 | 模式 | 描述 | 扩展倍数 |
|---|------|------|---------|
| `P0101` | 01010101... | 每两个元素取第一个 | ×2 |
| `P1010` | 10101010... | 每两个元素取第二个 | ×2 |
| `P0001` | 00010001... | 每四个元素取第一个 | ×4 |
| `P0010` | 00100010... | 每四个元素取第二个 | ×4 |
| `P0100` | 01000100... | 每四个元素取第三个 | ×4 |
| `P1000` | 10001000... | 每四个元素取第四个 | ×4 |
| `P1111` | 11111111... | 取全部元素（等同于TMOV） | ×1 |

### ScatterAxis枚举

定义于 `include/pto/common/type.hpp`：

| 值 | 描述 |
|---|------|
| `SCATTER_ROW` | 沿列方向散播，扩展列维度（默认） |
| `SCATTER_COL` | 沿行方向散播，扩展行维度 |

## 约束

### 索引散播

- **实现检查 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**:
    - `TileDataD::Loc`、`TileDataS::Loc`、`TileDataI::Loc` 必须是 `TileType::Vec`。
    - `TileDataD::DType`、`TileDataS::DType` 必须是以下之一：`int32_t`、`int16_t`、`int8_t`、`half`、`float16_t`、`float32_t`、`uint32_t`、`uint16_t`、`uint8_t`、`bfloat16_t`。
    - `TileDataI::DType` 必须是以下之一：`int16_t`、`int32_t`、`uint16_t` 或 `uint32_t`。
    - 不对 `indexes` 值执行边界检查。
    - 静态有效边界：`TileDataD::ValidRow <= TileDataD::Rows`、`TileDataD::ValidCol <= TileDataD::Cols`、`TileDataS::ValidRow <= TileDataS::Rows`、`TileDataS::ValidCol <= TileDataS::Cols`、`TileDataI::ValidRow <= TileDataI::Rows`、`TileDataI::ValidCol <= TileDataI::Cols`。
    - `TileDataD::DType` 与 `TileDataS::DType` 必须相同。
    - 当 `TileDataD::DType` 大小为4字节时，`TileDataI::DType` 大小必须为4字节。
    - 当 `TileDataD::DType` 大小为2字节时，`TileDataI::DType` 大小必须为2字节。
    - 当 `TileDataD::DType` 大小为1字节时，`TileDataI::DType` 大小必须为2字节。
- **实现检查 (Ascend 950PR/Ascend 950DT)**:
    - `TileDataD::Loc`、`TileDataS::Loc`、`TileDataI::Loc` 必须是 `TileType::Vec`。
    - `TileDataD::DType`、`TileDataS::DType` 必须是以下之一：`int32_t`、`int16_t`、`int8_t`、`half`、`float16_t`、`float32_t`、`uint32_t`、`int64_t`、`uint64_t`、`uint16_t`、`uint8_t`、`bfloat16_t`。
    - `TileDataI::DType` 必须是以下之一：`int16_t`、`int32_t`、`uint16_t` 或 `uint32_t`。
    - 不对 `indexes` 值执行边界检查。
    - 静态有效边界：`TileDataD::ValidRow <= TileDataD::Rows`、`TileDataD::ValidCol <= TileDataD::Cols`、`TileDataS::ValidRow <= TileDataS::Rows`、`TileDataS::ValidCol <= TileDataS::Cols`、`TileDataI::ValidRow <= TileDataI::Rows`、`TileDataI::ValidCol <= TileDataI::Cols`。
    - `TileDataD::DType` 与 `TileDataS::DType` 必须相同。
    - 当 `TileDataD::DType` 大小为4字节时，`TileDataI::DType` 大小必须为4字节。
    - 当 `TileDataD::DType` 大小为2字节时，`TileDataI::DType` 大小必须为2字节。
    - 当 `TileDataD::DType` 大小为1字节时，`TileDataI::DType` 大小必须为2字节。

    - 64 位数据要求 `int32_t` / `uint32_t` 索引；源和索引 Tile 支持 RowMajor/ColMajor，分别使用物理行列步长。

### 掩码散播

- **实现检查 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**:
    - `DstTileData::Loc`、`SrcTileData::Loc` 必须是 `TileType::Vec`。
    - `DstTileData::DType`、`SrcTileData::DType` 必须是以下之一：`int32_t`、`int16_t`、`int8_t`、`half`、`float16_t`、`float32_t`、`uint32_t`、`uint16_t`、`uint8_t`、`bfloat16_t`。
    - `DstTileData::DType` 与 `SrcTileData::DType` 必须相同。
    - `maskPattern` 必须在 `P0101` 到 `P1111` 范围内。
    - 静态有效边界：`DstTileData::ValidCol <= DstTileData::Cols`、`SrcTileData::ValidCol <= SrcTileData::Cols`、`DstTileData::ValidRow <= DstTileData::Rows`、`SrcTileData::ValidRow <= SrcTileData::Rows`。
    - `P1111` 模式等同 `TMOV`：要求 `validRow` 和 `validCol` 分别匹配，内部通过 `TMOV_IMPL` 实现。
- **实现检查 (Ascend 950PR/Ascend 950DT)**:
    - `DstTileData::Loc`、`SrcTileData::Loc` 必须是 `TileType::Vec`。
    - `DstTileData::DType`、`SrcTileData::DType` 必须是以下之一：`int32_t`、`int16_t`、`int8_t`、`half`、`float16_t`、`float32_t`、`uint32_t`、`int64_t`、`uint64_t`、`uint16_t`、`uint8_t`、`bfloat16_t`。
    - `DstTileData::DType` 与 `SrcTileData::DType` 必须相同。
    - `maskPattern` 必须在 `P0101` 到 `P1111` 范围内。
    - 静态有效边界：`DstTileData::ValidRow <= DstTileData::Rows`、`DstTileData::ValidCol <= DstTileData::Cols`、`SrcTileData::ValidRow <= SrcTileData::Rows`、`SrcTileData::ValidCol <= SrcTileData::Cols`。
    - `SCATTER_ROW` 模式运行时断言：
        - `src.GetValidRow()` 必须等于 `dst.GetValidRow()`。
        - `dst.GetValidCol()` 必须等于 `src.GetValidCol() × 扩展倍数`，扩展倍数取决于掩码模式（P1111为1，P1010/P0101为2，P0001/P0010/P0100/P1000为4）。
    - `SCATTER_COL` 模式运行时断言：
        - `src.GetValidCol()` 必须等于 `dst.GetValidCol()`。
        - `dst.GetValidRow()` 必须等于 `src.GetValidRow() × 扩展倍数`，扩展倍数取决于掩码模式（P1111为1，P1010/P0101为2，P0001/P0010/P0100/P1000为4）。

    - 64 位掩码散播要求 RowMajor 源和目标，支持六种扩展模式及两个 `ScatterAxis`。
    - `P1111` 委托给 [TMOV](TMOV_zh.md)，不执行整块清零；当前 Ascend 950PR/Ascend 950DT 不支持该模式的 `int64_t` / `uint64_t` Vec→Vec 移动。

## 重要提示

A5 的索引散播和非 `P1111` 掩码散播会先清零整个目标物理 Tile（`Rows * Cols` 个元素），再写入选中的位置。因此未被选中的位置（包括物理填充）为零；索引显式选中的填充位置仍会写入源值。源、索引与目标的活跃存储不得重叠。

CPU 模拟器的索引散播只更新索引指定位置，保留其他位置；空索引有效区域也不写入。跨后端代码若需要一致的零填充，应显式初始化目标。`P1111` 遵循 `TMOV` 的写入语义，不适用上述整块清零规则。

## 示例

### 索引散播（自动Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TSCATTER(dst, src, idx);
}
```

### 索引散播（手动Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(idx, 0x3000);
  TSCATTER(dst, src, idx);
}
```

### 掩码散播（自动Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_mask_auto() {
  // P1010: 目标大小 = 源大小 × 2
  using SrcTileT = Tile<TileType::Vec, half, 16, 64>;
  using DstTileT = Tile<TileType::Vec, half, 16, 128>;
  SrcTileT src;
  DstTileT dst;
  TSCATTER<MaskPattern::P1010>(dst, src);
}

void example_mask_p1000() {
  // P1000: 目标大小 = 源大小 × 4
  using SrcTileT = Tile<TileType::Vec, float, 16, 64>;
  using DstTileT = Tile<TileType::Vec, float, 16, 256>;
  SrcTileT src;
  DstTileT dst;
  TSCATTER<MaskPattern::P1000>(dst, src);
}

void example_mask_scatter_col() {
  // SCATTER_COL: 沿行方向散播，扩展行维度
  // P1010: 目标行数 = 源行数 × 2
  using SrcTileT = Tile<TileType::Vec, half, 64, 16>;
  using DstTileT = Tile<TileType::Vec, half, 128, 16>;
  SrcTileT src;
  DstTileT dst;
  TSCATTER<MaskPattern::P1010, ScatterAxis::SCATTER_COL>(dst, src);
}
```

### 掩码散播（手动Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_mask_manual() {
  using SrcTileT = Tile<TileType::Vec, half, 16, 64>;
  using DstTileT = Tile<TileType::Vec, half, 16, 128>;
  SrcTileT src;
  DstTileT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TSCATTER<MaskPattern::P1010>(dst, src);
}

void example_mask_manual_scatter_col() {
  // SCATTER_COL 手动绑定模式
  using SrcTileT = Tile<TileType::Vec, half, 64, 16>;
  using DstTileT = Tile<TileType::Vec, half, 128, 16>;
  SrcTileT src;
  DstTileT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TSCATTER<MaskPattern::P1010, ScatterAxis::SCATTER_COL>(dst, src);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO汇编形式

```text
%dst = tscatter %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
