# TSCATTER

## 指令示意图

![TSCATTER tile operation](../figures/isa/TSCATTER.svg)

## 简介

TSCATTER 提供两种操作模式：

1. **索引散播（Index-based Scatter）**：使用逐元素行索引将源 Tile 的行散播到目标 Tile 中。
2. **掩码散播（Mask Scatter）**：按照掩码模式将源元素散播到目标位置，并在元素间交错填充零值。支持按行散播（`SCATTER_ROW`）和按列散播（`SCATTER_COL`）两种模式。

## 数学语义

### 索引散播

对每个源元素 `(i, j)`，写入：

$$ \mathrm{dst}_{\mathrm{idx}_{i,j},\ j} = \mathrm{src}_{i,j} $$

若多个元素映射到同一目标位置，最终值由实现定义（当前实现中以最后写入者为准）。

### 掩码散播

对于掩码模式 `P`，将源元素散播并交错填充零值。散播方向由 `ScatterAxis` 控制：

#### SCATTER_ROW（默认）

沿列方向散播，扩展列维度：

$$ \mathrm{dst}_{i, P \cdot j + \mathrm{pos}_P} = \mathrm{src}_{i,j} $$

$$ \mathrm{dst}_{i, P \cdot j + \mathrm{zeros}_P} = 0 $$

其中：
- `DstTileData::ValidCol` = `SrcTileData::ValidCol` × 扩展倍数
- `DstTileData::ValidRow` = `SrcTileData::ValidRow`

#### SCATTER_COL

沿行方向散播，扩展行维度：

$$ \mathrm{dst}_{P \cdot i + \mathrm{pos}_P, j} = \mathrm{src}_{i,j} $$

$$ \mathrm{dst}_{P \cdot i + \mathrm{zeros}_P, j} = 0 $$

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

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

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

### MaskPattern 枚举

定义于 `include/pto/common/type.hpp`：

| 值 | 模式 | 描述 | 扩展倍数 |
|---|------|------|---------|
| `P0101` | 01010101... | 每两个元素取第一个 | ×2 |
| `P1010` | 10101010... | 每两个元素取第二个 | ×2 |
| `P0001` | 00010001... | 每四个元素取第一个 | ×4 |
| `P0010` | 00100010... | 每四个元素取第二个 | ×4 |
| `P0100` | 01000100... | 每四个元素取第三个 | ×4 |
| `P1000` | 10001000... | 每四个元素取第四个 | ×4 |
| `P1111` | 11111111... | 取全部元素（等同于 TMOV） | ×1 |

### ScatterAxis 枚举

定义于 `include/pto/common/type.hpp`：

| 值 | 描述 |
|---|------|
| `SCATTER_ROW` | 沿列方向散播，扩展列维度（默认） |
| `SCATTER_COL` | 沿行方向散播，扩展行维度 |

## 约束

### 索引散播

- **实现检查 (A2A3)**:
    - `TileDataD::Loc`、`TileDataS::Loc`、`TileDataI::Loc` 必须是 `TileType::Vec`。
    - `TileDataD::DType`、`TileDataS::DType` 必须是以下之一：`int32_t`、`int16_t`、`int8_t`、`half`、`float32_t`、`uint32_t`、`uint16_t`、`uint8_t`、`bfloat16_t`。
    - `TileDataI::DType` 必须是以下之一：`int16_t`、`int32_t`、`uint16_t` 或 `uint32_t`。
    - 不对 `indexes` 值执行边界检查。
    - 静态有效边界：`TileDataD::ValidRow <= TileDataD::Rows`、`TileDataD::ValidCol <= TileDataD::Cols`、`TileDataS::ValidRow <= TileDataS::Rows`、`TileDataS::ValidCol <= TileDataS::Cols`、`TileDataI::ValidRow <= TileDataI::Rows`、`TileDataI::ValidCol <= TileDataI::Cols`。
    - `TileDataD::DType` 与 `TileDataS::DType` 必须相同。
    - 当 `TileDataD::DType` 大小为 4 字节时，`TileDataI::DType` 大小必须为 4 字节。
    - 当 `TileDataD::DType` 大小为 2 字节时，`TileDataI::DType` 大小必须为 2 字节。
    - 当 `TileDataD::DType` 大小为 1 字节时，`TileDataI::DType` 大小必须为 2 字节。
- **实现检查 (A5)**:
    - `TileDataD::Loc`、`TileDataS::Loc`、`TileDataI::Loc` 必须是 `TileType::Vec`。
    - `TileDataD::DType`、`TileDataS::DType` 必须是以下之一：`int32_t`、`int16_t`、`int8_t`、`half`、`float32_t`、`uint32_t`、`uint16_t`、`uint8_t`、`bfloat16_t`。
    - `TileDataI::DType` 必须是以下之一：`int16_t`、`int32_t`、`uint16_t` 或 `uint32_t`。
    - 不对 `indexes` 值执行边界检查。
    - 静态有效边界：`TileDataD::ValidRow <= TileDataD::Rows`、`TileDataD::ValidCol <= TileDataD::Cols`、`TileDataS::ValidRow <= TileDataS::Rows`、`TileDataS::ValidCol <= TileDataS::Cols`、`TileDataI::ValidRow <= TileDataI::Rows`、`TileDataI::ValidCol <= TileDataI::Cols`。
    - `TileDataD::DType` 与 `TileDataS::DType` 必须相同。
    - 当 `TileDataD::DType` 大小为 4 字节时，`TileDataI::DType` 大小必须为 4 字节。
    - 当 `TileDataD::DType` 大小为 2 字节时，`TileDataI::DType` 大小必须为 2 字节。
    - 当 `TileDataD::DType` 大小为 1 字节时，`TileDataI::DType` 大小必须为 2 字节。

### 掩码散播（仅 A5）

- **实现检查 (A5)**:
    - `DstTileData::Loc`、`SrcTileData::Loc` 必须是 `TileType::Vec`。
    - `DstTileData::DType`、`SrcTileData::DType` 必须是以下之一：`int32_t`、`int16_t`、`int8_t`、`half`、`float32_t`、`uint32_t`、`uint16_t`、`uint8_t`、`bfloat16_t`。
    - `DstTileData::DType` 与 `SrcTileData::DType` 必须相同。
    - `maskPattern` 必须在 `P0101` 到 `P1111` 范围内。
    - 静态有效边界：`DstTileData::ValidRow <= DstTileData::Rows`、`DstTileData::ValidCol <= DstTileData::Cols`、`SrcTileData::ValidRow <= SrcTileData::Rows`、`SrcTileData::ValidCol <= SrcTileData::Cols`。
    - `SCATTER_ROW` 模式运行时断言：
        - `SrcTileData::ValidRow` 必须等于 `DstTileData::ValidRow`。
        - `SrcTileData::ValidCol` 必须等于 `DstTileData::ValidCol * 扩展倍数`，扩展倍数取决于掩码模式（P1111 为 1，P1010/P0101 为 2，P0001/P0010/P0100/P1000 为 4）。
    - `SCATTER_COL` 模式运行时断言：
        - `SrcTileData::ValidCol` 必须等于 `DstTileData::ValidCol`。
        - `SrcTileData::ValidRow` 必须等于 `DstTileData::ValidRow * 扩展倍数`，扩展倍数取决于掩码模式（P1111 为 1，P1010/P0101 为 2，P0001/P0010/P0100/P1000 为 4）。

## 重要提示

> **警告**：在执行散播操作前，目标 Tile 缓冲区会**完全初始化为 0**（整个 Tile 大小 `Rows × Cols`），**不受 `ValidRow` 和 `ValidCol` 限制**。这意味着：
> - 分配给 `dstTile` 的整个 UB 缓冲区都会被写入零值。
> - `ValidRow`/`ValidCol` 范围之外的元素在操作后也将为零。
> - 请确保目标 Tile 的 UB 缓冲区不会与其他活跃数据重叠。

## 示例

### 索引散播（自动 Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TSCATTER(dst, src, idx);
}
```

### 索引散播（手动 Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(idx, 0x3000);
  TSCATTER(dst, src, idx);
}
```

### 掩码散播（自动 Auto）

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

### 掩码散播（手动 Manual）

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

### PTO 汇编形式

```text
%dst = tscatter %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

