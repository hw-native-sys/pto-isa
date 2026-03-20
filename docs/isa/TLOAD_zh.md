# TLOAD

## 指令示意图

![TLOAD tile operation](../figures/isa/TLOAD.svg)

## 简介

从 GlobalTensor (GM) 加载数据到 Tile。

## 数学语义

符号表示取决于 `GlobalTensor` 的形状/步长和 `Tile` 的布局。概念上（二维视图，带基础偏移量）：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{r_0 + i,\; c_0 + j} $$

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%t0 = tload %sv[%c0, %c0] : (!pto.memref<...>, index, index) -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
!pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

### AS Level 2（DPS）

```text
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);
```

## 约束

- **实现检查 (A2A3)**:
    - `TileData::DType` 必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、`half`、`bfloat16_t`、`float`。
    - 目标 tile 位置必须是 `TileType::Vec` 或 `TileType::Mat`。
    - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`。
    - 运行时：所有 `src.GetShape(dim)` 值和 `dst.GetValidRow()/GetValidCol()` 必须 `> 0`。
    - `TileType::Vec` 加载仅支持匹配的布局：ND->ND、DN->DN、NZ->NZ。
    - `TileType::Mat` 加载支持：ND->ND、DN->DN、NZ->NZ，以及 ND->NZ 和 DN->ZN。
    - 对于 ND->NZ 或 DN->ZN：`GlobalData::staticShape[0..2] == 1` 且 `TileData::SFractalSize == 512`。
    - 对于 `int64_t/uint64_t`，仅支持 ND->ND 或 DN->DN。
- **实现检查 (A5)**:
    - `sizeof(TileData::DType)` 必须是 `1`、`2`、`4` 或 `8` 字节，且必须匹配 `sizeof(GlobalData::DType)`。
    - 对于 `int64_t/uint64_t`，`TileData::PadVal` 必须是 `PadValue::Null` 或 `PadValue::Zero`。
    - `TileType::Vec` 加载需要以下布局对之一：
    - ND 使用行主序 + `SLayout::NoneBox`（ND->ND），
    - DN 使用列主序 + `SLayout::NoneBox`（DN->DN），
    - NZ 使用 `SLayout::RowMajor`（NZ->NZ）。
    - 对于使用编译时已知形状的行主序 ND->ND，`TileData::ValidCol` 必须等于 `GlobalData::staticShape[4]`，且 `TileData::ValidRow` 必须等于 `GlobalData::staticShape[0..3]` 的乘积。
    - `TileType::Mat` 加载还受到 `TLoadCubeCheck` 的约束（例如，仅特定的 ND/DN/NZ 转换和 L1 大小限制）。
    - `TileType::Mat` 加载还处理 mx 格式的加载，包括 `MX_A_ZZ/MX_A_ND/MX_A_DN` 到 ZZ（用于 scalarA）和 `MX_B_NN/MX_B_ND/MX_B_DN` 到 NN（用于 scalarB）。
    - 对于 `MX_A_ZZ/MX_B_NN`：`GlobalData::staticShape[3] == 16` 且 `GlobalData::staticShape[4] == 2`。
    - 对于 `MX_A_ND/MX_ADN/MX_B_ND/MX_B_DN`：`GlobalData::staticShape[0] == 1` 且 `GlobalData::staticShape[1] == 1` 且 `GlobalData::staticShape[4] == 2`。
    - 对于 scaleA，`dst.GetValidCol() % 2 == 0`。
    - 对于 scaleB，`dst.GetValidRow() % 2 == 0`。

- **有效区域**:
    - 实现使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为传输大小。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_auto(__gm__ T* in) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gin(in);
  TileT t;
  TLOAD(t, gin);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_manual(__gm__ T* in) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gin(in);
  TileT t;
  TASSIGN(t, 0x1000);
  TLOAD(t, gin);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tload %mem : !pto.partition_tensor_view<MxNxdtype> ->
```

### PTO 汇编形式

```text
%t0 = tload %sv[%c0, %c0] : (!pto.memref<...>, index, index) -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tload ins(%mem : !pto.partition_tensor_view<MxNxdtype>) outs(%dst : !pto.tile_buf<...>)
```

