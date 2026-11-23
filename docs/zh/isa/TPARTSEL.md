# TPARTSEL

## 产品支持情况<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->


## 指令示意图

![TPARTSEL tile operation](../../figures/isa/TPARTSEL.svg)

## 简介

使用掩码 Tile 在两个源 Tile 之间逐元素选择，将结果写入目标有效区域。`dst`、`src0` 和 `src1` 可以使用不同的 Tile 类型和物理形状，元素类型必须一致。接口复用 `TSEL` 实现。

## 数学语义

对目标有效区域内的每个元素 `(i, j)`：

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src0}_{i,j} & \text{if } \mathrm{mask}_{i,j}\ \text{is true} \\
\mathrm{src1}_{i,j} & \text{otherwise}
\end{cases}
$$

## 汇编语法

同步形式：

```text
%dst = tpartsel %mask, %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tpartsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tpartsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：

> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename MaskTile, typename TileDataSrc0,
          typename TileDataSrc1, typename TmpTile, typename... WaitEvents>
PTO_INST RecordEvent TPARTSEL(TileDataDst &dst, MaskTile &selMask, TileDataSrc0 &src0,
                            TileDataSrc1 &src1, TmpTile &tmp, WaitEvents &... events);
```

## 约束

### 通用约束或检查

- `dst`、`src0` 和 `src1` 必须使用相同的元素类型，且均为行主序。
- 选择域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 源 Tile 的有效形状不参与选择域计算，也不进行相等性检查；实际访问范围由目标有效形状决定。
- 两路输入须在实际访问位置提供可读取的数据，并具有满足向量加载要求的存储空间。
- 掩码 Tile 按目标定义布局中的打包谓词位解释，其存储须覆盖实际访问范围；实现不读取掩码 Tile 的有效形状。

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品实现检查

- `sizeof(TileDataDst::DType)` 必须是 `2` 或 `4` 字节。
- 各数据 Tile 使用自身的 `RowStride` 计算行地址。

### Ascend 950PR/Ascend 950DT实现检查

- `sizeof(TileDataDst::DType)` 必须是 `1`、`2`、`4` 或 `8` 字节。
- 1、2、4 字节分支使用各数据 Tile 自身的 `RowStride`；8 字节分支使用各自的 `Cols` 计算行地址。

## 临时空间

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品

`tmp` **被使用**作为设置比较掩码（`cmpmask`）的临时缓冲区，存储按 `uint32_t` 访问，并在各行之间复用。

- 16 位数据需要至少 4 个 `uint32_t` 元素，32 位数据需要至少 2 个 `uint32_t` 元素。
- 典型声明：`Tile<TileType::Vec, uint32_t, 1, 16>`。

### Ascend 950PR/Ascend 950DT

`tmp` 被接口接受，但实现**不使用**。该参数用于保持接口一致。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using DstT = Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, 16, 16>;
  using Src0T = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, 16, 16>;
  using Src1T = Tile<TileType::Vec, float, 16, 96, BLayout::RowMajor, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, uint32_t, 1, 16>;
  Src0T src0;
  Src1T src1;
  DstT dst;
  MaskT mask(16, 2);
  TmpT tmp;
  TPARTSEL(dst, mask, src0, src1, tmp);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using DstT = Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, 16, 16>;
  using Src0T = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, 16, 16>;
  using Src1T = Tile<TileType::Vec, float, 16, 96, BLayout::RowMajor, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, uint32_t, 1, 16>;
  Src0T src0;
  Src1T src1;
  DstT dst;
  MaskT mask(16, 2);
  TmpT tmp;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x3000);
  TASSIGN(dst,  0x5000);
  TASSIGN(mask, 0x6000);
  TASSIGN(tmp,  0x7000);
  TPARTSEL(dst, mask, src0, src1, tmp);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tpartsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tpartsel %mask, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO汇编形式

```text
%dst = tpartsel %mask, %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.tpartsel ins(%mask, %src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
