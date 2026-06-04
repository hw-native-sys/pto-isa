# TTRANS

## 指令示意图

![TTRANS tile operation](../../../../figures/isa/TTRANS.svg)

## 简介

使用实现定义的临时 Tile 进行转置。

## 数学语义

对于二维 Tile，在有效转置域上：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{j,i} $$

确切的形状/布局及转置域取决于目标硬件（参见约束）。

## 汇编语法

PTO-AS 形式：参见 [汇编写法与操作数](../../../syntax-and-operands/assembly-model_zh.md)。

同步形式：

```text
%dst = ttrans %src : !pto.tile<...> -> !pto.tile<...>
```
降低时可能引入内部临时 Tile；C++ 内建接口需要显式传入 `tmp` 操作数。

### AS Level 1（SSA）

```text
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TTRANS(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束

!!! warning "约束"
    - **实现检查 (A2A3)**:
        - `sizeof(TileDataSrc::DType) == sizeof(TileDataDst::DType)`。
        - 源布局必须是行主序（`TileDataSrc::isRowMajor`）。
        - 元素大小必须是 `1`、`2` 或 `4` 字节。
        - 支持的元素类型按元素宽度限制如下：
        - 4 字节：`uint32_t`、`int32_t`、`float`
        - 2 字节：`uint16_t`、`int16_t`、`half`、`bfloat16_t`
        - 1 字节：`uint8_t`、`int8_t`
        - 转置大小取自 `src.GetValidRow()` / `src.GetValidCol()`。
    - **实现检查 (A5)**:
        - `sizeof(TileDataSrc::DType) == sizeof(TileDataDst::DType)`。
        - 对输入和输出的主维度强制执行 32 字节对齐约束（行主序检查 `Cols * sizeof(T) % 32 == 0`，列主序检查 `Rows * sizeof(T) % 32 == 0`）。
        - 支持的元素类型按元素宽度限制如下：
        - 4 字节：`uint32_t`、`int32_t`、`float`
        - 2 字节：`uint16_t`、`int16_t`、`half`、`bfloat16_t`
        - 1 字节：`uint8_t`、`int8_t`
        - 实现在静态 Tile 形状（`TileDataSrc::Rows/Cols`）上运算，不参考 `GetValidRow/GetValidCol`。
    - **临时 Tile**:
        - C++ API 需要 `tmp`。当选中的执行路径不需要临时存储时，部分实现可能不会使用它。
        - 基础参数：
            - `RowStride`：8-bit 元素类型为 `32`，16/32-bit 元素类型为 `16`。
            - `ElemPerBlock`：`32 / sizeof(T)`，即每个 32 字节块包含的元素数量。
            - 8-bit 类型包括 `uint8_t` 和 `int8_t`；16-bit 类型包括 `uint16_t`、`int16_t`、`half` 和 `bfloat16_t`；32-bit 类型包括 `uint32_t`、`int32_t` 和 `float`。
        - 当 stride 满足对齐要求（`dstStride % RowStride == 0`、`srcStride % ElemPerBlock == 0` 且 `srcStride / ElemPerBlock <= 255`）时，实现使用 `tmp` 走高效转置路径；否则使用 scalar copy，不需要 `tmp`。
        - 二维 Tile 转置 `[H, W] -> [W, H]`：

            ```text
            tmpSize = W * ceil(H / RowStride) * RowStride * sizeof(T)
            ```

            其中 `W` 是 `validCol`，`H` 是 `validRow`，`tmpStride` 必须按 `RowStride` 对齐。只有 stride 满足对齐条件时才需要 `tmp`。
        - `NCHW <-> NC1HWC0`：
            - 正向 `[N, C, H, W] -> [N, C1, H, W, C0]`：

                ```text
                tmpSize = H * W * ceil(C0 / RowStride) * RowStride * sizeof(T)
                ```

                其中 `C1 = (C + C0 - 1) / C0`，转置域为 `C0` 行、`H * W` 列。
            - 反向 `[N, C1, H, W, C0] -> [N, C, H, W]`：

                ```text
                tmpSize = C0 * ceil((H * W) / RowStride) * RowStride * sizeof(T)
                ```

                转置域为 `H * W` 行、`C0` 列。
        - `GNCHW <-> GNC1HWC0` 使用与 `NCHW <-> NC1HWC0` 相同的公式，`G` 作为外层 group 维度。
        - `NC1HWC0 -> FRACTAL_Z` 和 `GNC1HWC0 -> FRACTAL_Z` 不需要临时空间，会直接执行内存重排。
    - **ConvTile**:
        - 支持在`TileType::Vec`上的ConvTile的格式转换。其元素大小必须是 `1`、`2` 或 `4` 字节。元素类型限制为`uint32_t`、`int32_t`、`float`、`uint16_t`、`int16_t`、`half`、`bfloat16_t`、`uint8_t`、`int8_t`。
        - 支持ConvTile从`NCHW`到`NC1HWC0`的变换，其中`C1 == (C + C0 - 1)/C0`，HW满足对齐要求，即`H*W*sizeof(T)==0`. C0对应`c0_size`, 即`C0 * sizeof(T) == 32`。C0也可以为4。
        - 支持ConvTile从`NC1HWC0`到`FRACTAL_Z`的变换, 其中`N1 == (N + N0 - 1)/N0`。N0为16。
        - 支持ConvTile从`NCDHW`到`FRACTAL_Z_3D`的变换，目标形状为`[D * C1 * H * W, N1, N0, C0]`，其中`C1 == (C + C0 - 1) / C0`且`N1 == (N + N0 - 1) / N0`。`N0`为`16`。`C0`取决于元素宽度：4-bit数据为`64`，8-bit数据为`32`，16-bit数据为`16`，32-bit数据为`8`。临时Tile必须足够容纳一个`N * C1 * C0 * H * W`平面，以及第二段`max(N * C1 * C0 * H * W, H * W * alignedC0)`元素的区域；`alignedC0`会将`dstC0`按16/32-bit数据上取整到`16`，按8-bit数据上取整到`32`：

            ```text
            tmpTotalElem   = ncplaneElem + max(ncplaneElem, subTmpElem)
                           = N * C1 * C0 * H * W
                           + max(N * C1 * C0 * H * W, H * W * alignedC0)
            tmpAlignedElem = ceil(tmpTotalElem / elemPerBlock) * elemPerBlock
            tmpBytes       = tmpAlignedElem * sizeof(T)
            ```

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TTRANS(dst, src, tmp);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TTRANS(dst, src, tmp);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = ttrans %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
