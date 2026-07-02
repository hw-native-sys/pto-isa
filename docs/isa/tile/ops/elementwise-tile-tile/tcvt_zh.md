# pto.tcvt

`pto.tcvt` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

按指定舍入模式，对 tile 做逐元素类型转换；部分形式还允许显式指定饱和模式。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{cast}_{\mathrm{rmode},\mathrm{satmode}}\!\left(\mathrm{src}_{i,j}\right) $$

其中：

- `rmode` 控制舍入规则
- `satmode`（若显式给出）控制溢出时是否饱和

这条指令既覆盖 tile 内部的数值类型变化，也把“舍入 / 饱和是否显式暴露”做成了接口的一部分。

## 舍入模式

| 模式 | 行为 |
| --- | --- |
| `CAST_RINT` | 就近舍入，ties to even |
| `CAST_ROUND` | 就近舍入，ties away from zero |
| `CAST_FLOOR` | 向负无穷舍入 |
| `CAST_CEIL` | 向正无穷舍入 |
| `CAST_TRUNC` | 向 0 舍入 |

## 饱和模式

| 模式 | 行为 |
| --- | --- |
| `ON` | 开启饱和 |
| `OFF` | 关闭饱和 |

## 语法

### PTO-AS

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tcvt ins(%src {rmode = #pto.round_mode<CAST_RINT>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode,
                          SaturationMode satMode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode,
                          WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, WaitEvents &... events);
```

`tmp` 版本用于那些需要显式 scratch tile 的转换路径。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | 在 `dst` valid region 上逐坐标读取 |
| `%dst` | 目标 tile | 保存转换后的元素值 |
| `mode` | 舍入模式 | `CAST_RINT` / `CAST_ROUND` / `CAST_FLOOR` / `CAST_CEIL` / `CAST_TRUNC` |
| `satMode` | 可选饱和模式 | `ON` / `OFF` |
| `tmp` | 可选临时 tile | 需要显式 scratch 的路径使用 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | 逐元素转换后的结果 tile |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

!!! warning "约束"
    - `src` 与 `dst` 必须在 shape 和 valid region 上兼容。
    - 源 / 目标类型对必须属于目标 profile 支持的集合。
    - 给定类型对必须支持所选 rounding mode。
    - 对于需要显式 scratch 的路径，调用方必须使用 `tmp` 版本。
    - 关闭饱和可能改变某些低精度整数转换路径的溢出语义。

    - **临时 Tile**:
        - C++ API 提供显式传入 `tmp` Tile 的重载。在 A2/A3 上，当 `SaturationMode::OFF` 用于
          `float -> int16`、`half -> int16` 或 `half -> int8` 时，PyTorch 兼容的非饱和窄化路径会使用该临时 Tile。
          其他转换不需要 tmp 空间。
        - 实现会将 `tmp` 转换为 `int32_t *` 使用；因此应按字节数来规划 tmp Tile 大小，而不是按
          `TmpTileData::DType` 的类型理解。
        - 下列公式给出按实现使用的 32 字节向量块粒度向上取整后的最小分配大小。若 `C = 0`，不会发起需要
          tmp 的转换，所需 tmp 大小为 `0`。
        - 公共参数:
            - `R = dst.GetValidRow()`。
            - `C = dst.GetValidCol()`。
            - `SS = TileDataS::RowStride`，单位为源元素个数。
            - `REPEAT_MAX = 255`，`REPEAT_BYTE = 256`，`BLOCK_BYTE_SIZE = 32`。
        - `float -> int16`，非饱和 (`SaturationMode::OFF`):
            - 临时结果是第一步 `float -> int32` 转换产生的 `int32_t` Tile。
            - 由于 `float` 源行受 Tile 约束保证 32 字节对齐，`SS / 8` 是以 32 字节块为单位的源 repeat
              stride。
            - 对齐的主区域中，一次调用处理一行，最多处理 `REPEAT_MAX` 个 repeat，每个 repeat 为 `64` 个元素：
              $$ \text{tmpHeadBytes} = 4 \times 64 \times \min\left(\left\lfloor\frac{C}{64}\right\rfloor, 255\right) $$
            - 尾部区域中，一次调用最多处理 `REPEAT_MAX` 行，并使用源行 stride。由于向量 repeat stride
              以块为单位，空间范围按 32 字节块计算：
              $$ \text{tmpTailBytes} =
              \begin{cases}
              32 \times \left((\min(R, 255) - 1) \times \frac{SS}{8} + \left\lceil\frac{C \bmod 64}{8}\right\rceil\right), & C \bmod 64 > 0 \\
              0, & C \bmod 64 = 0
              \end{cases} $$
            - 该路径所需的最小 tmp 大小为：
              $$ \text{tmpFloatToInt16Bytes} = \max(\text{tmpHeadBytes}, \text{tmpTailBytes}) $$
            - 对主区域而言，一个紧凑的完整 repeat 上界是 `REPEAT_MAX * REPEAT_BYTE = 65280` 字节；但当 `SS`
              较大时，尾部会按源行 stride 写入，所需空间可能更大。
        - `half -> int16`，非饱和 (`SaturationMode::OFF`):
            - 实现按行处理，每行拆分为不超过 `64` 个元素的子块，并在每个子块之间复用同一段临时缓冲区。
              对于 `C > 0`，令：
              $$ H = \min(C, 64) $$
            - 该路径所需的最小 tmp 大小为：
              $$ \text{tmpHalfToInt16Bytes} = 32 \times \left\lceil\frac{H}{8}\right\rceil $$
            - 对任意非空 Tile，该路径的形状无关上界为 `256` 字节。
        - `half -> int8`，非饱和 (`SaturationMode::OFF`):
            - 实现同样按不超过 `64` 个元素的子块处理，并复用同一段 256 字节临时区域。第一步最多将 `64`
              个 `int32_t` 写入字节 `[0, 255]`；完成 `int32 -> int16` 窄化后，字节 `[0, 127]` 保存
              `int16_t` 值，字节 `[128, 255]` 被复用为 scratch。
            - `tempMaskBuf = tempAndBuf + 64` 会前进 `64 * sizeof(int16_t) = 128` 字节，因此它指向同一
              256 字节临时区域的上半部分，不需要额外再分配 256 字节。
            - 该路径所需的最小 tmp 大小为：
              $$ \text{tmpHalfToInt8Bytes} = \max\left(32 \times \left\lceil\frac{H}{8}\right\rceil,\ 128 + 32 \times \left\lceil\frac{H}{16}\right\rceil\right) $$
            - 对任意非空 Tile，该路径的形状无关上界为 `256` 字节。
        - 覆盖所有 tmp-backed TCVT 转换的总体最小值:
            - 由于 `tmpHalfToInt8Bytes >= tmpHalfToInt16Bytes`，对于同一形状，能覆盖所有会使用 tmp 的 TCVT
              转换路径的最小 tmp 大小为：
              $$ \text{tmpSizeAllBytes} = \max(\text{tmpFloatToInt16Bytes},\ \text{tmpHalfToInt8Bytes}) $$
            - 如果 Tile 非空，并且 half 路径使用形状无关的紧凑上界即可，也可写为：
              $$ \text{tmpSizeAllBytes} = \max(\text{tmpFloatToInt16Bytes},\ 256) $$
        - 对于不需要 PyTorch 兼容 tmp-backed 路径的转换，或者原生饱和行为已经满足需求时，可以继续使用不带
          `tmp` 的重载。

## 不允许的情形

!!! danger "不允许的情形"
    - 使用目标 profile 不支持的类型对。
    - 使用该类型对不支持的 rounding mode。
    - 在关闭饱和时仍假设溢出会被 clamp。

## Target-Profile 限制

`pto.tcvt` 在 CPU 仿真、A2/A3 和 A5 上都保留 PTO 可见语义，但具体支持的类型对、是否需要 scratch、以及饱和关闭后的溢出处理仍然依赖 backend。

当前 checkout 中，fp16 → int8 的非饱和路径通过带 scratch 的 helper 实现，并且会按行做子分块处理。

## 支持的转换

| 源类型 | A2A3 目标类型 | A5 目标类型 | 差异 |
|---|---|---|---|
| FP32 | FP16, FP32（仅舍入）, BF16, I16, I32, I64 | FP32, FP16, BF16, I16, I32, I64, FP8_E4M3, FP8_E5M2, H8 | A5 新增 FP8/H8 目标 |
| FP16 | FP32, I32, I16, I8, U8, S4（int4b_t） | FP32, I32, I16, I8, U8, H8 | A2A3 有 S4；A5 有 H8 |
| BF16 | FP32, I32 | FP32, I32, FP16, FP4_E1M2X2, FP4_E2M1X2 | A5 新增 FP16/FP4 目标 |
| I16 | FP16, FP32 | U8, FP16, FP32, U32, I32 | A5 新增 U8/U32/I32 |
| I32 | FP32, I16, I64, FP16（deq 路径） | FP32, I16, U16, I64, U8 | A2A3 支持 I32 -> FP16（half，deq）；A5 不支持 |
| I64 | FP32, I32 | FP32, I32 | 相同 |
| U8 | FP16 | FP16, U16 | A5 新增 U16 |
| I8 | FP16 | FP16, I16, I32 | A5 新增 I16/I32 |
| S4（int4b_t） | FP16 | N/A | A2A3 独有 |
| U32 | N/A | U8, U16, I16 | A5 独有源类型 |
| FP8_E4M3 | N/A | FP32 | A5 独有源类型 |
| FP8_E5M2 | N/A | FP32 | A5 独有源类型 |
| H8 | N/A | FP32 | A5 独有源类型 |
| FP4_E1M2X2 | N/A | BF16 | A5 独有源类型 |
| FP4_E2M1X2 | N/A | BF16 | A5 独有源类型 |

说明：

- A2A3 支持 I32 -> FP16（half，deq 路径）；A5 不支持 I32 -> FP16。
- A5 不支持 FP16 -> FP8_E4M3 和 FP16 -> FP8_E5M2。

## 示例

### 自动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  SrcT src;
  DstT dst;
  TCVT(dst, src, RoundMode::CAST_RINT);
}
```

### 显式饱和 / scratch

```cpp
using TmpT = Tile<TileType::Vec, int32_t, 16, 16>;
TmpT tmp;
TCVT(dst, src, tmp, RoundMode::CAST_TRUNC, SaturationMode::OFF);
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.tsubc](./tsubc_zh.md)
- 下一条指令：[pto.tsel](./tsel_zh.md)
