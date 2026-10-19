# TCVT

## 指令示意图

![TCVT tile operation](../figures/isa/TCVT.svg)

## 简介

带指定舍入模式的逐元素类型转换。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{cast}_{\mathrm{rmode}}\!\left(\mathrm{src}_{i,j}\right) $$

其中 `rmode` 是舍入策略（参见 `pto::RoundMode`）。

## C++内建接口

使用此接口应包含 `<pto/pto-inst.hpp>`。各重载定义于 `include/pto/common/pto_instr.hpp`，
`RoundMode` 和 `SaturationMode` 定义于 `include/pto/common/type.hpp`。

```cpp
template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode,
                          SaturationMode satMode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode,
                          WaitEvents &... events);
```

## 约束

- `dst` 和 `src` 必须在形状/有效区域方面兼容，如实现所要求的。
- 对于给定的 `RoundMode`，转换 `(src 元素类型) -> (dst 元素类型)` 必须被目标支持。
- **实现说明**:
    - 一种形式接受显式的 `SaturationMode`，指定的饱和行为会直接传递给实现。
    - 另一种形式不显式给出 `SaturationMode`；此时实现会针对具体类型对选择目标定义的默认饱和行为。
    - CPU_SIM 实现了上面列出的全部四种 C++ 重载；其默认饱和模式为 `SaturationMode::OFF`。
    - CPU_SIM 的 FP32 到 FP8 E4M3/E5M2 转换实现 `CAST_RINT`（最近舍入，中点取偶数）、`CAST_ROUND`
      （最近舍入，中点远离零）、`CAST_FLOOR`、`CAST_CEIL` 和 `CAST_TRUNC`。与 A5 分派一致，
      `CAST_NONE` 及 FP8 不支持的 `CAST_ODD` 回退为 `CAST_RINT`。
      舍入保留有符号零，且不依赖主机浮点舍入模式。
    - 对上述 CPU_SIM FP8 转换，饱和模式将有限值溢出及输入无穷大限制为带原符号的最大有限值
      （E4M3：448；E5M2：57344）。输入 NaN 仍为 NaN，但不保留其 payload。
      非饱和模式下，E4M3 舍入结果超出有限范围或输入为无穷大时生成 NaN。
      对 E5M2 有限输入，最近舍入（`CAST_RINT`/`CAST_ROUND`）在绝对值 >= 61440 时生成带原符号的无穷大；
      向远离零方向舍入在绝对值 > 57344 时生成无穷大。向零方向舍入对绝对值 > 57344 的输入取带原符号的
      最大有限值；输入无穷大仍保留为无穷大。这里的远离零方向指正数 `CAST_CEIL` 或负数 `CAST_FLOOR`，
      向零方向指 `CAST_TRUNC`、正数 `CAST_FLOOR` 或负数 `CAST_CEIL`。
      E4M3 的 `0x7f`/`0xff` 解码为 NaN，E5M2 区分无穷大与 NaN。
      E8M0 转换及单参数 MXType 编码构造函数保持不变。
- **A5 FP4 E2M1 到 FP8 E4M3**:
  - 直接转换 `float4_e2m1x2_t -> float8_e4m3_t` 仅在 A5 支持。所有 `RoundMode` 及两种饱和模式均精确
    保留数值（包括负零）；带临时 Tile 的重载行为相同，且不使用临时存储。
  - 源为 `Vec`、RowMajor/NoneBox、`CompactMode::Null`；物理逻辑列数至少为 64，且为 64 的倍数。
    FP4 的列数及有效列数按 nibble 计数，每个物理行占 `Cols / 2` 字节。
  - 目标为 `Vec` ND（RowMajor/NoneBox、Null）或标准 512 字节 NZ（ColMajor/RowMajor 分形、Null 或 Normal）。
    源和目标的有效行数及逻辑有效列数必须相同，支持部分及动态有效区；两者物理尺寸可不同。无效输出字节保持原值。
  - NZ 物理行数须为 16 的倍数。逻辑坐标 `(r,c)` 的字节偏移为
    `(c / 32) * panelRows * 32 + r * 32 + c % 32`；Null 使用物理 `Rows`，Normal 使用
    `ceil(validRows / 16) * 16`，与 Vec 到 Mat 的 NZ 传输一致。`RowPlusOne` 会得到不支持的编译诊断。
  - 此类型对通过寄存器字节 LUT 直接写入 ND/NZ，不分配显式 UB 表、索引 Tile 或 ND 中间 Tile。
    本扩展不为 A2/A3 或 CPU_SIM 增加 FP4 到 FP8 的实现。
- **临时Tile**:
  - **CPU_SIM**：两个带 `tmp` 的重载与各自对应的不带 `tmp` 重载结果一致。CPU_SIM 为保持源码兼容而接受
    `tmp`，但不会调用 `tmp.data()` 或访问其存储。跨后端 kernel 仍须按照目标 NPU 后端的要求声明并分配
    `tmp`。
  - **Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品**：当 `SaturationMode::OFF` 用于
    `float -> int16`、`half -> int16` 或 `half -> int8` 时，PyTorch 兼容的非饱和窄化路径会使用临时 Tile；
    其他转换不需要 tmp 空间。
  - **Ascend 950PR/Ascend 950DT**：两个带 `tmp` 的重载会接受但不使用临时 Tile；转换行为和默认饱和行为与对应的不带 `tmp` 重载一致。
  - Atlas A2/A3 实现会将 `tmp` 转换为 `int32_t *` 使用；因此应按字节数来规划tmp Tile大小，而不是按 `TmpTileData::DType` 的类型理解。
  - 下列公式给出按实现使用的32字节向量块粒度向上取整后的最小分配大小。若 `C = 0`，不会发起需要tmp的转换，所需tmp大小为 `0`。
  - **公共参数**:
    - `R = dst.GetValidRow()`。
    - `C = dst.GetValidCol()`。
    - `SS = TileDataS::RowStride`，单位为源元素个数。
    - `REPEAT_MAX = 255`，`REPEAT_BYTE = 256`，`BLOCK_BYTE_SIZE = 32`。
  - **`float -> int16`，非饱和 （`SaturationMode::OFF`）**:
    - 临时结果是第一步 `float -> int32` 转换产生的 `int32_t` Tile。
    - 由于 `float` 源行受Tile约束保证32字节对齐，`SS / 8` 是以32字节块为单位的源repeat stride。
    - 对齐的主区域中，一次调用处理一行，最多处理 `REPEAT_MAX` 个repeat，每个repeat为 `64` 个元素：
    $$ \text{tmpHeadBytes} = 4 \times 64 \times \min\left(\left\lfloor\frac{C}{64}\right\rfloor, 255\right) $$
    - 尾部区域中，一次调用最多处理 `REPEAT_MAX` 行，并使用源行stride。由于向量repeat stride以块为单位，空间范围按32字节块计算：
    $$ \text{tmpTailBytes} =
    \begin{cases}
    32 \times \left((\min(R, 255) - 1) \times \frac{SS}{8} + \left\lceil\frac{C \bmod 64}{8}\right\rceil\right), & C \bmod 64 > 0 \\
    0, & C \bmod 64 = 0
    \end{cases} $$
    - 该路径所需的最小tmp大小为：
    $$ \text{tmpFloatToInt16Bytes} = \max(\text{tmpHeadBytes}, \text{tmpTailBytes}) $$
    - 对主区域而言，一个紧凑的完整repeat上界是 `REPEAT_MAX * REPEAT_BYTE = 65280` 字节；但当 `SS` 较大时，尾部会按源行stride写入，所需空间可能更大。
  - **`half -> int16`，非饱和 （`SaturationMode::OFF`）**:
    - Atlas A2/A3 的 tmp-backed 路径先进行带饱和的 `half -> int32` 转换，再进行不饱和的
      `int32 -> int16` 窄化。因此，超出 `int16` 范围的有限值按低 16 位回绕；`-inf`、`+inf` 和 `NaN`
      的结果依次为 `0`、`-1` 和 `0`。
    - 实现按行处理，每行拆分为不超过 `64` 个元素的子块，并在每个子块之间复用同一段临时缓冲区。对于 `C > 0`，令：
    $$ H = \min(C, 64) $$
    - 该路径所需的最小tmp大小为：
    $$ \text{tmpHalfToInt16Bytes} = 32 \times \left\lceil\frac{H}{8}\right\rceil $$
    - 对任意非空Tile，该路径的形状无关上界为 `256` 字节。
  - **`half -> int8`，非饱和 （`SaturationMode::OFF`）**:
    - 实现同样按不超过 `64` 个元素的子块处理，并复用同一段256字节临时区域。第一步最多将 `64` 个 `int32_t` 写入字节 `[0, 255]`；完成 `int32 -> int16` 窄化后，字节 `[0, 127]` 保存 `int16_t` 值，字节 `[128, 255]` 被复用为scratch。
    - `tempMaskBuf = tempAndBuf + 64` 会前进 `64 * sizeof(int16_t) = 128` 字节，因此它指向同一256字节临时区域的上半部分，不需要额外再分配256字节。
    - 该路径所需的最小tmp大小为：
    $$ \text{tmpHalfToInt8Bytes} = \max\left(32 \times \left\lceil\frac{H}{8}\right\rceil,\ 128 + 32 \times \left\lceil\frac{H}{16}\right\rceil\right) $$
    - 对任意非空Tile，该路径的形状无关上界为 `256` 字节。
  - **覆盖所有tmp-backed TCVT转换的总体最小值**:
    - 由于 `tmpHalfToInt8Bytes >= tmpHalfToInt16Bytes`，对于同一形状，能覆盖所有会使用tmp的TCVT转换路径的最小tmp大小为：
    $$ \text{tmpSizeAllBytes} = \max(\text{tmpFloatToInt16Bytes},\ \text{tmpHalfToInt8Bytes}) $$
    - 如果Tile非空，并且half路径使用形状无关的紧凑上界即可，也可写为：
    $$ \text{tmpSizeAllBytes} = \max(\text{tmpFloatToInt16Bytes},\ 256) $$
  - 在 NPU 后端上，对于不需要 PyTorch 兼容 tmp-backed 路径的转换，或者原生饱和行为已经满足需求时，
    可以继续使用不带 `tmp` 的重载。

## 支持的转换（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品与Ascend 950PR/Ascend 950DT并排对比）

| 源类型 | Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品目标类型 | Ascend 950PR/Ascend 950DT目标类型 | 差异 |
|---|---|---|---|
| FP32 | FP16, FP32（仅舍入）, BF16, I16, I32, I64 | FP32, FP16, BF16, I16, I32, I64, FP8_E4M3, FP8_E5M2, H8 | Ascend 950PR/Ascend 950DT新增FP8/H8目标 |
| FP16 | FP32, I32, I16, I8, U8, S4（int4b_t） | FP32, I32, I16, I8, U8, H8 | Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品有S4；Ascend 950PR/Ascend 950DT有H8 |
| BF16 | FP32, I32 | FP32, I32, FP16, FP4_E1M2X2, FP4_E2M1X2 | Ascend 950PR/Ascend 950DT新增FP16/FP4目标 |
| I16 | FP16, FP32 | U8, FP16, FP32, U32, I32 | Ascend 950PR/Ascend 950DT新增U8/U32/I32 |
| I32 | FP32, I16, I64, FP16（deq路径） | FP32, I16, U16, I64, U8 | Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品支持I32 -> FP16（half，deq）；Ascend 950PR/Ascend 950DT不支持 |
| I64 | FP32, I32 | FP32, I32 | 相同 |
| U8 | FP16 | FP16, U16 | Ascend 950PR/Ascend 950DT新增U16 |
| I8 | FP16 | FP16, I16, I32 | Ascend 950PR/Ascend 950DT新增I16/I32 |
| S4（int4b_t） | FP16 | N/A | Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品独有 |
| U32 | N/A | U8, U16, I16 | Ascend 950PR/Ascend 950DT独有源类型 |
| FP8_E4M3 | N/A | FP32 | Ascend 950PR/Ascend 950DT独有源类型 |
| FP8_E5M2 | N/A | FP32 | Ascend 950PR/Ascend 950DT独有源类型 |
| H8 | N/A | FP32 | Ascend 950PR/Ascend 950DT独有源类型 |
| FP4_E1M2X2 | N/A | BF16 | Ascend 950PR/Ascend 950DT独有源类型 |
| FP4_E2M1X2 | N/A | BF16, FP8_E4M3 | Ascend 950PR/Ascend 950DT独有源类型 |

说明：

- 关键差异：Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品支持I32 -> FP16（half，deq路径），Ascend 950PR/Ascend 950DT不支持I32 -> FP16。
- Ascend 950PR/Ascend 950DT上不支持FP16 -> FP8_E4M3和FP16 -> FP8_E5M2。

## 示例

### 自动（Auto）

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

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TCVT(dst, src, RoundMode::CAST_RINT);
}
```
