# TEXTRACT

## 指令示意图

![TEXTRACT tile operation](../figures/isa/TEXTRACT.svg)

## 简介

从较大的源Tile中提取较小的子Tile。

## 数学语义

概念上从较大的 `src` Tile中，以 `(indexRow, indexCol)` 为起点复制一个较小窗口到 `dst`。确切的映射取决于tile布局。

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。对于 `0 <= i < R` 和 `0 <= j < C`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{indexRow}+i,\; \mathrm{indexCol}+j} $$

## 汇编语法

同步形式：

```text
%dst = textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, FpTileData &fp,
                              uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);
```

`STPhase` 重载把 unit flag（单元标志）写入 L0C 搬出指令，用于与 `TMATMUL<AccPhase>` 配对完成
Cube 到 Fixpipe 的硬件同步，从而省去显式的 `set_flag`/`wait_flag`。
仅在存在对应后端实现的目标上暴露（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品、
Ascend 950PR/Ascend 950DT 和 CPU 模拟器），适用范围与配对规则见下方实现检查。

`TEXTRACT_FP(...)` 为历史 fp 量化形式保留源码兼容入口，并直接映射到无 `mode` 的
`TEXTRACT_IMPL(dst, src, fp, indexRow, indexCol)` 路径。规范同名 `TEXTRACT(..., fp, ...)`
重载仅在 `FpTileData::Loc == TileType::Scaling` 时参与匹配。
规范接口还提供显式 `AccToVecMode` 形式，用于目标支持的 Acc-to-Vec 路由。
`TEXTRACT_FP` 同样提供 `STPhase` 版本，与 `TMOV_FP` / `TSTORE_FP` 对齐，语义与规范接口一致。

## 约束

### 通用约束或检查

- `STPhase` 重载（unit flag）仅支持 `TileType::Acc -> TileType::Mat`（L0C→L1）路径；
  目标为 `TileType::Vec` 时编译期报错。该重载在
  Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品、Ascend 950PR/Ascend 950DT 与 CPU 模拟器上暴露。
- 搬出侧的取值规则与累加侧不同，不是简单对应关系：
  产生该 L0C 结果的 `TMATMUL` 必须已经是 `AccPhase::Final`，即数据已就绪；
  `STPhase::Final` 用于最后一次搬出并释放 unit flag；
  `STPhase::Partial` 只用于同一块 L0C 分多次搬出时的非末次那几条，它不释放 unit flag。
  把 `STPhase::Partial` 与 `AccPhase::Partial` 配对会让 fixpipe 等待一个不会到来的标志而挂死，
  该现象已在 Ascend 950PR 仿真器上复现。
- Acc→Mat 搬出已通过 A3 上板测试，`tmov_acc2mat` 覆盖了 `STPhase::Final` 和
  `STPhase::Partial` 后接 `STPhase::Final` 的场景。
  Ascend 950PR 仿真测试已通过；在 Ascend 950PR 板机、CANN 9.2.0 环境下，
  `textract` 的 7 个定向用例（`case1`、`case21`–`case26`）全部通过，max diff 均为 0，
  覆盖 NZ512/NZ1024、`Final`、`Partial` 后接 `Final`、`Unspecified` 和 K 切分累加。
- 对于同 dtype 抽取/布局路径，`DstTileData::DType` 必须等于 `SrcTileData::DType`。
  Acc 转换和量化路径使用下述后端特定 dtype 组合。
- A2A3 原有 Mat/Acc 提取路径检查以下物理矩形边界（下述小 M Mat→Left 路径改为分别检查有效窗口和完整分形读取范围）：
    - `indexRow + DstTileData::Rows <= SrcTileData::Rows`
    - `indexCol + DstTileData::Cols <= SrcTileData::Cols`

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品实现检查

对于 Mat→Left/Right 布局提取：

- 支持的元素类型：`int8_t`、`half`、`bfloat16_t`、`float`。
- 源布局必须满足以下已检查到的Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品提取布局之一：
    - `(SFractal == ColMajor && isRowMajor)`，或
    - `(SFractal == RowMajor && !isRowMajor)`。
- 在以 `TileType::Left` 为目标的GEMV场景中，已检查到的源布局还允许 `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`。
- 目标必须是 `TileType::Left` 或 `TileType::Right`，并具有目标支持的布局配置。

### Ascend 950PR/Ascend 950DT实现检查

- 支持的元素类型：`int8_t`、`hifloat8_t`、`float8_e5m2_t`、`float8_e4m3_t`、`half`、`bfloat16_t`、`float`、`float4_e2m1x2_t`、`float4_e1m2x2_t`、`float8_e8m0_t`。
- 普通 Acc→Mat 提取还支持 `int32_t -> int32_t`。
- 源布局必须满足以下已检查到的Ascend 950PR/Ascend 950DT提取布局之一：
    - 对于 `Left` / `Right`：`(SFractal == ColMajor && isRowMajor)` 或 `(SFractal == RowMajor && !isRowMajor)`
    - 对于 `ScaleLeft`：`(SFractal == RowMajor && isRowMajor)`
    - 对于 `ScaleRight`：`(SFractal == ColMajor && !isRowMajor)`
- 在以 `Left` 为目标的GEMV场景中，已检查到的源布局还允许 `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`。
- 目标支持 `TileType::Mat -> TileType::Left/Right/Scale`、`TileType::Acc -> TileType::Mat`（ReLU 和量化形式受 dtype、布局限制，见下方 Acc→Mat NZ 约束）、`TileType::Acc -> TileType::Vec`，以及特定的 `TileType::Vec -> TileType::Mat` 提取路径。
- 规范向量量化 `TEXTRACT(..., fp, ...)` 形式额外要求提供 `FpTileData` Scaling 操作数。
  `TEXTRACT_FP(...)` 仍作为源码兼容历史 alias 保留，并由所选后端实现继续检查合法性。
- 向量量化 Acc-to-Vec 形式仅在存在对应后端实现的目标上暴露
  （A5、kirin9030、kirinX90 和 CPU 模拟器），接受
  `mode = AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`。
- 对于 `TileType::Acc -> TileType::Vec`，当目标为32位类型（`float`/`int32_t`）且使用 `DualModeSplitN` 时，切分前的 `ValidCol` 必须是 `32` 的整数倍。

### A5 Acc→Mat NZ 布局转换

以下路径的目标为 NZ 布局（`BLayout::ColMajor`、`SLayout::RowMajor`）：

| 源 → 目标类型 | 目标布局 | 支持的操作 |
| --- | --- | --- |
| `float → half/bfloat16_t` | NZ1024，每组 32 列 | 普通转换，可选 `NoRelu` 或 `NormalRelu`；标量/向量量化在编译期拒绝 |
| `int32_t → int32_t` | NZ512，每组 8 列 | `NoQuant`、`NoRelu` 下保持位模式的搬运；ReLU 在编译期拒绝 |

普通 `TEXTRACT(dst, src, row, col)` 与显式 `STPhase` 形式均可进入上述路径。
普通形式使用 `STPhase::Unspecified`，须显式同步 Cube 与 Fixpipe。
ReLU 需选择显式指定 `ReluPreMode` 模板实参的重载。

half/bfloat16 路径按每组最多 32 列执行 Fixpipe NZ→ND。组内目标行跨度为 32 个元素，
各组起始偏移为 `group * DstTileData::Rows * 32` 个元素；末组只有 16 个有效列时仅写这 16 列。
int32 路径通过 float 指令重载的 channel split 生成 8 列分组，不进行数值类型转换或浮点运算。

调用方须满足以下存储要求：

- 源使用常规 16 列 L0C 布局，物理跨度取 `SrcTileData::Rows`；
  此路径不会根据源有效行数推导 compact 跨度。
- L0C 源窗口起始地址须 64 字节对齐（对于基址已对齐的常规 L0C tile，`indexCol` 为 16 的倍数）；
  L1 目标地址须 32 字节对齐。
- 有效行列数须为正且不超过目标物理形状，完整源窗口须落在源已分配的存储范围内。
- half/bfloat16 NZ1024 有效列数须为 16 的倍数。辅助函数使用 `PTO_ASSERT` 检查这一条件和
  源存储边界，仅在定义 `_DEBUG` 时生效；这不是无条件的运行时检查，也不检查源有效形状。
- int32 NZ512 指令将有效列数向上补齐到 8 的倍数。补齐后的窗口须位于源和目标存储范围内，
  最后一个块内的额外列可能被写入。此分支未新增运行时边界检查；
  要求列 padding 保持不变的测试使用 8 的倍数作为有效列数。

half/bfloat16 NZ1024 各条指令的 flag 随外层 phase 设置：

| 外层 phase | 非末组 | 末组 |
| --- | --- | --- |
| `Final` | `Partial` | `Final` |
| `Partial` | `Partial` | `Partial` |
| `Unspecified` | `Unspecified` | `Unspecified` |

因此外层为 `Partial` 时，调用方仍须对同一累加器执行后续 `Final` 提取。
int32 路径只发出一条指令，原样传递外层 phase。

`textract_acc2mat_layout` 覆盖原生 NZ 布局对照、phase 配对、K 切分累加、偏移、
部分有效形状、padding、输出保护区及代表性的 int32 特殊位模式。
case22–23 覆盖 half/bfloat16 ReLU；case24–26 覆盖 int32/half/bfloat16 的普通重载。

在 CPU 模拟中，若通过 `TMATMUL` 将 FP8/HIF8 输入计算到源 Acc Tile，应在绑定 Tile 前选择 A5。
输入类型和精度约束遵循 [TMATMUL](TMATMUL_zh.md)；源 Acc Tile 的元素类型为 `float`，不是 FP8/HIF8。
架构配置见 [选择模拟目标架构](../coding/cpu_sim_zh.md)。

### 小 M Mat→Left 提取（A2A3 和 A5）

A2A3 和 A5 上，普通 `TEXTRACT(dst, src, indexRow, indexCol, events...)` 重载在以下条件下自动
选择 `load_cbuf_to_ca`（L1→L0A 搬运，也称 load2d）。无需新增模式或公共重载，原有事件接口不变。
应使用不带 `ReluPreMode` 模板实参的重载。即使显式指定 `ReluPreMode::NoRelu`，
在 A2A3 上也会选中 Acc→Mat 重载，该重载不支持此 Mat→Left 路径。

| 属性 | 要求 |
| --- | --- |
| 元素类型 | 源与目标均为 `half`，或均为 `bfloat16_t` |
| 源 | `TileType::Mat`，NZ512：`BLayout::ColMajor`、`SLayout::RowMajor`、分形大小 512 |
| 源物理形状 | 行数、列数均为 16 的倍数 |
| A2A3 目标 | `TileType::Left`，ZZ512：`BLayout::RowMajor`、`SLayout::RowMajor`、分形大小 512 |
| A5 目标 | `TileType::Left`，NZ512：`BLayout::ColMajor`、`SLayout::RowMajor`、分形大小 512 |
| 目标物理形状 | 恰好 16 行，列数为 16 的倍数 |
| 目标 Compact 模式 | `CompactMode::Null`（普通 `TileLeft`）或 `CompactMode::Normal`（`TileLeftCompact`） |
| 有效形状 | 支持静态或动态形状；此 load2d 路径处理 1～15 个有效行 |

令 `M = dst.GetValidRow()`、`K = dst.GetValidCol()`、`r = indexRow`、`c = indexCol`。
两个索引均为无符号类型 `uint16_t`。动态有效行列数须在调用前设置；`TEXTRACT` 使用已有有效形状，
不会根据索引自动推导或更新有效形状。小 M 路径要求：

```text
1 <= M <= 15
1 <= K <= DstTileData::Cols
r + M <= src.GetValidRow()
c + K <= src.GetValidCol()
c % 16 == 0
```

`r` 无需按 4 或 16 对齐。只要满足这些边界及下述存储边界，有效窗口可以跨越源内部的 16 行分形边界。
源有效行列数也可以是动态值。动态目标在运行时 `M = 16` 时使用原有普通/Compact 实现，
仍要求行、列索引按 16 对齐。A2A3 还保留上述物理矩形边界；A5 保留 Compact 按对齐后有效宽度搬运的原有规则。
静态完整 M、转置、Right、GEMV、其他数据类型以及 A2A3/A5 以外的架构路径保持原有行为。本次扩展不增加空有效形状、超出物理容量的有效形状，
以及 Mat→Left 的 ReLU、量化或 `STPhase` 变体支持。

**物理读写范围。** 每个 repeat 仍搬运完整的 512B 分形，减小有效 M 不会减少搬运字节数。
使用整数除法定义 `ceil16(K) = ((K + 15) / 16) * 16`，实际搬运宽度为：

```text
copyCols = DstTileData::Cols         // 普通 TileLeft
copyCols = ceil16(K)                // TileLeftCompact
```

还须满足以下边界，其中 `srcEndBytes` 是相对源 tile 基址的读取末尾（字节，右开区间）：

```text
c + copyCols <= SrcTileData::Cols
copyCols <= DstTileData::Cols
srcEndBytes = (c + copyCols - 16) * SrcTileData::Rows * 2 + r * 32 + 512
srcEndBytes <= SrcTileData::Numel * 2
```

因此，Compact 目标的物理列数可以大于源，只要对齐后的有效搬运宽度满足边界；普通目标则必须容纳
完整物理宽度的搬运。在 A2A3 上，该放宽仅适用于小 M：动态 `M = 16` 时，
即便是 Compact tile，仍要求 `c + DstTileData::Cols <= SrcTileData::Cols`。
A5 动态 `M = 16` 保留 Compact 按对齐后有效宽度搬运的规则，不增加上述 A2A3 限制。

两个后端均将源指针偏移 `(c / 16) * SrcTileData::Rows * 16 + r * 16` 个元素，但 load2d 参数格式不同：

| 后端 | 小 M 搬运参数 |
| --- | --- |
| A2A3 `pto_load_cbuf_to_ca` | `baseIdx = 0`、`repeat = copyCols / 16`、`srcStride = SrcTileData::Rows / 16`、目标 gap 为 0 |
| A5 `load_cbuf_to_ca` | `mStart = 0`、`kStart = 0`、`mStep = 1`、`kStep = copyCols / 16`、`srcStride = SrcTileData::Rows / 16`、`dstStride = 1`、transpose 为 0 |

相邻源分形起点相距 `srcStride * 512` 字节，目标分形连续排列。
实际读取和写入的数据量均为 `copyCols * 32` 字节；`srcStride > 1` 时，这一数据量可以小于源地址跨度。
repeat/kStep 超过 255 时会分段并相应推进源、目标指针；
这不放宽平台原有的 L1/L0A 容量限制。

**源 padding 与同步。** 当 `c = 0` 时，复用 K 为 16 倍数的 16×K 源，从行 4/8/12 提取四行直至最后一个 K 分形时，
会分别超出紧凑分配的 16×K tile 末尾 128/256/384B。可将物理列数声明为 `K + 16`、有效列数保留为 `K`，
显式拥有额外 512B 存储。对于尾 K，源物理列数取 `ceil16(K) + 16`、目标物理列数取 `ceil16(K)`，
也可得到同样安全的安排。ND→NZ `TLOAD` 仍可只加载一次 16×K GlobalTensor，NZ 行跨度不变。
列起点非零时，源有效列数至少为 `c + K`，物理读取末尾也必须计入 `c`。
其他形状应按读取末尾公式计算：现有容量可能已经足够，且增加 16 列占用 `SrcTileData::Rows * 32` 字节，
只有物理行数为 16 时才是 512B。

整个源分配（包括 padding）必须保持可读，在 MTE1 读取完成前不能被覆盖或复用。
padding 无需初始化为零。目标有效窗口以外的元素未定义，不得作为有效结果使用。
手动模式下，`TLOAD` 后需 MTE2→MTE1 同步，`TMATMUL` 前需 MTE1→M 同步，
覆盖仍被 Cube 使用的 Left tile 前需 M→MTE1 同步。自动模式通过 tile 操作数跟踪这些依赖。
`STPhase` 用于 Acc→Mat 搬出，不能替代这些依赖。分配和同步需覆盖整个物理 tile，
有效形状用于定义结果，不能用它代替物理分配大小。运行时检查使用 `PTO_ASSERT`，由 `_DEBUG` 启用；
所有构建模式下调用方都必须满足约束。

以下片段在 A2A3 或 A5 设备编译上下文（`__CCE_AICORE__`）中使用 `<pto/pto-inst.hpp>` 和
`using namespace pto;`；Left 别名会选择对应架构的布局。假定已分配完整 tile、已加载源的 16×64 有效窗口，且调用方或自动模式负责必要同步：

```cpp
using Mat = Tile<TileType::Mat, half, 16, 80, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512>;
using Left = TileLeft<half, 16, 64, 4, 64>;
AICORE void ExtractFourRows(Left &left, Mat &mat)
{
    TEXTRACT(left, mat, /*indexRow=*/12, /*indexCol=*/0);
}
```

也可以改用动态 M/K 的 Compact 目标，传入 `m = 4, k = 63` 时，物理列数为 128 仍只读取 64 列。
辅助函数显式设置有效形状。对于此源和行起点，合法参数为 `1 <= m <= 4`、`1 <= k <= 64`；
调用方需先分配目标：

```cpp
using CompactLeft = TileLeftCompact<half, 16, 128, DYNAMIC, DYNAMIC>;
AICORE void ExtractCompactRows(CompactLeft &left, Mat &mat, uint32_t m, uint32_t k)
{
    left.SetValidShape(m, k);
    TEXTRACT(left, mat, /*indexRow=*/12, /*indexCol=*/0);
}
```

参见 [A2A3 实现](../../include/pto/npu/a2a3/TExtract.hpp)与
[A5 实现](../../include/pto/npu/a5/TExtract.hpp)、
[A2A3 数值 ST](../../tests/npu/a2a3/src/st/testcase/textract_small_m/textract_small_m_kernel.cpp)与
[A5 数值 ST](../../tests/npu/a5/src/st/testcase/textract_small_m/textract_small_m_kernel.cpp)，以及
[A2A3 指令与边界测试](../../tests/costmodel/st/testcase/textract_small_m/main.cpp)与
[A5 指令与边界测试](../../tests/cpu/st/testcase/textract_small_m_a5/main.cpp)。
A5 主机测试模拟底层搬运指令并执行真实 A5 后端分发入口及 helper；NPU ST 对完整指令序列进行数值验证。

### Vec → Vec 抽取路径

除上述 `Mat/Acc -> ...` 路径外，`TEXTRACT` 还支持 `TileType::Vec -> TileType::Vec` 抽取路径（ND 与 NZ 布局）。A2A3 使用 `CheckTExtractVecToVecCommon`，A5 在 `TEXTRACT_IMPL` 中单独检查：

- `DstTileData::DType` 必须等于 `SrcTileData::DType`。
- A2A3 元素类型：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。
- A5 元素类型：`int8_t`、`int32_t`、`half`、`bfloat16_t`、`float`、`hifloat8_t`、`float8_e4m3_t`、`float8_e5m2_t`、`float8_e8m0_t`、`float4_e2m1x2_t`、`float4_e1m2x2_t`。A5 不支持此路径的 `uint8_t`、`int16_t`、`uint16_t`、`uint32_t` 或 64 位整数。
- A5 ND fp4 路径以打包元素（每个 1 字节包含两个 fp4 值）计数；行步长、静态有效列字节数以及列偏移字节数须 32 字节对齐。
- ND 路径：源/目标行步进须 32 字节对齐；`Dst` 行/列不得超过 `Src`。
- A5 ND Vec→Vec 路径先检查 `indexRow + dst.GetValidRow() <= SrcTileData::Rows` 和 `indexCol + dst.GetValidCol() <= SrcTileData::Cols`；通过检查后，目标有效行数或列数为 0 时直接返回，不读取源或写入目标。对齐与非对齐列偏移均遵循此规则；这不扩展该路径的类型支持，仍不支持 int64。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT src;
  DstT dst;
  TEXTRACT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TEXTRACT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

带 unit flag 的 L0C→L1 搬出（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品、Ascend 950PR/Ascend 950DT）：

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_unit_flag() {
  TileLeft<half, 32, 32> a;
  TileRight<half, 32, 32> b;
  TileAcc<float, 32, 32> c;
  Tile<TileType::Mat, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor> l1;
  TASSIGN(a, 0x0);
  TASSIGN(b, 0x0);
  TASSIGN(c, 0x0);
  TASSIGN(l1, 0x2000);
  // 数据就绪后再搬出：两条指令之间不需要显式 set_flag/wait_flag
  TMATMUL<AccPhase::Final>(c, a, b);
  TEXTRACT<STPhase::Final>(l1, c, /*indexRow=*/0, /*indexCol=*/0);

  // 同一块 L0C 分多次搬出时，非末次用 Partial 不释放 unit flag，末次用 Final 释放
  // TEXTRACT<STPhase::Partial>(l1, c, /*indexRow=*/0, /*indexCol=*/0);
  // TEXTRACT<STPhase::Final>(l1, c, /*indexRow=*/0, /*indexCol=*/0);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### PTO汇编形式

```text
%dst = textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```
