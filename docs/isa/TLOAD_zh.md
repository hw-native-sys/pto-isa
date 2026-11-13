# TLOAD

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

![TLOAD tile operation](../figures/isa/TLOAD.svg)

## 简介

从GlobalTensor (GM) 加载数据到Tile。

## 数学语义

符号表示取决于 `GlobalTensor` 的形状/步长和 `Tile` 的布局。概念上（二维视图，带基础偏移量 `r_0` 和 `c_0`）：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{r_0 + i,\; c_0 + j} $$

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);

template <
    TLoadL2Hint l2Control, typename TileData, typename GlobalData, typename... WaitEvents,
    std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);
```

原有 `TLOAD(dst, src)` 与 `TLOAD<TileT, GTensor>(dst, src)` 仍可用。`TLoadL2Hint` 为首模板的形式为额外重载（`l2Control` 无默认值）。

## L2 cache hint

可选首模板参数重载（原有 `TLOAD(dst, src)` 不变）：

```cpp
TLOAD<TLoadL2Hint::NotAllocKeep>(dst, src);
```

支持的 `TLoadL2Hint`：

| 枚举 | 值 | A2/A3 | A5 |
| --- | --- | --- | --- |
| NormalFirstVictim | 0 | 默认分配（无效果） | 支持 |
| NormalLastVictim | 1 | 默认分配（无效果） | 支持 |
| NormalPersistent | 2 | 默认分配（无效果） | 支持 |
| NotAllocKeep | 4 | 非分配（GM 地址加上运行时 `l2Cacheoffset`） | 支持 |
| NotAllocClean | 5 | 非分配（同 Keep） | 支持 |
| NotAllocDrop | 6 | 非分配（同 Keep） | 支持 |

A2/A3 上实际只有 **两种行为**：

1. **默认分配** — `NormalFirstVictim` (0)、`NormalLastVictim` (1)、`NormalPersistent` (2)：在 A2/A3 上为无效果（对 VLU / first/last/persist 等不同提示无影响）。它们不是有意义的独立模式，均走默认分配路径。
2. **非分配** — `NotAllocKeep` (4)、`NotAllocClean` (5)、`NotAllocDrop` (6)：生效，通过 GM 地址加上运行时 `l2Cacheoffset`（A2/A3 上 Keep/Clean/Drop 行为相同）。

A5 上表内取值均透传给 DMA。CPU / costmodel 接受该模板并忽略。

## 约束

- **实现检查 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**:
    - `TileData::DType` 必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、`half`、`bfloat16_t`、`float`。
    - 目标tile位置必须是 `TileType::Vec` 或 `TileType::Mat`。
    - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`。
    - 运行时：所有 `src.GetShape(dim)` 值和 `dst.GetValidRow()/GetValidCol()` 必须 `> 0`。
    - `TileType::Vec` 加载仅支持匹配的布局：ND->ND、DN->DN、NZ->NZ。
    - `TileType::Mat` 加载支持：ND->ND、DN->DN、NZ->NZ，以及ND->NZ和DN->ZN。
    - 对于ND->NZ：`GlobalData::staticShape[0..1] == 1` 且 `TileData::SFractalSize == 512`。
    - 对于DN->ZN：`GlobalData::staticShape[0..2] == 1` 且 `TileData::SFractalSize == 512`。
    - ND->NZ 也支持 `GlobalData::staticShape[2] != 1`。此时 `Shape2` 是单条指令搬运的 ND 小矩阵个数
      （指令自带的 `ndNum` 操作数），每个小矩阵形状为 `[Shape3, Shape4]`，多个小矩阵沿 tile 行方向堆叠，
      因此 `dst.GetValidRow() == Shape2 * Shape3`。该平台的 `srcNdMatrixStride` 与 `dstNzMatrixStride`
      是 16 位的元素数，因此运行时 `Shape2 > 1` 时还要求 `1 <= Stride2 <= 65535`，
      且 `Shape3 * (32 / sizeof(DType)) <= 65535`。动态 Shape2 在运行时为 1 时不使用这两个矩阵步长，
      不受这两项检查限制；`Shape2 * Shape3 <= TileData::Rows` 和 `1 <= Shape2 <= 65535` 仍须满足。
    - 对于 `int64_t/uint64_t`，仅支持ND->ND或DN->DN。
    - Vec ND->ND 要求 `TileData::Rows < 4096`。普通同布局 UB/L1 加载要求 burst 数对应的维度小于 4096：
      ND 为 `Shape3`，DN 为 `Shape4`，NZ 为 `Shape1`。单行/单列 Mat 路径见下文。
    - Mat ND->NZ 要求 `1 <= Shape3 <= 16384`、`1 <= Shape4 <= 65535`、
      `1 <= Stride3 <= 65535`，且 `TileData::Rows <= 16384`。
    - Mat DN->ZN 要求 `1 <= Shape4 <= 16384`、`1 <= Shape3 <= 65535`、
      `1 <= Stride4 <= 65535`，且 `TileData::Cols <= 16384`。
    - tile 尺寸及放置位置必须满足目标 UB/L1 的容量限制；上述维度限制仅适用于对应路径。
- **实现检查 (Ascend 950PR/Ascend 950DT)**:
    - `sizeof(TileData::DType)` 必须是 `1`、`2`、`4` 或 `8` 字节，且必须匹配 `sizeof(GlobalData::DType)`。
    - 对于 `int64_t/uint64_t`，`TileData::PadVal` 必须是 `PadValue::Null` 或 `PadValue::Zero`。
    - `TileType::Vec` 加载需要以下布局对之一：
    - ND使用行主序 + `SLayout::NoneBox`（ND->ND），
    - DN使用列主序 + `SLayout::NoneBox`（DN->DN），
    - NZ使用 `SLayout::RowMajor`（NZ->NZ）。
    - 对于使用编译时已知形状的行主序ND->ND，`TileData::ValidCol` 必须等于 `GlobalData::staticShape[4]`，且 `TileData::ValidRow` 必须等于 `GlobalData::staticShape[0..3]` 的乘积。
    - A5 Vec NZ->NZ 要求静态内层形状 `Shape3 == 16`、`Shape4 == 32 / sizeof(DType)`，
      打包 FP4 的 `Shape4 == 64`；源形状不要求等于 Tile 的逻辑有效窗口。
      CPU_SIM 的 A5 模式适用相同的内层形状约束。
    - `TileType::Mat` 加载还受到 `TLoadCubeCheck` 的约束（例如，仅特定的ND/DN/NZ转换和L1大小限制）。
    - 对于 `TileType::Mat` 的 ND->NZ 和 DN->NZ：`TileData::SFractalSize == 512`、`sizeof(TileData::DType) != 8`，
      且 `GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1`。
    - ND->NZ 额外支持 `GlobalData::staticShape[2] != 1`（包括动态 Shape2）。`Shape2` 表示源中可用的
      ND 小矩阵个数，每个小矩阵形状为 `[Shape3, Shape4]`，多个小矩阵沿 tile 行方向堆叠。
      运行时：`1 <= Shape2 <= 65535`、`1 <= Shape3 <= 16384`，且
      `1 <= dst.GetValidRow() <= min(TileData::Rows, Shape2 * Shape3)`。数据类型不能是 fp4。
    - DN->NZ 要求 `GlobalData::staticShape[2] == 1`。
    - 对于 `TileType::Mat` 的 DN->ZN，ZN tile 使用 `BLayout::RowMajor` 和 `SLayout::ColMajor`。
      `GlobalData::staticShape[2] != 1`（包括动态 Shape2）时选择多矩阵路径。
      `Shape2` 表示源中可用的 DN 矩阵数量，每个矩阵形状为 `[Shape3, Shape4]`，沿 tile 列方向合并；
      `Stride2` 表示相邻矩阵起始地址之间的距离，单位为元素。
      每个矩阵的行必须连续（`Stride3 == 1`）；`Stride4` 表示相邻列起始地址之间的距离，单位为元素。
    - 多矩阵 DN->ZN 路径要求 `GlobalData::staticShape[0..1] == 1`、`TileData::SFractalSize == 512`、
      `sizeof(TileData::DType)` 为 `1`、`2` 或 `4` 字节（不支持 fp4/hif4），且 `TileData::Cols <= 65535`。
      运行时：`1 <= Shape2 <= 65535`、`1 <= Shape4 <= 16384`、
      `1 <= dst.GetValidCol() <= min(TileData::Cols, Shape2 * Shape4)`，以及
      `1 <= dst.GetValidRow() <= min(TileData::Rows, Shape3)`。
    - A5 `TileType::Mat` 多矩阵 ND->NZ 和 DN->ZN 的源步长保留 64 位精度，静态和动态 `Stride` 均适用。
      `Stride2` 和矩阵内步长（ND 的 `Stride3`、DN 的 `Stride4`）均为 `int64_t` 类型的元素数。
      对支持的 b8/b16/b32 类型，字节间距为 `stride * sizeof(DType)`，使用 64 位计算；
      超过 `INT32_MAX` 的元素步长以及 4 GiB 及以上的字节间距不会被截断为 32 位。
      传入 `Stride` 前，应使用 64 位操作数计算大步长表达式，并确保 GM 存储覆盖所有访问地址。
    - A5 ConvTile 的 `NCHW -> NC1HWC0` 与 `NCDHW -> NDC1HWC0` 加载会读取目的 ConvTile 的
      `strideH` / `strideW` 配置字段（通过 `SetStrideH` / `SetStrideW` 设置）。
      该配置用于反向卷积 dx 场景的输入扩张：源 GlobalTensor 的 H/W 仍描述紧凑输入，
      目的 ConvTile 的 H/W 应描述扩张后的空间，通常为 `(H - 1) * strideH + 1` 和
      `(W - 1) * strideW + 1`。调用方须自行保证源形状、目的形状和 stride 配置一致。
    - 在上述 A5 扩张路径中，`TLOAD` 只写源数据覆盖的位置；stride 扩张产生的空洞不会被写入，
      也不会被自动清零。若后续计算要求空洞为 0，须先初始化目的 ConvTile 所在存储。
      `strideH` / `strideW` 均为 1 时不启用扩张摆放，保持既有加载语义。
    - `TileType::Mat` 加载还处理mx格式的加载，包括 `MX_A_ZZ/MX_A_ND/MX_A_DN` 到ZZ（用于scalarA）和 `MX_B_NN/MX_B_ND/MX_B_DN` 到NN（用于scalarB）。
    - 对于 `MX_A_ZZ/MX_B_NN`：`(GlobalData::staticShape[3] == 16 || GlobalData::staticShape[3] == -1)` 且 `(GlobalData::staticShape[4] == 2 || GlobalData::staticShape[4] == -1)`。
    - 对于 `MX_A_ND/MX_A_DN/MX_B_ND/MX_B_DN`：`(GlobalData::staticShape[0] == 1 || GlobalData::staticShape[0] == -1)` 且 `(GlobalData::staticShape[1] == 1 || GlobalData::staticShape[1] == -1)` 且 `(GlobalData::staticShape[4] == 2 || GlobalData::staticShape[4] == -1)`。
    - 对于scaleA，`dst.GetValidCol() % 2 == 0`。
    - 对于scaleB，`dst.GetValidRow() % 2 == 0`。

- **有效区域**:
    - 传输范围由所选布局路径决定，不一定等于逻辑有效矩形。
    - 源形状与 `dst.GetValidRow()` / `dst.GetValidCol()` 必须满足所选路径的对应关系，不能将有效行列数
      视为独立的裁剪边界。例如，A2/A3 的 ND->NZ 和 DN->ZN 按源矩阵形状搬运；
      A5 单矩阵 ND->NZ 搬运 `Shape3` 行、`dst.GetValidCol()` 列。
    - A5 Vec NZ->NZ（CPU A5 模式也模拟此行为）搬运 `Shape0` 组、每组 `Shape1` 个列块。
      每个 burst 复制 `dst.GetValidRow() * 32` 字节；源组步长、列块步长分别为以元素计数的
      `Stride0`、`Stride1`。目标列块步长为 `TileData::Rows * 32` 字节，组步长为
      `Shape1 * TileData::Rows * Shape4` 个元素；FP4 的元素步长除以 2 后换算为字节。
      此路径不按 `validCol` 裁剪列块，也不增加 `PadVal` 填充。有效行之外及未搬运列块保留原值，
      GM 和 Tile 的物理存储必须覆盖全部搬运范围。
      CPU_SIM 需在绑定 Tile 前选择 A5，见
      [选择模拟目标架构](../coding/cpu_sim_zh.md)。
    - 在A2/A3上，同布局且按块对齐的 `TileType::Mat` 加载仅写入有效区域。ND到NZ、DN到ZN加载（以及单行/单列Mat特殊路径）还会将最后一个不完整C0块的尾部填零。其他数据保持不变，包括共享同一底层存储的其他tile视图所对应的数据。
    - 在A5上，同布局 `TileType::Mat` 的ND/DN加载仅按 `PadVal` 填充最后一个不完整32B块；ND/DN到分形布局的加载将最后一个不完整C0块的尾部填零。完整32B间隔块以及未参与传输的行或列保持不变。
    - 在A5上，`GlobalData::staticShape[2] != 1` 的 ND->NZ 加载合并后的前 `dst.GetValidRow()` 行。
      先用一条指令搬运 `dst.GetValidRow() / Shape3` 个完整矩阵，若 `dst.GetValidRow() % Shape3` 非零，
      再单独搬运这些尾行；没有完整矩阵时只搬运尾块。两次搬运均保持整个 tile 的 NZ 列块步长。
      Shape2 为动态维度且运行时取值为 1 时也适用。例如 `Shape3 = 3`、`dst.GetValidRow() = 17` 时，
      搬运 5 个完整矩阵和第 6 个矩阵的前 2 行，此时要求 `Shape2 >= 6`。
    - 在 A5 上，`GlobalData::staticShape[2] != 1` 的 DN->ZN 加载合并后的前 `dst.GetValidCol()` 列。
      单条指令先搬运 `dst.GetValidCol() / Shape4` 个完整矩阵；若 `dst.GetValidCol() % Shape4` 非零，
      再单独搬运下一个矩阵的这些尾列。没有完整矩阵时只搬运尾块。
      Shape2 为动态维度且运行时取值为 1 时也适用。
      存在尾列时，令 `n = dst.GetValidCol() / Shape4`，尾矩阵从 `src.data() + n * Stride2` 开始
      （偏移单位为元素）；即使对应的字节距离达到或超过 4 GiB，源偏移也保留 64 位精度。
      两次搬运均保持 `TileData::Cols` 对应的 C0 块步长，单位为 32 字节。
      仅对有效行末尾不足一个 C0 块的部分填零；未参与搬运的列以及有效行之外的完整 C0 块保留原值。
      编译期 Shape2 为 1 时使用单矩阵路径：搬运 `Shape4` 列、`dst.GetValidRow()` 行，
      因此源形状必须描述要搬运的列范围。
    - 在A2/A3和A5上，`PadVal` 非空时，`TileType::Vec` 的ND/DN加载仅填充每个burst传输后不足32B的尾部；完整32B间隔块以及未参与传输的行或列保持不变。NZ加载和`PadValue::Null`不增加填充。

### 步长与长度边界

A2/A3、A5、A6、KirinX90、Kirin9030 和 Kirin9050 的相关 TLOAD 路径保留 `int64_t` 源步长。
对于 b8/b16/b32/b64 数据，元素数到字节数的转换使用 64 位计算。构造大 `Stride` 表达式时须使用
64 位操作数，GM 存储须覆盖所有访问地址，目标存储须满足实际物理容量限制。字节间距及计算出的偏移
必须在所用类型的可表示范围内；各路径还须满足下文列出的约束。

**gap** 是前一段 burst 结束到下一段开始之间的间隔；**stride** 是相邻两段 burst 起始地址之间的距离。
对于 b8/b16/b32/b64 同布局路径，令 `E = sizeof(DType)`，ND 的源字节 stride 为 `Stride3 * E`，
DN 为 `Stride4 * E`，NZ 为 `Stride1 * E`。memory 后端对应的字节 gap 分别为
`(Stride3 - Shape4) * E`、`(Stride4 - Shape3) * E` 和 `(Stride1 - Shape2 * Shape3 * Shape4) * E`。
这些使用 gap 的路径要求 gap 非负。下列上限分别对应不同的指令字段：

| 路径 | 字段与回退行为 |
| --- | --- |
| A2/A3、KirinX90 普通 GM 到 UB | 源 gap 单位为字节。超过 `UINT32_MAX` 时，使用 64 位源偏移逐 burst 搬运。 |
| A2/A3 普通 GM 到 L1 | 源 gap 单位为 32 字节块。超过 `UINT16_MAX` 个块时逐 burst 搬运；同时须满足块对齐要求。 |
| KirinX90 普通 GM 到 L1 | 按所选块搬运或字节对齐指令的单位检查长度和 gap，超出范围时先拆分 burst 和长度，再编码。 |
| A5/A6 普通 GM 到 UB/L1 | 源 stride 单位为字节。超过 `2^40 - 1` 时逐 burst 搬运；ND/DN 的 Shape1/Shape2 循环字节步长超限时也改用显式源地址。 |
| A5 NC1HWC0 和五维 FRACTAL_Z（`[C1, H, W, N, C0]`） | 二者均使用 `TLoad5HD`：由 `Stride0`/`Stride1` 换算出的源循环字节步长超过 `2^40 - 1` 时使用软件循环；由 `Stride2` 换算出的 burst 字节步长使用共享逐 burst 回退。 |
| Kirin9030、Kirin9050 普通 DMA | 源步长和字节数转换保留 64 位；通过共享 register DMA 封装的调用使用相同的 burst 步长回退。 |
| KirinX90 b32 ND->NZ/DN->ZN | 转换为 b16 单位后再检查字段，转换值超过 16 位时拆分搬运；公开接口的矩阵内步长上限为 65535 个元素。 |

上述上限均包含边界值：`UINT32_MAX` 字节、`UINT16_MAX` 个块及 `2^40 - 1` 字节仍可由对应字段编码；
`2^40` 字节已触发 register helper 的回退。A5 四维 FRACTAL_Z（`[C1HW, N/16, 16, C0]`）也使用
共享逐 burst 回退，其源字节 stride 由 `Stride1` 换算而来。A2/A3 的 GM 到 L1 gap 回退同样适用于使用该 helper 的
同布局卷积加载（NC1HWC0、两种 FRACTAL_Z 表示及 NDC1HWC0），但不会拆分超长 burst，也不会放宽
目标 gap 或 burst 数量限制。普通 A2/A3 块搬运的长度、源 gap 和目标 gap 仍须为 32 字节的整数倍，
长度和目标 gap 的块数仍须能用 16 位表示。

KirinX90 的 GM 到 L1 仅在目标 gap 为零、长度与源 gap 均为 32 字节的整数倍，且二者的块数均能用
16 位表示时，直接使用块搬运指令。否则，只有长度和源 gap 均不超过 65535 字节、目标 gap 小于
32 字节时，才直接使用字节对齐搬运指令。回退路径先独立计算每个 burst 的源和目标地址，再在剩余
长度超过 65535 字节时按 65504 字节拆分；最后不超过 65535 字节的部分作为末段搬运。
因此完整的目标 gap 通过地址计算保留，不会被收窄到 padding 字段。

对于 KirinX90 b32 ND->NZ/DN->ZN，连续维度长度与矩阵内步长需乘 2 后传给 b16 指令；
32768 个 b32 元素已超出转换后的 16 位字段。回退按行搬运，并按 C0 对齐边界拆分连续维度，
每段最多 65520 个 b16 元素。布局和多矩阵形式仍须满足对应接口约束。

在 A2/A3 和 KirinX90 上，`TileData::Rows == 1` 的 ND 或 `TileData::Cols == 1` 的 DN 同布局 Mat 加载
先搬运完整的 32 字节块，再搬运最后不足一块的有效数据并将块内尾部填零。元素数不会收窄为 16 位，
b64 长度也不经过 b16/b32 单位转换。因此，65536 个元素以及 b64 的 16384/32768 个元素
不构成额外的长度上限；源形状、有效行列数和 L1 容量约束仍然有效。这些路径要求
`GlobalData::staticShape[0..2] == 1`，且向量内元素连续、有效行列数与源形状一致：
ND 为 `Shape3 == validRow == 1`、`Shape4 == validCol`；DN 为 `Shape4 == validCol == 1`、
`Shape3 == validRow`。元素数量取源 `Shape4`（ND）或 `Shape3`（DN）。每次完整块搬运最多包含
65535 个块；最后的 1–31 字节通过 b8 转换搬运，并将块内剩余部分填零，此行为与 `PadVal` 无关。
向量长度已按块对齐时不需要尾部搬运。补齐块之后的存储保持不变。

上述回退不会扩展独立格式转换指令的形状和步长约束。特别是，普通 DMA 的 40 位处理不能用来推断
A5/A6 ND 到分形布局指令的支持范围；仅凭 C++ 参数为 64 位也不能确定硬件范围。
burst 长度/数量、目标 stride，以及硬件循环次数/目标循环步长字段，
也须满足对应后端的范围约束。

### 运行时路径选择与标量开销

A5/A6 及 Kirin9030/Kirin9050 调用的共享 register DMA helper 提供直接原生路径：源字节 stride
不超过 `2^40 - 1` 时，执行多 burst DMA 后直接返回，不进入逐 burst 循环。只有 stride 超限时
才进入循环；每次循环搬运一个 burst，将指令内的源和目标 stride 置零，通过完整位宽的软件偏移
计算地址。A5 上，当编译器能够确定 burst 数为常量且不超过 `8` 时，对回退循环进行有限展开以降低循环
开销。更大的 burst 数、运行时才能确定的 burst 数，以及使用该 helper 的其他后端，仍禁止展开以限制
代码体积。这一编译期选择不会增加运行时 burst 数检查。L2 hint 的传递和 padding 参数保持不变，
包括 Mat 的 b64 到 b32 padding 数量换算。

逐 burst 回退的软件地址计算和 DMA 发射次数随 burst 数量增长。大量短 burst 即使没有在每次加载后
进行标量等待，也可能产生明显开销。原生路径中范围判断的测量结果不能代表回退性能；应按实际
burst 数量、搬运大小和同步方式分别评估两条路径。

A5/A6 的 ND/DN 外层循环步长检查仍然存在。A5 `TLoad5HD` 在任一路径结束后均将硬件循环次数恢复为 1；
A5/A6 的 MX-A 向量路径将循环次数设为 1，不设置未使用的外层循环步长。
helper 走原生路径时，这些设置及格式转换操作也同样适用。

已知的静态 `Shape`/`Stride` 可以让编译器消除范围判断；动态值仍可能保留比较和分支，因此不能保证
标量开销为零。静态 stride 超限时，即使范围比较被消除，仍然需要逐 burst 搬运。
公开 `TLOAD` 模板没有关闭步长回退的开关；`TLoadL2Hint` 仅选择缓存提示。
ptoas 生成的 C++ 调用这些模板时遵循相同行为。

只要定义了 `_DEBUG`（包括 `_DEBUG=0`），就会启用 `PTO_ASSERT`；未定义时移除。
`NDEBUG` 不控制它。调试断言与原生/回退路径选择相互独立，未定义 `_DEBUG` 时回退判断仍然生效，
编译期断言也仍然有效。发布构建的调用者同样必须满足接口约束。

各后端注册用例见[步长 ST 覆盖说明](../../tests/npu/a2a3/src/st/testcase/tload_large_stride/README.md)。

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

### A5 多矩阵 DN 到 ZN 加载（手动模式）

下面的 Cube 函数加载合并后的前 17 列，即 5 个完整的 3 列矩阵和第 6 个矩阵的前 2 列。
GM 物理存储为 `[8, 9, 64]`（矩阵、列、行），DN 视图选择每个矩阵的 3 列、35 行。
以下步长均以元素为单位。

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

AICORE void example_dn_to_zn(__gm__ int16_t* in) {
  using SrcShape = Shape<1, 1, 8, 35, 3>;
  using SrcStride = pto::Stride<4608, 4608, 576, 1, 64>;
  using SrcGlobal = GlobalTensor<int16_t, SrcShape, SrcStride, Layout::DN>;
  using MatTile = Tile<TileType::Mat, int16_t, 64, 32, BLayout::RowMajor,
                       35, 17, SLayout::ColMajor, 512>;

  SrcGlobal src(in);
  MatTile dst;
  TASSIGN(dst, 0x1000);
  TLOAD(dst, src);
}
```

对于 `0 <= r < 35`、`0 <= c < 17`，结果为
`dst[r, c] = in[(c / 3) * 576 + (c % 3) * 64 + r]`。

### A5 动态大步长 DN 到 ZN 加载（手动模式）

下面的 Cube 函数加载合并后的前 3 列，即第一个矩阵的全部 2 列和第二个矩阵的第 1 列。
`matrixStride` 在运行时传入，表示相邻矩阵起始地址之间的距离，单位为 `uint16_t` 元素。
例如，`(int64_t{1} << 31) + 128` 个元素对应 `4294967552` 字节（4 GiB + 256 字节）。
GM 分配须覆盖所有访问元素，包括第二个矩阵起始处的 32 行。
使用上述示例步长时，从 `in` 开始所需的存储跨度为 `(matrixStride + 32) * sizeof(uint16_t)` 字节。
`SrcStride` 中的 `-1` 仅将 `Stride2` 设为动态值；`Shape2` 仍是编译期确定的矩阵数量 2。

```cpp
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

AICORE void example_dn_to_zn_large_stride(__gm__ uint16_t* in, int64_t matrixStride) {
  using SrcShape = Shape<1, 1, 2, 32, 2>;
  using SrcStride = pto::Stride<1, 1, -1, 1, 64>;
  using SrcGlobal = GlobalTensor<uint16_t, SrcShape, SrcStride, Layout::DN>;
  using MatTile = Tile<TileType::Mat, uint16_t, 64, 64, BLayout::RowMajor,
                       32, 3, SLayout::ColMajor, 512>;

  SrcGlobal src(in, SrcShape{}, SrcStride(1, 1, matrixStride, 1, 64));
  MatTile dst;
  TASSIGN(dst, 0x1000);
  TLOAD(dst, src);
}
```

对于 `0 <= r < 32`、`0 <= c < 3`，结果为
`dst[r, c] = in[(c / 2) * matrixStride + (c % 2) * 64 + r]`。
因此最后加载的一列来自 `in[matrixStride + r]`。
