# SYNCALL

## 指令示意图

> 仓库当前未提供 `SYNCALL.svg`（与多数向量算子不同）。`SYNCALL` 为**跨核控制面**原语，不描述单Tile上的逐元素数据变换；语义上可理解为「所有选定参与者在同一点汇合后再前进」。

以下示意区分硬件（FFTS）与软件（GM轮询）两条路径（概念图，非规范绑定）：

```mermaid
flowchart TB
  subgraph hard [硬件模式 Hard / FFTS]
    H1[各参与者到达调用点] --> H2[ffts_cross_core_sync 等]
    H2 --> H3[wait_flag_dev 等]
    H3 --> H4[屏障完成]
  end
  subgraph soft [软件模式 Soft / GM]
    S1[写本地 GM slot 计数] --> S2[轮询全部 slot 达阈值]
    S2 --> S3[屏障完成]
  end
```

## 简介

`SYNCALL` 是跨核同步屏障，支持Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品和Ascend 950PR/Ascend 950DT NPU后端。通过模板参数 `SyncCoreType` 选择核类型模式：

- **AIV-only**（默认）：`SYNCALL()` 同步所有AIV核。
- **AIC-only**：`SYNCALL<SyncCoreType::AICOnly>()` 同步所有AIC核（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品支持硬件和软件模式；Ascend 950PR/Ascend 950DT仅支持硬件模式）。
- **MIX（AIC+AIV）**：`SYNCALL<SyncCoreType::Mix>()` 同步AIC和AIV混合核。

通过 `SyncAllMode`（在带workspace的重载中显式给出）选择 **硬件模式（FFTS）** 或 **软件模式（GM轮询）**。无workspace的重载对应硬件路径。

## 数学语义

不适用逐元素算术语义。`SYNCALL` 表达的是 **barrier（屏障）到达** 关系：

- 在某一动态程序点上，凡属于当前 `SyncCoreType` 所划定参与者集合的core，均须执行到该 `SYNCALL` 调用之后，任一参与者方可越过该点继续执行后续代码。
- 硬件模式：由FFTS旗标与设备侧 `wait_flag_dev` 等原语保证跨核可见顺序。
- 软件模式：由GM中各参与者独占slot的单调计数与 `dcci`/`dsb` 等一致性原语，在轮询中判定「全员已到达当前代数」。

该语义**不**对barrier之后的GM或其它buffer内容作额外保证；跨核数据可见性需调用方自行维护，详见「跨核GM通信注意事项」。

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`。软件模式接口使用类型安全的 `GlobalTensor` 和 `Tile` 参数（通过SFINAE约束）：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
// 硬件模式（所有 CoreType 通用）
template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
PTO_INST void SYNCALL();

// 软件模式 — AIV-only（GlobalTensor + Vec Tile）
template <SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::AIVOnly,
          typename GlobalData, typename TileData,
          std::enable_if_t<is_global_data_v<GlobalData> &&
                           is_tile_data_v<TileData> && TileData::Loc == TileType::Vec, int> = 0>
PTO_INST void SYNCALL(GlobalData &gmWorkspace, TileData &ubWorkspace, int32_t usedCores = 0);

// 软件模式 — AIC-only（GlobalTensor + Mat Tile）
template <SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::AICOnly,
          typename GlobalData, typename TileData,
          std::enable_if_t<is_global_data_v<GlobalData> &&
                           is_tile_data_v<TileData> && TileData::Loc == TileType::Mat, int> = 0>
PTO_INST void SYNCALL(GlobalData &gmWorkspace, TileData &l1Workspace, int32_t usedCores = 0);

// 软件模式 — MIX（GlobalTensor + Vec Tile + Mat Tile）
template <SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::Mix,
          typename GlobalData, typename UbTileData, typename L1TileData,
          std::enable_if_t<is_global_data_v<GlobalData> &&
                           is_tile_data_v<UbTileData> && UbTileData::Loc == TileType::Vec &&
                           is_tile_data_v<L1TileData> && L1TileData::Loc == TileType::Mat, int> = 0>
PTO_INST void SYNCALL(GlobalData &gmWorkspace, UbTileData &ubWorkspace, L1TileData &l1Workspace,
                       int32_t usedCores = 0);
```

## 参数

- `gmWorkspace`: `GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>>`（在Ascend C与 `using namespace pto` 并存时，建议写全 `pto::`，避免与编译器内置头中的 `Stride` 枚举同名冲突）。软件模式使用的GM workspace，调用前需要初始化为0。每个参与core占用8个 `int32_t`（按cache line隔离同步计数）。
- `ubWorkspace`: `Tile<TileType::Vec, int32_t, 1, SYNCALL_SOFT_SLOT_INT32>`（模板参数固定为 `SYNCALL_SOFT_SLOT_INT32 = 8`，即每核一个cache line槽位）。AIV-only和MIX软件模式使用的UB scratch，运行时后备内存容量须至少为 `usedCores * 8 * sizeof(int32_t)`（实现通过裸指针访问，不校验模板容量；示例中以编译期最大参与核数 × `SYNCALL_SOFT_SLOT_INT32` 声明以保证后备内存充足）。
- `l1Workspace`: `Tile<TileType::Mat, int32_t, 1, SYNCALL_SOFT_SLOT_INT32>`。AIC-only和MIX软件模式使用的L1（cbuf）scratch，用于 `create_cbuf_matrix` 填充同步值后经DMA搬移到GM。
- `usedCores`: 参与软件barrier的core数。为0时自动推算——AIV-only / AIC-only使用 `get_block_num()`，MIX使用 `SYNCALL_GET_MIX_PARTICIPANT_COUNT()`（即 `AIC blocks × (1 + AIV ratio)`）。

## Kernel Meta宏

下列场景需在ELF中**手写** `.ascend.meta`，供runtime正确调度：**Hard AIV-only**、**Soft AIC-only**、以及 **register-ELF的MIX**（如1:1 hard）。`dav-c220` 自动拆分场景由Bisheng生成meta，见本节末尾。宏定义于 `include/pto/common/kernel_meta.hpp`：

> `kernelName` 须与 `__global__` 入口符号**完全一致**（写入section `.ascend.meta.<kernelName>`）。

```cpp
// AIV 侧 kernel（标记为 MIX_AIV_MAIN，ratio 固定 0:1）
PTO_SYNCALL_AIV_KERNEL_META(kernelName);

// AIC 侧 kernel（标记为 MIX_AIC_MAIN，指定 AIC:AIV 比例）
PTO_SYNCALL_MIX_AIC_KERNEL_META(kernelName, aicRatio, aivRatio);
```

**使用示例**

Hard AIV-only（单kernel，chevron启动）：

```cpp
PTO_SYNCALL_MIX_AIC_KERNEL_META(MyKernel_mix_aic, 1, 2);  // AIC kernel ELF
PTO_SYNCALL_AIV_KERNEL_META(MyKernel_mix_aiv);             // AIV kernel ELF
```

Soft AIC-only（单kernel，chevron启动）：

```cpp
PTO_SYNCALL_AIC_KERNEL_META(MyKernel);
extern "C" __global__ AICORE void MyKernel(...) { SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(...); }
```

register-ELF通用配对（AIC侧指定比例 + AIV侧）。注意：当前 `syncall` ST的MIX 1:2已改用 `dav-c220` 自动拆分、无需手写meta；下例仅演示register-ELF路径的宏配对写法：

```cpp
PTO_SYNCALL_MIX_AIC_KERNEL_META(MyKernel_mix_aic, 1, 2);
PTO_SYNCALL_AIV_KERNEL_META(MyKernel_mix_aiv);
```

register-ELF MIX 1:1 hard（**AIC与AIV两侧均用** `PTO_SYNCALL_MIX_AIC_KERNEL_META(..., 1, 1)`，AIV侧**不要**用 `PTO_SYNCALL_AIV_KERNEL_META`）：

```cpp
PTO_SYNCALL_MIX_AIC_KERNEL_META(MyKernel_mix_aic, 1, 1);
PTO_SYNCALL_MIX_AIC_KERNEL_META(MyKernel_mix_aiv, 1, 1);
```

**无需手写meta的常见场景**（完整对照见下文「编译与调度指南」场景速查表）：

- AIV-only Soft（`dav-c220-vec`）
- MIX 1:2 Hard / Soft、Hard AIC-only（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品，`dav-c220` 自动拆分）
- MIX 1:1 Soft（双流chevron）

> **dav-c220自动拆分**：使用 `--cce-aicore-arch=dav-c220` 编译时，Bisheng会自动生成AIC/AIV子kernel及对应 `.ascend.meta`，物理比例为 **1:2**（每个AIC block配2个AIV subblock）。此时**无需**手写 `PTO_SYNCALL_MIX_AIC_KERNEL_META`，也**不能**通过meta把比例改成1:1（见下文「MIX 1:1」）。

## 编译与调度指南（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）

本节以ST用例 [`tests/npu/a2a3/src/st/testcase/syncall/`](../../tests/npu/a2a3/src/st/testcase/syncall/) 为准，说明不同 `SyncCoreType` / 模式 / AIC:AIV比例下应采用的**编译arch**、**Meta** 与 **Host启动**方式。Host侧通过 [`syncall_core_config.hpp`](../../tests/npu/a2a3/src/st/testcase/syncall/syncall_core_config.hpp) 在运行时决定launch grid（910B1：24 AIC + 48 AIV；910B4：20 AIC + 40 AIV），同一套kernel二进制可跨芯片复用。

### 场景速查表

| 场景 | 同步模式 | 参与者数 | 编译 `--cce-aicore-arch` | Kernel Meta | Host启动 | 参考源文件 |
|------|---------|----------|--------------------------|-------------|-----------|-----------|
| AIV-only | Hard | `aiv` | `dav-c220-vec` | `PTO_SYNCALL_AIV_KERNEL_META` | chevron `<<<aiv>>>` | `syncall_kernel.cpp` |
| AIV-only | Soft | `aiv` | `dav-c220-vec` | 无 | chevron `<<<aiv>>>` | `syncall_soft_kernel.cpp` |
| AIC-only | Hard | `aic` | **`dav-c220`**（MIX自动拆分，AIV空stub） | 由Bisheng自动生成 | chevron `<<<aic>>>` | `syncall_aic_hard_kernel.cpp` |
| AIC-only | Soft | `aic` | `dav-c220-cube` | `PTO_SYNCALL_AIC_KERNEL_META` | chevron `<<<aic>>>` | `syncall_aic_kernel.cpp` |
| MIX 1:2 | Hard / Soft | `aic×3` | **`dav-c220`** | 由Bisheng自动生成 | chevron `<<<aic>>>`（hard/soft同一 `.so`） | `syncall_mix_1_2_kernel.cpp` |
| MIX 1:1 | Soft | `aic×2` | cube + vec各编一份 `.o` | 无 | **双流** chevron：AIC `<<<aic>>>` + AIV `<<<aiv>>>` | `syncall_mix_1_1_soft_kernel.cpp` |
| MIX 1:1 | Hard | `aic×2` | cube + vec各编一份 `.o` | **`PTO_SYNCALL_MIX_AIC_KERNEL_META(..., 1, 1)`** | **register ELF** + `rtKernelLaunchWithHandleV2` | `syncall_mix_1_1_kernel.cpp` |

Hard与Soft kernel **不可共用同一 `.so`**（AIV-only / AIC-only等场景下soft会污染hard的FFTS配置导致hang）；MIX 1:2的hard与soft因均走dav-c220自动拆分，可放在同一源文件的同一 `.so` 中。

### 各路径说明

#### 1. Chevron单arch编译（AIV-only / AIC-only soft）

- 编译：单个源文件 + 对应arch（`dav-c220-vec` 或 `dav-c220-cube`），产出独立 `.so`。
- 启动：`kernel<<<blockDim, nullptr, stream>>>(..., totalBlocks)`，`blockDim` 与 `totalBlocks` 由Host在运行时传入（ST中来自 `syncall_cfg::GetCoreConfig()`）。
- Hard AIV-only须在kernel上声明 `PTO_SYNCALL_AIV_KERNEL_META`。

#### 2. Chevron MIX自动拆分（MIX 1:2、Hard AIC-only）

- 编译：`--cce-aicore-arch=dav-c220`；CMake使用 `pto_syncall_chevron_kernel(<target> <source>)`。
- 启动：单次chevron `<<<aic>>>`；runtime按物理1:2拉起全部MIX参与者。
- Kernel参数：`aicBlocks` 与 `totalParticipants` 作为标量从Host传入（AIC/AIV两侧读同一参数），以支持910B1/910B4等不同cube数。
- **Hard AIC-only特例**：纯 `dav-c220-cube` 无法建立AIC-only硬同步所需的FFTS上下文。须用 `dav-c220` MIX编译：AIC执行 `SYNCALL<AICOnly>()`，AIV为空stub；`totalBlocks` 由Host传入。

#### 3. 双arch双stream（MIX 1:1 Soft）

- 原因：ccec/bisheng路径下 `GetTaskRatio()` 恒为 **2**，`dav-c220` 自动拆分物理固定 **1:2**，无法得到真1:1。
- 编译：同一源文件分别以 `dav-c220-cube`（`-DSYNCALL_MIX_BUILD_AIC`）和 `dav-c220-vec`（`-DSYNCALL_MIX_BUILD_AIV`）各编一份 `.o`，链接为一个 `.so`；CMake使用 `pto_syncall_mix11_soft_kernel`。
- 启动：AIC与AIV分别在两个 `aclrtStream` 上chevron `<<<aic>>>` 与 `<<<aiv>>>`；`aicBlocks` / `totalParticipants` 由Host运行时传入。

#### 4. Register ELF（MIX 1:1 Hard）

- 原因：Hard MIX同步需要单一MIX FFTS上下文；chevron自动拆分在ccec下做不到真1:1。
- 编译：cube / vec各编带 `PTO_SYNCALL_MIX_AIC_KERNEL_META(name, 1, 1)` 的 `.o`，再以 `-DSYNCALL_MIX_REGISTER_BUILD` 生成register专用 `.o`，经 `make_mix_register_elf.py` 合成registration ELF；CMake使用 `pto_syncall_mix_kernel`。
- 启动：`rtRegisterAllKernel` + `rtKernelLaunchWithHandleV2(handle, tilingKey, aicBlocks, ...)`；device侧用 `get_block_num()` 推导参与者数（register路径仅传 `ffts/out/flags` 三个参数）。

## 模式支持矩阵

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品

| 核类型 | 硬件模式 | 软件模式 |
|--------|---------|---------|
| AIV-only | 支持 | 支持 |
| AIC-only | 支持 | 支持 |
| MIX | 支持 | 支持 |

### Ascend 950PR/Ascend 950DT

| 核类型 | 硬件模式 | 软件模式 |
|--------|---------|---------|
| AIV-only | 支持 | 支持 |
| AIC-only | 支持 | 不支持 |
| MIX | 不支持 | 支持 |

## 约束

- 软件模式各平台GM写入路径：
  - Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品（AIC-only与MIX的AIC侧）：AIC通过 `copy_cbuf_to_gm`（L1→GM DMA）写GM slot；MIX的AIV侧通过UB workspace写入。
  - Ascend 950PR/Ascend 950DT MIX：Ascend 950PR/Ascend 950DT AIC（`dav-c310-cube`）不支持 `copy_cbuf_to_gm`，改为通过 `intra_block` 信号委托同block的AIV subblock 0代写UB→GM。
- Ascend 950PR/Ascend 950DT平台限制原因（对应「模式支持矩阵」）：
  - AIC-only软件不可用：Ascend 950PR/Ascend 950DT AIC缺少 `copy_cbuf_to_gm` 等独立写GM的DMA路径，无法实现GM轮询。
  - 硬件MIX不可用：`rtGetC2cCtrlAddr` 在Ascend 950PR/Ascend 950DT（`CHIP_DAVID`）返回 `RT_ERROR_FEATURE_NOT_SUPPORT`（207000），取不到FFTS基地址。
  - AIC-only硬件：通过 `ffts_cross_core_sync` + `wait_flag_dev` 实现，不需要 `set_ffts_base_addr`。
- 软件模式要求所有参与core以相同顺序进入同一组barrier（基于单调代数计数，进入次数/顺序不一致会导致错配或死锁）。
- `SYNCALL` 不参与PTO的Event自动依赖编排：既不接受 `WaitEvents`，也不返回可被后续指令等待的 `RecordEvent`。因此它不会自动等待前序数据指令（如 `TSTORE`）完成，`SYNCALL` 前后与数据指令之间的顺序与可见性需调用方自行保证（见「跨核GM通信注意事项」）。
- 在auto构建路径（`__PTO_AUTO__`）下，`SYNCALL` 为no-op，不发射跨核硬件同步（与 `TSYNC` 等一致）；真实同步只在manual kernel中发生。

## 跨核GM通信注意事项

`SYNCALL` 只提供barrier **到达**语义（hard / soft皆然），**不**保证barrier前后业务数据的跨核cache可见性。当算子在barrier前各核写GM、barrier后各核读他核GM（如跨核histogram / 前缀和）时，调用方需自行满足以下两点，否则会读到脏数据或发生丢写。

### 1. cache一致性：必须显式 `dcci` / `dsb`

- **写方**：`copy_ubuf_to_gm` / `copy_cbuf_to_gm` 之后接 `dcci(addr, SINGLE_CACHE_LINE)` + `dsb(DSB_DDR)`，把数据刷出到DDR。
- **读方**：读前 `dcci(addr, SINGLE_CACHE_LINE)`（invalidate）+ `dsb`，确保读到DDR最新值而非本核旧cache。
- 仅有 `set_flag` / `wait_flag`（核内流水同步）**不足以**保证跨核可见性。
- 该要求与barrier模式无关：**硬件FFTS barrier同样不刷cache**，只保证「全员到达」的控制面顺序。
- `SYNCALL` 内部对自己的同步槽位已做完整 `dcci` + `dsb(DDR)` 处理，但**不会**替调用方刷业务数据。

### 2. 每核slot按cache line独占：避免false sharing丢写

- `dcci` / DMA以 **32Byte cache line** 为粒度操作；若相邻核slot共享同一条cache line，跨核刷新会互相覆盖 / 丢写。
- 每核slot应按32Byte对齐并**独占一条cache line**（`int32` 场景即stride = 8，而非4）。
- `SYNCALL` 自身的同步槽位即按此设计：`SYNCALL_SOFT_SLOT_INT32 = 8`（见 `include/pto/common/type.hpp`），调用方的业务workspace也应遵循同样的隔离原则。

## 示例

### 手动（Manual）—硬件模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// AIV-only：全 AIV 核 FFTS 屏障（需正确 kernel meta / ELF）
void example_hard_aiv() {
  SYNCALL();
}

// AIC-only：仅编译到 AIC（__DAV_CUBE__）单元时可用；A5 上已验证硬模式路径
void example_hard_aic() {
  SYNCALL<SyncCoreType::AICOnly>();
}

// MIX：AIC 与 AIV 配对 ELF，见上文「Kernel Meta 宏」一节
void example_hard_mix() {
  SYNCALL<SyncCoreType::Mix>();
}
```

### 手动（Manual）—软件模式

软件模式需传入 **已清零** 的GM workspace与合法容量的UB/L1 Tile。`Mode` 须为 `SyncAllMode::Soft`（`Hard` 时忽略workspace，行为同无参 `SYNCALL_IMPL`）。

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_soft_aiv(__gm__ int32_t *gmPtr) {
  GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(gmPtr);
  Tile<TileType::Vec, int32_t, 1, SYNCALL_SOFT_SLOT_INT32> ub;
  SYNCALL<SyncAllMode::Soft, SyncCoreType::AIVOnly>(gmWs, ub, 0);
}
```

MIX软件模式需同时提供UB与L1（Mat）Tile；Ascend 950PR/Ascend 950DT AIC侧通过代理路径写GM，详见「约束」一节。
