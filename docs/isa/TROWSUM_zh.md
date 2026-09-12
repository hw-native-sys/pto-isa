# TROWSUM

## 指令示意图

![TROWSUM tile operation](../figures/isa/TROWSUM.svg)

## 简介

通过对列求和来归约每一行。

## 数学语义

设 `R = src.GetValidRow()`，`C = src.GetValidCol()`。对 `0 <= i < R`：

$$ \mathrm{dst}_{i,0} = \sum_{j=0}^{C-1} \mathrm{src}_{i,j} $$

## 汇编语法

同步形式：

```text
%dst = trowsum %src : !pto.tile<...> -> !pto.tile<...>
```

降低时可能引入内部临时Tile；C++内建接口需要显式传入 `tmp` 操作数。

### AS Level 1（SSA）

```text
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束

### 通用约束或检查

以下约束描述 NPU 后端；CPU_SIM 的检查及兼容性例外见下文。

- `dst` 和 `src` 必须均为 `TileType::Vec`。
- `src` 必须使用标准ND布局：行主且非分形（`BLayout::RowMajor`、`SLayout::NoneBox`）。
- `dst` 必须是非分形布局（`!isBoxedLayout`），且使用以下两种布局之一：
    - ND布局（`BLayout::RowMajor`、`SLayout::NoneBox`），或
    - 列数严格为1的DN布局（`BLayout::ColMajor`、`SLayout::NoneBox`、`Cols == 1`）。
- `dst` 和 `src` 的元素类型必须一致。
- 运行时有效区域检查：
    - `src.GetValidRow() != 0`
    - `src.GetValidCol() != 0`
    - `src.GetValidRow() == dst.GetValidRow()`
- 内建接口签名要求显式传入 `tmp` 操作数。

### NPU 实现检查

- 支持的元素类型 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）：`half`、`float`、`int32_t`、`int16_t`。
- 支持的元素类型 (Ascend 950PR/Ascend 950DT)：`half`、`float`、`int32_t`、`int64_t`、`uint64_t`、`int16_t`。
- 实现同时接受ND输出和 `Cols == 1` 的DN输出，并非仅支持DN输出。
- 运行时检查遵循共享的行归约检查路径：
    - `src.GetValidRow() != 0`
    - `src.GetValidCol() != 0`
    - `src.GetValidRow() == dst.GetValidRow()`

### CPU_SIM实现检查

目标架构取自调用线程的内存模型，不取决于宿主机硬件；设置方式见
[选择模拟目标架构](../coding/cpu_sim_zh.md#选择模拟目标架构)。

- **A5**：输入输出类型必须相同，支持 `half`、`float`、`int16_t`、`int32_t`、
  `int64_t` 和 `uint64_t`，不支持原生 BF16 或混合输入输出类型。
  上述 Vec、ND/DN 布局、输入非空及有效行数相等约束均在运行时检查；
  输入输出的物理行数可以不同。
- **A2A3 兼容路径**：支持的输入/输出类型组合为 `half`/`half`、`half`/`float`、
  `bfloat16_t`/`bfloat16_t`、`bfloat16_t`/`float`、`float`/`float`、
  `int16_t`/`int16_t`、`int16_t`/`int32_t` 和 `int32_t`/`int32_t`。
  物理行数必须相等，但不执行 A5 专用的布局及有效区域断言。
  这些兼容行为不扩展 A2A3 NPU 的接口约束。
- 两条 CPU 路径都不支持的类型组合在编译期拒绝；仅所选架构不支持的类型组合在运行时拒绝。
  架构相关断言失败会终止进程，定义 `NDEBUG` 不会关闭这些检查。
- 调用方须保证动态有效形状位于物理 Tile 内，并提供覆盖全部输出行及第零列的目标存储和有效区域。
  当前实现不检查 `dst.GetValidCol()`，也不提供通用动态形状边界检查。
  两条路径均只写被处理行的第零列，保持其余目标元素及有效形状元数据不变。
- **数值差异**：A5 浮点路径采用分组二叉树归约，每次加法转换回元素类型；
  整数路径按输出位宽回绕，不饱和。A2A3 保留原累加循环，`half`/`bfloat16_t`
  输出使用 `float` 累加后转换，其他输出使用自身类型，未采用 A5 的无符号模运算溢出处理。
  A2A3 行主序循环的向量化提示可能重排浮点加法，即使未启用 `-ffast-math`；
  不保证严格从左到右累加，也不保证不同编译器的结果逐位一致。
- 不承诺完整的真机逐位等价：A5 的 NaN payload、非正规数/flush-to-zero 等浮点细节尚未证明等价，
  A2A3 路径仅用于兼容。依赖 A5 归约顺序时，不应启用 `-ffast-math` 等允许浮点重结合的选项。
  分组顺序、舍入及 BF16 占位类型说明见 [TROWSUM 实现说明](../coding/cpu_sim_zh.md#trowsum-实现说明)。

## 临时空间

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品

`tmp` **被使用**作为行归约的暂存存储。

- 对于**整数**类型（`int32_t`、`int16_t`）：`tmp` 用作逐行累加器缓冲区（1个块）。对于每一行，`tmp` 初始化为0，然后通过 `vadd` 累加 `src` 的各个块。最终求和结果在标量模式下从 `tmp` 读取。
  - `tmp` 大小：至少1行和 `BLOCK_BYTE_SIZE / sizeof(T)` 列（`int32_t` 为8，`int16_t` 为16）。
- 对于**浮点**类型（`float`、`half`）：`tmp` 用于通过 `vcadd`/`vcgadd` 的二叉树归约。
  - 安全的默认设置：将 `tmp` 设为与 `src` 相同的形状。

### Ascend 950PR/Ascend 950DT

`tmp` 被接口接受但Ascend 950PR/Ascend 950DT实现**不使用**。Ascend 950PR/Ascend 950DT后端使用基于向量寄存器的归约（`vcadd` 指令），不需要暂存Tile存储。`tmp` 仅为了与Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品的API兼容性而保留在C++内建接口签名中。

### CPU_SIM

两种架构路径均接受 `tmp` 以兼容接口，但不访问其存储。跨后端 kernel 仍须保留目标 NPU 后端要求的临时空间。

## 示例

以下片段仅展示调用和存储绑定方式，不是完整可执行程序；归约前须先初始化源数据。
在 CPU_SIM 下编译时需定义 `__CPU_SIM`，自动模式示例还需定义 `__PTO_AUTO__`。
要验证 A5 路径，应按 CPU_SIM 指南在应用启动阶段选择 A5；这两个函数自身不选择架构，
否则使用已配置的默认架构（初始为 A2A3）。

### 自动（Auto）

```cpp
#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWSUM(dst, src, tmp);
}
```

### 手动（Manual）

```cpp
#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TROWSUM(dst, src, tmp);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO汇编形式

```text
%dst = trowsum %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
