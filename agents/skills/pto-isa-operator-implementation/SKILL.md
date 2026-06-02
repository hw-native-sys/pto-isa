---
name: PTO-ISA算子实现指南
description: 使用PTO-ISA实现指定算子功能的完整流程指南，涵盖ISA指令选择、数据流分析、指令功能解释和kernel代码生成
license: CANN Open Software License Agreement Version 2.0
---

# PTO-ISA算子实现指南

本指南为使用PTO-ISA实现指定算子功能提供完整的流程指导。

## 目录
1. [概述](#概述)
2. [工作流程](#工作流程)
3. [步骤详解](#步骤详解)
4. [ISA指令分类参考](#isa指令分类参考)
5. [数据流分析框架](#数据流分析框架)
6. [示例](#示例)
7. [最佳实践](#最佳实践)
8. [常见问题](#常见问题)

## 概述

本skill专门用于帮助开发者从PTO-ISA指令集中选择合适的指令来实现指定的算子功能，并生成完整的kernel代码。

### 适用场景

- 用户指定具体的算子功能需求（如: "实现GELU激活函数"、"实现Batch Normalization"）
- 需要分析PTO-ISA指令集并选择合适的指令组合
- 需要理解数据在硬件上的流动过程
- 需要生成可直接使用的kernel代码

### 关键特性

- **ISA指令选择**: 从PTOISA_zh.md文档中分析并选择合适的指令
- **数据流分析**: 按数据处理顺序分析数据流向
  - **Vector计算**: gm → ub → vector → ub → gm
  - **Cube计算(矩阵乘)**: GM → L1 → L0A/L0B → L0C → GM
- **指令功能解释**: 详细解释每个ISA指令在算子实现中的作用
- **代码生成**: 输出完整的kernel代码实现

## 工作流程

当用户指定算子功能后，遵循以下工作流程：

```
用户指定算子功能
    ↓
步骤1: 阅读PTOISA_zh.md
    ↓
步骤2: 分析算子需求，列举ISA指令
    ↓
步骤3: 按数据流顺序解释指令功能
    ↓
步骤4: 输出kernel代码实现
```

## 步骤详解

### 步骤1: 阅读PTOISA_zh.md文档

**目标**: 全面了解PTO-ISA指令集，识别可能与算子相关的指令类别。

**行动**:
1. 阅读文档路径: `pto-isa/docs/PTOISA_zh.md`
2. 重点关注指令索引表，识别以下类别的指令：
   - **内存指令**: TLOAD, TSTORE, TPREFETCH (GM <-> Tile数据搬运)
   - **逐元素计算**: TADD, TSUB, TMUL, TDIV, TMAX, TMIN (Tile-Tile操作)
   - **标量操作**: TADDS, TSUBS, TMULS, TDIVS (Tile-标量操作)
   - **数学函数**: TLOG, TEXP, TSQRT, TRSQRT, TPOW (数学运算)
   - **激活函数**: TRELU, TPRELU, TLRELU (激活操作)
   - **轴归约/扩展**: TROWSUM, TCOLSUM, TROWMAX, TCOLMAX (轴操作)
   - **广播操作**: TROWEXPANDADD, TCOLEXPANDADD (广播加法)
   - **类型转换**: TCVT (类型转换)
   - **选择操作**: TSEL, TSELS (条件选择)
   - **矩阵操作**: TMATMUL, TGEMV (矩阵计算)

3. 记录每个相关指令的：
   - 指令名称
   - 功能描述
   - 所属类别
   - 适用场景

**输出**: 相关ISA指令列表

### 步骤2: 整理并列举需要使用的ISA指令

**目标**: 根据算子功能需求，确定具体的ISA指令组合。

**分析框架**:

#### 2.1 算子功能分解

将算子功能分解为基本操作：

| 算子类型 | 分解步骤 | 典型指令 |
|---------|---------|---------|
| 激活函数 | 输入加载 + 计算 + 输出存储 | TLOAD + TEXP/TLOG/TRELU + TSTORE |
| 归约操作 | 输入加载 + 归约 + 输出存储 | TLOAD + TROWSUM/TCOLSUM + TSTORE |
| 逐元素运算 | 输入加载 + 运算 + 输出存储 | TLOAD + TADD/TSUB/TMUL/TDIV + TSTORE |
| 广播操作 | 输入加载 + 广播 + 运算 + 存储 | TLOAD + TROWEXPANDADD + TSTORE |
| 矩阵运算(Cube) | 输入加载 + 数据搬运 + 矩阵乘 + 输出存储 | TLOAD + TMOV + TMATMUL + TSTORE (GM→L1→L0A/L0B→L0C→GM) |
| 类型转换 | 输入加载 + 转换 + 输出存储 | TLOAD + TCVT + TSTORE |
| 条件运算 | 输入加载 + 比较 + 选择 + 存储 | TLOAD + TCMP + TSEL + TSTORE |

#### 2.2 ISA指令选择原则

**最小化原则**: 使用最少的指令完成功能，减少数据搬运。

**数据流优化**: 
- **Vector计算**: 遵循 gm → ub → vector → ub → gm 的基本数据流
- **Cube计算(矩阵乘)**: 遵循 GM → L1 → L0A/L0B → L0C → GM 的矩阵数据流

**同步考虑**: 指令间使用Event同步或手动标志同步。

**示例**:

```
算子: GELU激活函数
GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

指令选择:
1. TLOAD: 从GM加载输入x到UB
2. TMUL: 计算 x^3 (x * x * x)
3. TMULS: 计算 0.044715 * x^3 (标量乘法)
4. TADD: 计算 x + 0.044715 * x^3
5. TMULS: 计算 sqrt(2/π) * (x + ...) (标量乘法)
6. TEXP/TLOG: 计算tanh函数 (可选，或使用近似)
7. TADD: 计算 1 + tanh(...)
8. TMULS: 计算 0.5 * (结果) (标量乘法)
9. TMUL: 计算 x * 最终结果
10. TSTORE: 将结果从UB存储到GM
```

**输出**: 按执行顺序排列的ISA指令列表

### 步骤3: 按数据处理顺序详细解释指令功能

**目标**: 详细说明每个指令在数据流中的作用。

**数据流框架**:

#### Vector计算数据流

```
数据流向: gm → ub → vector → ub → gm
阶段1: GM → UB (数据加载，使用TLOAD)
阶段2: UB → Vector (计算准备)
阶段3: Vector计算 (核心计算，使用TADD/TMUL/TEXP等)
阶段4: Vector → UB (计算结果)
阶段5: UB → GM (数据存储，使用TSTORE)
```

#### Cube计算数据流 (矩阵乘)

```
数据流向: GM → L1 → L0A/L0B → L0C → GM
阶段1: GM → L1 (矩阵数据加载，使用TLOAD)
阶段2: L1 → L0A/L0B (数据搬运到矩阵计算单元，使用TMOV)
阶段3: Cube计算 (矩阵乘法，使用TMATMUL，结果到L0C)
阶段4: L0C → GM (计算结果存储，使用TSTORE)
```

**关键区别**:
- **Vector计算**: 使用UB(Unified Buffer)作为中间缓冲区，执行逐元素操作
- **Cube计算**: 使用L1和L0缓冲区(L0A/L0B/L0C)，执行矩阵乘法操作

#### 3.1 指令功能解释模板

对每个指令，按以下模板解释：

```
指令: [指令名称]
阶段: [GM/UB/Vector阶段]
功能: [具体功能描述]
数据流: [输入 → 输出的数据流向]
示例: [具体使用示例]
同步需求: [是否需要同步，如何同步]
```

#### 3.2 阶段详解

**阶段1: GM → UB (数据加载)**

```
指令: TLOAD
功能: 从GlobalTensor (GM) 加载数据到Tile (UB)
数据流: GlobalMemory[srcGlobal] → UnifiedBuffer[srcTile]
输入: GlobalTensor对象，描述GM上的数据布局
输出: Tile对象，存储加载到UB的数据
同步需求: 
  - 推荐使用Event同步: Event<Op::TLOAD, Op::NextOp>
  - 或手动同步: set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0)

示例:
TLOAD(srcTile, srcGlobal);
event0 = TLOAD(src1Tile, src1Global);  // 带Event返回
```

**阶段2/3: Vector计算 (核心计算)**

根据具体算子，选择相应的计算指令：

```
逐元素加法: TADD
功能: 两个Tile的逐元素加法
数据流: UB[src0Tile] + UB[src1Tile] → UB[dstTile]
输入: 两个源Tile
输出: 一个目标Tile
同步需求: Event<Op::TLOAD, Op::TADD>

示例:
event1 = TADD(dstTile, src0Tile, src1Tile, event0);
```

```
标量乘法: TMULS
功能: Tile与标量的逐元素乘法
数据流: UB[srcTile] * scalar → UB[dstTile]
输入: 一个源Tile + 一个标量值
输出: 一个目标Tile
同步需求: Event<Op::PreviousOp, Op::TMULS>

示例:
event2 = TMULS(dstTile, srcTile, (T)scalar, event1);
```

```
指数运算: TEXP
功能: Tile的逐元素指数运算 (e^x)
数据流: exp(UB[srcTile]) → UB[dstTile]
输入: 一个源Tile
输出: 一个目标Tile
同步需求: Event<Op::PreviousOp, Op::TEXP>

示例:
event3 = TEXP(dstTile, srcTile, event2);
```

```
最大值选择: TMAX
功能: 两个Tile的逐元素最大值
数据流: max(UB[src0Tile], UB[src1Tile]) → UB[dstTile]
输入: 两个源Tile
输出: 一个目标Tile
同步需求: Event<Op::PreviousOp, Op::TMAX]

示例:
event4 = TMAX(dstTile, src0Tile, src1Tile, event3);
```

**Cube计算阶段: 矩阵乘法 (GM → L1 → L0A/L0B → L0C → GM)**

```
矩阵乘法: TMATMUL
功能: 矩阵乘法计算 C = A * B
数据流: 
  - GM → L1: GlobalMemory[矩阵A/B] → L1Buffer[MatTile] (TLOAD)
  - L1 → L0A/L0B: L1Buffer[MatTile] → L0Buffer[Left/RightTile] (TMOV)
  - L0A/L0B → L0C: 矩阵乘法计算 (TMATMUL)
  - L0C → GM: L0Buffer[AccTile] → GlobalMemory[结果] (TSTORE)
输入: 矩阵A和B (通过MatTile加载)
输出: 矩阵C (通过AccTile存储)
同步需求: 
  - TLOAD完成后: Event<Op::TLOAD, Op::TMOV> 或 set_flag(PIPE_MTE2, PIPE_MTE1)
  - TMOV完成后: Event<Op::TMOV, Op::TMATMUL> 或 set_flag(PIPE_MTE1, PIPE_M)
  - TMATMUL完成后: Event<Op::TMATMUL, Op::TSTORE_VEC> 或 set_flag(PIPE_M, PIPE_FIX)

示例:
// 1. 加载矩阵到L1
TLOAD(aMatTile, src0Global);
TLOAD(bMatTile, src1Global);

// 2. 搬运数据到L0A/L0B
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
TMOV(aTile, aMatTile);
TMOV(bTile, bMatTile);

// 3. 矩阵乘法计算
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
TMATMUL(cTile, aTile, bTile);

// 4. 存储结果
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
TSTORE(dstGlobal, cTile);
```

**阶段5: UB → GM (数据存储)**

```
指令: TSTORE
功能: 将Tile数据存储到GlobalTensor (GM)
数据流: UnifiedBuffer[dstTile] → GlobalMemory[dstGlobal]
输入: Tile对象，UB上的数据
输出: GlobalTensor对象，GM上的数据
同步需求: 
  - 推荐使用Event同步: Event<Op::LastOp, Op::TSTORE_VEC>
  - 或手动同步: set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0)

示例:
TSTORE(dstGlobal, dstTile, eventLast);
```

**输出**: 按数据流顺序的完整指令功能解释文档

### 步骤4: 输出kernel代码实现

**目标**: 生成完整的、可运行的kernel代码。

**代码结构**:

```cpp
/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
...
*/

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

namespace OperatorName {

// ==================== Device函数 ====================
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runOperator(__gm__ T *out, __gm__ T *src0, ...)
{
    // 1. 类型定义
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStrideDim5 = Stride<1, 1, 1, vCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStrideDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    // 2. Tile和GlobalTensor声明
    TileData src0Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, sizeof(T) * TileData::Numel);

    GlobalData src0Global(src0);
    GlobalData dstGlobal(out);

    // 3. Event声明 (推荐使用Event同步)
    Event<Op::TLOAD, Op::CALC_OP> event0;
    Event<Op::CALC_OP, Op::TSTORE_VEC> event1;

    // 4. 数据加载 (gm → ub)
    event0 = TLOAD(src0Tile, src0Global);

    // 5. 核心计算 (vector计算)
    event1 = CALC_OP(dstTile, src0Tile, ..., event0);

    // 6. 数据存储 (ub → gm)
    TSTORE(dstGlobal, dstTile, event1);

    out = dstGlobal.data();
}

// ==================== Host函数 ====================
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void launchOperator(T *out, T *src0, ..., void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runOperator<half, kTRows_, kTCols_, vRows, vCols>
            <<<1, nullptr, stream>>>((half *)out, (half *)src0, ...);
    } else {
        runOperator<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, ...);
    }
}

// ==================== 模板实例化 ====================
template void launchOperator<float, 64, 64, 64, 64>(float *out, float *src0, ..., void *stream);
template void launchOperator<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src0, ..., void *stream);

} // namespace OperatorName
```

**代码生成要点**:

1. **命名规范**: kernel文件命名为 `t<操作指令>_kernel.cpp`
2. **模板参数**: T (数据类型), kTRows_, kTCols_ (Tile维度), vRows, vCols (有效数据维度)
3. **类型转换**: aclFloat16需要转换为half类型
4. **缓冲区分配**: 使用TASSIGN紧密排布Tile地址
5. **同步策略**: 推荐使用Event同步，备选手动标志同步
6. **模板实例化**: 为常用配置提供实例化

**输出**: 完整的kernel代码文件

## ISA指令分类参考

### 内存指令 (GM <-> Tile)

| 指令 | 功能 | 数据流 | 适用场景 |
|------|------|--------|---------|
| TLOAD | GM → UB/L1 | GlobalMemory → UnifiedBuffer/L1Buffer | Vector和Cube计算 |
| TSTORE | UB/L0C → GM | UnifiedBuffer/L0Buffer → GlobalMemory | Vector和Cube计算 |
| TPREFETCH | 预取到UB缓存 | 提示性预取 | Vector计算优化 |
| MGATHER | 索引收集加载 | GM[索引] → UB | Vector计算 |
| MSCATTER | 索引散播存储 | UB → GM[索引] | Vector计算 |

**注意**:
- TLOAD可以加载到UB (Vector计算) 或L1Buffer (Cube计算)
- TSTORE可以存储UB (Vector计算) 或L0C (Cube计算) 的数据

### 逐元素计算指令 (Tile-Tile)

| 指令 | 功能 | 表达式 |
|------|------|--------|
| TADD | 逐元素加法 | dst = src0 + src1 |
| TSUB | 逐元素减法 | dst = src0 - src1 |
| TMUL | 逐元素乘法 | dst = src0 * src1 |
| TDIV | 逐元素除法 | dst = src0 / src1 |
| TMAX | 逐元素最大值 | dst = max(src0, src1) |
| TMIN | 逐元素最小值 | dst = min(src0, src1) |
| TCMP | 比较(生成掩码) | predicate = cmp(src0, src1) |
| TSHL | 逐元素左移 | dst = src0 << src1 |
| TSHR | 逐元素右移 | dst = src0 >> src1 |
| TAND | 逐元素按位与 | dst = src0 & src1 |
| TOR | 逐元素按位或 | dst = src0 | src1 |
| TXOR | 逐元素按位异或 | dst = src0 ^ src1 |
| TNOT | 逐元素按位取反 | dst = ~src |

### 标量操作指令 (Tile-标量)

| 指令 | 功能 | 表达式 |
|------|------|--------|
| TADDS | Tile加标量 | dst = src + scalar |
| TSUBS | Tile减标量 | dst = src - scalar |
| TMULS | Tile乘标量 | dst = src * scalar |
| TDIVS | Tile除标量 | dst = src / scalar |
| TMINS | Tile与标量最小值 | dst = min(src, scalar) |
| TMAXS | Tile与标量最大值 | dst = max(src, scalar) |
| TCMPS | Tile与标量比较 | predicate = cmp(src, scalar) |
| TEXPANDS | 标量广播到Tile | dst = broadcast(scalar) |

### 数学函数指令

| 指令 | 功能 | 表达式 |
|------|------|--------|
| TLOG | 自然对数 | dst = log(src) |
| TEXP | 指数运算 | dst = exp(src) |
| TSQRT | 平方根 | dst = sqrt(src) |
| TRSQRT | 倒数平方根 | dst = 1/sqrt(src) |
| TPOW | 幂运算 | dst = src0 ^ src1 |
| TRECIP | 倒数 | dst = 1/src |
| TABS | 绝对值 | dst = abs(src) |
| TNEG | 取负 | dst = -src |

### 激活函数指令

| 指令 | 功能 | 表达式 |
|------|------|--------|
| TRELU | ReLU | dst = max(0, src) |
| TPRELU | PReLU | dst = max(0, src) + slope * min(0, src) |
| TLRELU | Leaky ReLU (标量斜率) | dst = max(0, src) + scalar * min(0, src) |

### 轴归约/扩展指令

| 指令 | 功能 | 操作 |
|------|------|------|
| TROWSUM | 行求和 | 每行所有列求和 |
| TROWPROD | 行乘积 | 每行所有列乘积 |
| TROWMAX | 行最大值 | 每行所有列最大值 |
| TROWMIN | 行最小值 | 每行所有列最小值 |
| TROWARGMAX | 行argmax | 每行最大值列索引 |
| TROWARGMIN | 行argmin | 每行最小值列索引 |
| TCOLSUM | 列求和 | 每列所有行求和 |
| TCOLPROD | 列乘积 | 每列所有行乘积 |
| TCOLMAX | 列最大值 | 每列所有行最大值 |
| TCOLMIN | 列最小值 | 每列所有行最小值 |
| TCOLARGMAX | 列argmax | 每列最大值行索引 |
| TCOLARGMIN | 列argmin | 每列最小值行索引 |
| TROWEXPAND | 行广播 | 将行首元素广播到整行 |
| TCOLEXPAND | 列广播 | 将列首元素广播到整列 |

### 广播运算指令

| 指令 | 功能 | 操作 |
|------|------|------|
| TROWEXPANDADD | 行广播加法 | 每行 + 广播标量向量 |
| TROWEXPANDSUB | 行广播减法 | 每行 - 广播标量向量 |
| TROWEXPANDMUL | 行广播乘法 | 每行 * 广播标量向量 |
| TROWEXPANDDIV | 行广播除法 | 每行 / 广播标量向量 |
| TROWEXPANDMAX | 行广播最大值 | max(每行, 广播标量向量) |
| TROWEXPANDMIN | 行广播最小值 | min(每行, 广播标量向量) |
| TROWEXPANDEXPDIF | 行指数差 | exp(每行 - 广播标量向量) |
| TCOLEXPANDADD | 列广播加法 | 每列 + 广播标量向量 |
| TCOLEXPANDSUB | 列广播减法 | 每列 - 广播标量向量 |
| TCOLEXPANDMUL | 列广播乘法 | 每列 * 广播标量向量 |
| TCOLEXPANDDIV | 列广播除法 | 每列 / 广播标量向量 |
| TCOLEXPANDMAX | 列广播最大值 | max(每列, 广播标量向量) |
| TCOLEXPANDMIN | 列广播最小值 | min(每列, 广播标量向量) |
| TCOLEXPANDEXPDIF | 列指数差 | exp(每列 - 广播标量向量) |

### 数据搬运指令 (缓冲区数据移动)

| 指令 | 功能 | 数据流 | 适用场景 |
|------|------|--------|---------|
| TMOV | L1 → L0A/L0B | MatTile → LeftTile/RightTile | Cube计算 (矩阵乘法) |
| TMOV | Tile之间移动 | srcTile → dstTile | 数据格式转换 |
| TMOV_FP | 带缩放的移动 | srcTile * scale → dstTile | 量化操作 |
| TRESHAPE | Tile重解释 | 保持字节，改变类型/形状 | 类型转换 |
| TTRANS | Tile转置 | srcTile^T → dstTile | 矩阵转置 |

**TMOV在矩阵乘法中的关键作用**:
- 将L1Buffer的MatTile数据搬运到L0Buffer
- 准备LeftTile和RightTile供Cube计算单元使用
- 数据流: L1 → L0A/L0B → TMATMUL → L0C

### 类型转换与选择指令

| 指令 | 功能 | 操作 |
|------|------|------|
| TCVT | 类型转换 | src_type → dst_type |
| TSEL | 条件选择(Tile) | mask ? src0 : src1 |
| TSELS | 条件选择(Tile-标量) | mask ? src : scalar |

### 矩阵运算指令 (使用Cube核心)

**重要**: 矩阵运算使用Cube核心，数据流为 **GM → L1 → L0A/L0B → L0C → GM**

| 指令 | 功能 | 数据流 | 表达式 |
|------|------|--------|--------|
| TLOAD | GM → L1 | GlobalMemory → L1Buffer (加载矩阵数据) | MatTile加载 |
| TMOV | L1 → L0A/L0B | L1Buffer → L0Buffer (搬运到计算单元) | MatTile → LeftTile/RightTile |
| TMATMUL | L0A/L0B → L0C | Cube矩阵乘法计算 | C = A * B |
| TSTORE | L0C → GM | L0Buffer → GlobalMemory (存储结果) | AccTile → GlobalMemory |
| TMATMUL_ACC | 矩阵乘法(累加) | L0A/L0B → L0C (带累加) | C = A * B + C |
| TMATMUL_BIAS | 矩阵乘法(加偏置) | L0A/L0B → L0C + bias | C = A * B + bias |
| TGEMV | 矩阵向量乘 | L0A/L0B → L0C | y = A * x |
| TGEMV_ACC | 矩阵向量乘(累加) | L0A/L0B → L0C (带累加) | y = A * x + y |
| TGEMV_BIAS | 矩阵向量乘(加偏置) | L0A/L0B → L0C + bias | y = A * x + bias |

**矩阵乘法完整数据流示例**:
```
GM → L1 (TLOAD) → L0A/L0B (TMOV) → L0C (TMATMUL) → GM (TSTORE)

详细步骤:
1. TLOAD: 加载矩阵A和B从GM到L1Buffer (MatTile)
2. TMOV: 将MatTile数据搬运到L0Buffer (LeftTile和RightTile)
3. TMATMUL: 在Cube核心执行矩阵乘法，结果存储到L0C (AccTile)
4. TSTORE: 将AccTile结果存储到GM
```

### 三元运算指令

| 指令 | 功能 | 表达式 |
|------|------|--------|
| TADDC | 三元加法 | dst = src0 + src1 + src2 |
| TSUBC | 三元减法 | dst = src0 - src1 + src2 |
| TADDSC | Tile+标量+Tile加法 | dst = src0 + scalar + src1 |
| TSUBSC | Tile-标量+Tile运算 | dst = src0 - scalar + src1 |

## 数据流分析框架

### 基本数据流模式

#### Vector计算数据流 (逐元素操作)

```
┌─────────────────────────────────────────────────────────────┐
│              Vector计算数据流 (gm → ub → vector → ub → gm)    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GlobalMemory (GM)                                           │
│       │                                                      │
│       │ TLOAD                                                │
│       ↓                                                      │
│  UnifiedBuffer (UB)                                          │
│       │                                                      │
│       │ 计算指令 (TADD/TMUL/TEXP等)                           │
│       ↓                                                      │
│  Vector计算单元                                               │
│       │                                                      │
│       │ 计算结果                                              │
│       ↓                                                      │
│  UnifiedBuffer (UB)                                          │
│       │                                                      │
│       │ TSTORE                                               │
│       ↓                                                      │
│  GlobalMemory (GM)                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Cube计算数据流 (矩阵乘法)

```
┌─────────────────────────────────────────────────────────────┐
│               Cube计算数据流 (GM → L1 → L0 → GM)               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GlobalMemory (GM)                                           │
│       │                                                      │
│       │ TLOAD                                                │
│       ↓                                                      │
│  L1Buffer (L1)                                               │
│       │                                                      │
│       │ TMOV                                                 │
│       ↓                                                      │
│  L0Buffer (L0A/L0B)                                          │
│       │                                                      │
│       │ TMATMUL                                              │
│       ↓                                                      │
│  L0Buffer (L0C)                                              │
│       │                                                      │
│       │ TSTORE                                               │
│       ↓                                                      │
│  GlobalMemory (GM)                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**两种数据流的区别**:

| 特性 | Vector计算 | Cube计算 |
|------|-----------|----------|
| 计算单元 | Vector Unit (PIPE_V) | Matrix Unit (PIPE_M) |
| 中间缓冲 | UnifiedBuffer (UB) | L1Buffer + L0Buffer (L0A/L0B/L0C) |
| 适用场景 | 逐元素操作 (TADD/TMUL/TEXP等) | 矩阵乘法 (TMATMUL) |
| 数据流路径 | GM → UB → V → UB → GM | GM → L1 → L0A/L0B → L0C → GM |
| 同步流水线 | MTE2 → V → MTE3 | MTE2 → MTE1 → M → FIX → MTE3 |

### 同步机制

**Event同步（推荐）**:
```cpp
Event<Op::TLOAD, Op::TADD> event0;
Event<Op::TADD, Op::TSTORE_VEC> event1;

event0 = TLOAD(srcTile, srcGlobal);         // TLOAD完成时event0触发
event1 = TADD(dstTile, src0Tile, src1Tile, event0);  // 等待event0，完成后触发event1
TSTORE(dstGlobal, dstTile, event1);         // 等待event1
```

**手动标志同步**:
```cpp
TLOAD(src0Tile, src0Global);
TLOAD(src1Tile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);   // MTE2(内存加载) → V(向量计算)
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);  // 等待内存加载完成
#endif

TADD(dstTile, src0Tile, src1Tile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);   // V(向量计算) → MTE3(内存存储)
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);  // 等待计算完成
#endif

TSTORE(dstGlobal, dstTile);
```

### 流水线阶段

| 流水线 | 缩写 | 功能 | 适用场景 |
|--------|------|------|---------|
| Memory Transfer Engine 1 | PIPE_MTE1 | 矩阵数据搬运 (L1 → L0) | Cube计算 (矩阵乘法) |
| Memory Transfer Engine 2 | PIPE_MTE2 | 向量/矩阵数据加载 (GM → UB/L1) | Vector和Cube计算 |
| Memory Transfer Engine 3 | PIPE_MTE3 | 数据存储 (UB/L0C → GM) | Vector和Cube计算 |
| Vector Unit | PIPE_V | 向量计算 (逐元素操作) | Vector计算 (TADD/TMUL等) |
| Matrix Unit | PIPE_M | 矩阵计算 (矩阵乘法) | Cube计算 (TMATMUL) |
| Fix Unit | PIPE_FIX | 格式转换 | Cube计算结果格式化 |
| Scalar Unit | PIPE_S | 标量计算 | 标量操作 |
| All Pipelines | PIPE_ALL | 所有流水线 | 全局同步 |

**流水线同步策略**:
- **Vector计算**: MTE2 (GM → UB) → V (计算) → MTE3 (UB → GM)
- **Cube计算**: MTE2 (GM → L1) → MTE1 (L1 → L0) → M (计算) → FIX (格式转换) → MTE3 (L0 → GM)

### 核间同步机制 (TPUSH/TPOP)

**重要**: Cube核与Vector核之间的数据传输必须使用TPUSH/TPOP，不能使用简单的TMOV。

#### 核间同步架构

```
┌─────────────────────────────────────────────┐
│         A5 AI Core Architecture             │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────┐  ┌──────────────────┐     │
│  │  Cube Core   │  │  Vector Core 0   │     │
│  │  (PIPE_M)    │  │  (PIPE_V)        │     │
│  │  L1/L0       │  │  UB              │     │
│  └              │  │                  │     │
│  │  TPUSH (C2V) │  │  TPOP (C2V)      │     │
│  │  TPOP (V2C)  │  │  TPUSH (V2C)     │     │
│  └              │  │                  │     │
│  └──────────────┘  └──────────────────┘     │
│                                              │
│  ┌──────────────────┐                       │
│  │  Vector Core 1   │                       │
│  │  (PIPE_V)        │                       │
│  │  UB              │                       │
│  │                  │                       │
│  │  TPOP (C2V)      │                       │
│  │  TPUSH (V2C)     │                       │
│  │                  │                       │
│  └──────────────────┘                       │
│                                              │
│  核间同步：TPUSH → TPOP → TFREE              │
│                                              │
└─────────────────────────────────────────────┘
```

#### 核间同步方向类型

| 方向类型 | 定义 | 数据流 | 生产者流水线 | 消费者流水线 |
|---------|------|--------|-------------|-------------|
| **DIR_C2V** | Cube → Vector | L0C → UB | PIPE_FIX | PIPE_V |
| **DIR_V2C** | Vector → Cube | UB → L1 | PIPE_MTE3 | PIPE_MTE1 |
| **DIR_BOTH** | 双向 | L0C ↔ UB | PIPE_FIX + PIPE_MTE3 | PIPE_V + PIPE_MTE1 |

#### Vector/Cube核代码区分宏定义

**重要**: PTO-ISA使用编译器宏来区分Vector核和Cube核的执行路径。同一份kernel代码会被编译两次，分别生成Vector核和Cube核的可执行文件。

**宏定义模式**:

```cpp
// 编译器在编译不同核时自动定义以下宏：
// - 编译Vector核时：定义 __DAV_VEC__
// - 编译Cube核时：定义 __DAV_CUBE__

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif
```

**使用示例**:

```cpp
template <typename T, int M, int K, int N>
__global__ AICORE void runOperator(__gm__ T *out, __gm__ T *srcA, __gm__ T *srcB)
{
    // Vector核执行路径
    if constexpr (DAV_VEC) {
        // Vector计算：逐元素操作、激活函数、归约等
        TLOAD(vecTile, srcGlobal);
        TADD(dstTile, src0Tile, src1Tile);
        
        // V2C: TPUSH数据到Cube核
        TPUSH<V2CPipe, VecTileNZ, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        
        // C2V: TPOP从Cube核接收数据
        TPOP<C2VPipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, recvTile);
        TFREE<C2VPipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        
        TSTORE(dstGlobal, dstTile);
    }
    
    // Cube核执行路径
    if constexpr (DAV_CUBE) {
        // Cube计算：矩阵乘法
        TLOAD(matTileA, srcAGlobal);
        TLOAD(matTileB, srcBGlobal);
        
        // V2C: TPOP从Vector核接收数据
        TPOP<V2CPipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, matTileB);
        TFREE<V2CPipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        
        TMOV(leftTile, matTileA);
        TMOV(rightTile, matTileB);
        TMATMUL(accTile, leftTile, rightTile);
        
        // C2V: TPUSH数据到Vector核
        TPUSH<C2VPipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, accTile);
        
        TSTORE(dstGlobal, accTile);
    }
}
```

**宏定义规则**:

| 宏名称 | 定义时机 | 适用场景 |
|--------|---------|---------|
| `__DAV_VEC__` | 编译Vector核时 | Vector计算、UB操作、PIPE_V流水线 |
| `__DAV_CUBE__` | 编译Cube核时 | Cube计算、L1/L0操作、PIPE_M流水线 |

**注意事项**:
1. 使用 `if constexpr (DAV_VEC)` 和 `if constexpr (DAV_CUBE)` 进行分支判断
2. 不要在同一个核的执行路径中混用另一核的Tile类型
3. TPUSH/TPOP必须成对使用：Vector核TPUSH对应Cube核TPOP
4. 每个TPOP后必须调用TFREE释放缓冲区

#### TPUSH三步流程

TPUSH用于生产者核推送数据到消费者核：

**步骤1: Alloc (分配空间)**
- 生产者核等待消费者核释放空间
- C2V: `wait_intra_block(PIPE_FIX, FlagID+1)`
- V2C: `wait_intra_block(PIPE_MTE3, FlagID+1)`

**步骤2: Store (写入数据)**
- 根据TileType和FIFO类型选择搬运方式
- AccTile → VecFIFO: `pushAcc2VecFiFo` (L0C → UB)
- VecTile → MatFIFO: `pushVec2MatFiFo` (UB → L1)

**步骤3: Commit (信号通知)**
- 通知消费者核数据已就绪
- C2V: `set_intra_block(PIPE_FIX, FlagID)`
- V2C: `set_intra_block(PIPE_MTE3, FlagID)`

#### TPOP三步流程

TPOP用于消费者核从生产者核读取数据：

**步骤1: Wait (等待数据)**
- 消费者核等待生产者核数据就绪
- C2V: `wait_intra_block(PIPE_V, FlagID)`
- V2C: `wait_intra_block(PIPE_MTE1, FlagID)`

**步骤2: Pop (读取数据)**
- 根据TileType和FIFO类型选择读取方式
- VecFIFO → VecTile: `popTileFromVecFiFo`
- MatFIFO → MatTile: `popTileFromMatFiFo`

**步骤3: Free (释放空间)**
- 通知生产者核空间已释放
- 使用TFREE指令

#### TPipe结构定义

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum>
using TPipe = TPipe<FlagID, DirType, SlotSize, SlotNum>;

// 参数说明:
// FlagID: 核间同步标志ID (0-7)
// DirType: 通信方向 (DIR_C2V=1, DIR_V2C=2, DIR_BOTH=3)
// SlotSize: FIFO槽大小（字节）
// SlotNum: FIFO槽数量（建议2）

// TPipe初始化:
// GM_SLOT_BUFFER: GM FIFO基地址
// C2V_CONSUMER_BUF: Cube→Vec消费者UB地址
// V2C_CONSUMER_BUF: Vec→Cube消费者L1地址
using MatPipe = TPipe<FLAG_ID, Direction::DIR_C2V, sizeof(T) * M * N, 2>;
MatPipe mPipe((__gm__ void *)(uint64_t)0x0, (uint32_t)0x0, (uint32_t)0x20000);
```

#### TileSplitAxis分块模式

| SplitAxis | 说明 | Vector核分配 |
|-----------|------|------------|
| **TILE_UP_DOWN** | 沿行分块 | Vec0处理上半部分，Vec1处理下半部分 |
| **TILE_LEFT_RIGHT** | 沿列分块 | Vec0处理左半部分，Vec1处理右半部分 |
| **TILE_NO_SPLIT** | 不分块 | 单Vector核处理全部 |

#### FlagID分配策略

A5架构提供**8个FlagID**（0-7），用于核间同步：

| FlagID | 用途 | 说明 |
|--------|------|------|
| **FlagID** | 数据就绪信号 | 生产者设置，消费者等待 |
| **FlagID+1** | 空间释放信号 | 消费者设置，生产者等待 |
| **FlagID+16** | Vec核1信号 | 双Vector核时使用 |

**双Vector核时的FlagID分配**:
```
Vec0: FlagID (主核)
Vec1: FlagID+16 (从核)

Cube核需要等待双核：
wait_intra_block(PIPE_FIX, FlagID);      // Vec0信号
wait_intra_block(PIPE_FIX, FlagID+16);    // Vec1信号
```

#### 核间同步最佳实践

**1. FlagID管理**: 为每个TPipe分配独立的FlagID，避免冲突

**2. FIFO深度设置**: 推荐使用深度=2

**3. 同步顺序匹配**: 一个TPUSH必须对应一个TPOP + TFREE

**4. 错误示例**:
```cpp
// 错误：连续两次TPUSH，没有对应的TPOP
TPUSH(pipe, tile1);
TPUSH(pipe, tile2);  // ERROR

// 正确：
TPUSH(pipe, tile1);
// ... 消费者核 ...
TPOP(pipe, vecTile1);
TFREE(pipe);
// 然后才能进行下一次TPUSH
```

#### 融合算子中的核间同步

当算子涉及Vector计算和Cube计算的交替使用时，需要在切换点使用TPUSH/TPOP：

| 切换场景 | 数据流 | 方向 | 核间同步 |
|---------|-------|------|---------|
| Vector → Cube | UB → L1 | V2C | TPUSH (Vec) + TPOP (Cube) |
| Cube → Vector | L0C → UB | C2V | TPUSH (Cube) + TPOP (Vec) |

**Flash Attention核间同步示例**:
- Phase 1 (Vector): K转置 → TPUSH K^T到Cube核 (V2C)
- Phase 2 (Cube): QK矩阵乘 → TPUSH Score到Vector核 (C2V)
- Phase 3 (Vector): Softmax归一化 → TPUSH P到Cube核 (V2C)
- Phase 4 (Cube): PV矩阵乘 → TSTORE输出

详细参考: `/home/developer/.agents/skills/pto-isa-operator-implementation/TPUSH_TPOP_GUIDE.md`

## 示例

### 示例1: ReLU激活函数

**算子功能**: ReLU(x) = max(0, x)

**ISA指令**: TLOAD → TRELU → TSTORE

**Kernel代码**:
```cpp
namespace ReLU {

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runReLU(__gm__ T *out, __gm__ T *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStrideDim5 = Stride<1, 1, 1, vCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStrideDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData srcTile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, sizeof(T) * TileData::Numel);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TRELU> event0;
    Event<Op::TRELU, Op::TSTORE_VEC> event1;

    event0 = TLOAD(srcTile, srcGlobal);
    event1 = TRELU(dstTile, srcTile, event0);
    TSTORE(dstGlobal, dstTile, event1);

    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void launchReLU(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runReLU<half, kTRows_, kTCols_, vRows, vCols>
            <<<1, nullptr, stream>>>((half *)out, (half *)src);
    } else {
        runReLU<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src);
    }
}

template void launchReLU<float, 64, 64, 64, 64>(float *out, float *src, void *stream);
template void launchReLU<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src, void *stream);

} // namespace ReLU
```

### 示例2: 逐元素加法 (TADD)

**算子功能**: dst = src0 + src1

**ISA指令**: TLOAD → TLOAD → TADD → TSTORE

**Kernel代码**:
```cpp
namespace TAdd {

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTAdd(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStrideDim5 = Stride<1, 1, 1, vCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStrideDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, sizeof(T) * TileData::Numel);
    TASSIGN(dstTile, 2 * sizeof(T) * TileData::Numel);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TADD> event0;
    Event<Op::TADD, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(src1Tile, src1Global);
    event1 = TADD(dstTile, src0Tile, src1Tile, event0);
    TSTORE(dstGlobal, dstTile, event1);

    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void launchTAdd(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTAdd<half, kTRows_, kTCols_, vRows, vCols>
            <<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1);
    } else {
        runTAdd<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTAdd<float, 64, 64, 64, 64>(float *out, float *src0, float *src1, void *stream);
template void launchTAdd<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);

} // namespace TAdd
```

### 示例3: Softmax归一化

**算子功能**: Softmax(x) = exp(x) / sum(exp(x))

**ISA指令**: TLOAD → TEXP → TCOLSUM → TCOLEXPANDDIV → TSTORE

**Kernel代码**:
```cpp
namespace Softmax {

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runSoftmax(__gm__ T *out, __gm__ T *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStrideDim5 = Stride<1, 1, 1, vCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStrideDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using SumTileData = Tile<TileType::Vec, T, vRows, 1, BLayout::ColMajor, -1, -1>;

    TileData srcTile(vRows, vCols);
    TileData expTile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    SumTileData sumTile(vRows, 1);
    
    TASSIGN(srcTile, 0x0);
    TASSIGN(expTile, sizeof(T) * TileData::Numel);
    TASSIGN(sumTile, 2 * sizeof(T) * TileData::Numel);
    TASSIGN(dstTile, 3 * sizeof(T) * TileData::Numel);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TEXP> event0;
    Event<Op::TEXP, Op::TCOLSUM> event1;
    Event<Op::TCOLSUM, Op::TCOLEXPANDDIV> event2;
    Event<Op::TCOLEXPANDDIV, Op::TSTORE_VEC> event3;

    event0 = TLOAD(srcTile, srcGlobal);
    event1 = TEXP(expTile, srcTile, event0);
    event2 = TCOLSUM(sumTile, expTile, event1);
    event3 = TCOLEXPANDDIV(dstTile, expTile, sumTile, event2);
    TSTORE(dstGlobal, dstTile, event3);

    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void launchSoftmax(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runSoftmax<half, kTRows_, kTCols_, vRows, vCols>
            <<<1, nullptr, stream>>>((half *)out, (half *)src);
    } else {
        runSoftmax<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src);
    }
}

template void launchSoftmax<float, 64, 64, 64, 64>(float *out, float *src, void *stream);
template void launchSoftmax<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src, void *stream);

} // namespace Softmax
```

### 示例4: Batch Normalization

**算子功能**: BN(x) = (x - mean) / sqrt(var + eps) * gamma + beta

**ISA指令**: TLOAD → TSUBS → TDIVS → TMULS → TADDS → TSTORE

**Kernel代码**:
```cpp
namespace BatchNorm {

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runBatchNorm(__gm__ T *out, __gm__ T *src,
                                    float mean, float var, float eps, float gamma, float beta)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStrideDim5 = Stride<1, 1, 1, vCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStrideDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData srcTile(vRows, vCols);
    TileData normTile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(normTile, sizeof(T) * TileData::Numel);
    TASSIGN(dstTile, 2 * sizeof(T) * TileData::Numel);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TSUBS> event0;
    Event<Op::TSUBS, Op::TDIVS> event1;
    Event<Op::TDIVS, Op::TMULS> event2;
    Event<Op::TMULS, Op::TADDS> event3;
    Event<Op::TADDS, Op::TSTORE_VEC> event4;

    T std_val = (T)sqrt(var + eps);

    event0 = TLOAD(srcTile, srcGlobal);
    event1 = TSUBS(normTile, srcTile, (T)mean, event0);
    event2 = TDIVS(normTile, normTile, std_val, event1);
    event3 = TMULS(dstTile, normTile, (T)gamma, event2);
    event4 = TADDS(dstTile, dstTile, (T)beta, event3);
    TSTORE(dstGlobal, dstTile, event4);

    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void launchBatchNorm(T *out, T *src, float mean, float var, float eps, float gamma, float beta, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runBatchNorm<half, kTRows_, kTCols_, vRows, vCols>
            <<<1, nullptr, stream>>>((half *)out, (half *)src, mean, var, eps, gamma, beta);
    } else {
        runBatchNorm<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src, mean, var, eps, gamma, beta);
    }
}

template void launchBatchNorm<float, 64, 64, 64, 64>(float *out, float *src, float mean, float var, float eps, float gamma, float beta, void *stream);
template void launchBatchNorm<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src, float mean, float var, float eps, float gamma, float beta, void *stream);

} // namespace BatchNorm
```

### 示例5: 矩阵乘法 (TMATMUL) - Cube核心

**算子功能**: C = A * B (矩阵乘法)

**ISA指令**: TLOAD → TMOV → TMATMUL → TSTORE (GM → L1 → L0 → L0C → GM)

**重要**: 矩阵乘法使用Cube核心，需要使用`DAV_CUBE`宏判断执行路径。

**Kernel代码**:
```cpp
namespace MatMul {

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

template <typename T, typename U, typename S, int validM, int validK, int validN>
__global__ AICORE void runMatMul(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    if constexpr (DAV_CUBE) {
        constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
        constexpr int M = CeilAlign<int>(validM, 16);
        constexpr int N = CeilAlign<int>(validN, blockAlign);
        constexpr int K = CeilAlign<int>(validK, blockAlign);

        using GlobalDataSrc0 = GlobalTensor<U, pto::Shape<1, 1, 1, validM, validK>,
                                            pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
        using GlobalDataSrc1 = GlobalTensor<S, pto::Shape<1, 1, 1, validK, validN>,
                                            pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
        using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
                                           pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;

        GlobalDataSrc0 src0Global(src0);
        GlobalDataSrc1 src1Global(src1);
        GlobalDataOut dstGlobal(out);

        using TileMatAData = Tile<TileType::Mat, U, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
        using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
        using LeftTile = TileLeft<U, M, K, validM, validK>;
        using RightTile = TileRight<S, K, N, validK, validN>;
        using AccTile = TileAcc<T, M, N, validM, validN>;

        TileMatAData aMatTile;
        TileMatBData bMatTile;
        LeftTile aTile;
        RightTile bTile;
        AccTile cTile;

        TASSIGN(aMatTile, 0x0);
        TASSIGN(bMatTile, 0x20000);

        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        TMATMUL(cTile, aTile, bTile);

#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

        TSTORE(dstGlobal, cTile);
        out = dstGlobal.data();
    }
}

template <typename T, typename U, typename S, int validM, int validK, int validN>
void launchMatMul(T *out, U *src0, S *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16> || std::is_same_v<U, aclFloat16> || std::is_same_v<S, aclFloat16>) {
        runMatMul<half, half, half, validM, validK, validN>
            <<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1);
    } else {
        runMatMul<T, U, S, validM, validK, validN><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchMatMul<float, float, float, 16, 16, 16>(float *out, float *src0, float *src1, void *stream);
template void launchMatMul<half, half, half, 16, 16, 16>(half *out, half *src0, half *src1, void *stream);

} // namespace MatMul
```

## 最佳实践

### 1. ISA指令选择原则

**最小化指令数量**: 使用最少指令完成功能，减少数据搬运开销。

**优先使用融合指令**: 选择融合指令减少中间步骤：
- TAXPY (融合乘加) - Vector计算
- TMATMUL_ACC (融合累加矩阵乘) - Cube计算
- TMATMUL_BIAS (融合加偏置矩阵乘) - Cube计算
- TROWEXPANDADD (融合广播加法) - Vector计算
- TADDC/TSUBC (融合三元运算) - Vector计算

**选择合适的数据流**: 根据算子特性选择最优数据流路径。
- **Vector计算**: 使用GM → UB → V → UB → GM数据流，适用于逐元素操作
- **Cube计算**: 使用GM → L1 → L0A/L0B → L0C → GM数据流，适用于矩阵乘法

### 2. 数据流优化

**Vector计算优化**:
- 尽量在UB上进行多次计算
- 使用原地操作减少中间Tile
- 合理使用缓冲区重用
- 避免不必要的TMOV操作

**Cube计算优化**:
- 合理规划L1和L0缓冲区大小
- 使用TileType::Mat和TileType::Vec的转换
- 优化矩阵分块策略
- 减少GM访问次数

**对齐和布局**:
- RowMajor: cols需要32字节对齐
- ColMajor: rows需要32字节对齐
- 使用constexpr计算对齐维度

### 3. 同步策略选择

**推荐Event同步**:
- 自动依赖跟踪
- 编译器优化友好
- 代码简洁易维护
- 支持手动和自动模式

**备选手动同步**:
- 复杂流水线控制
- 需要细粒度同步
- 性能关键路径优化

### 4. Tile维度选择

**常见Tile维度配置**:
| 数据类型 | 推荐Tile维度 |
|---------|-------------|
| float | 64x64, 32x32, 16x16 |
| aclFloat16 | 16x256, 8x768, 4x1024 |
| int32 | 64x64, 32x32 |
| int16 | 64x128, 32x256 |

### 5. 类型处理

**aclFloat16转换**:
- API类型: aclFloat16
- 硬件类型: half
- launch函数中进行转换

**混合精度支持**:
- 使用模板参数支持多类型
- 标量参数统一使用float

### 6. 代码组织

**命名规范**:
- Kernel文件: `t<操作指令>_kernel.cpp`
- 命名空间: `OperatorName`
- 函数名: `runOperator`, `launchOperator`

**模板实例化**:
- 为常用配置提供显式实例化
- 减少编译时间
- 确保代码可链接

## 常见问题

### Q1: 如何选择合适的ISA指令组合?

**回答**:
1. 分析算子功能，分解为基本操作
2. 在PTOISA_zh.md中查找对应指令
3. 按数据流顺序排列指令
4. 检查指令间的依赖关系
5. 考虑融合指令减少步骤

### Q2: 数据流顺序是什么?

**回答**:
PTO有两种主要的数据流模式:

**1. Vector计算数据流 (逐元素操作)**: gm → ub → vector → ub → gm
- **阶段1**: GM → UB (TLOAD)
- **阶段2/3**: Vector计算 (TADD/TMUL/TEXP等)
- **阶段5**: UB → GM (TSTORE)

**2. Cube计算数据流 (矩阵乘法)**: GM → L1 → L0A/L0B → L0C → GM
- **阶段1**: GM → L1 (TLOAD)
- **阶段2**: L1 → L0A/L0B (TMOV)
- **阶段3**: Cube计算 (TMATMUL)
- **阶段4**: L0C → GM (TSTORE)

**关键区别**:
- Vector计算使用UB缓冲区，适用于逐元素操作
- Cube计算使用L1和L0缓冲区，适用于矩阵乘法

### Q3: 何时使用Event同步，何时使用手动同步?

**回答**:
- **Event同步（推荐）**: 简单融合、清晰依赖关系、自动模式支持
- **手动同步**: 复杂流水线、细粒度控制、性能优化

### Q4: 如何处理复杂算子（如GELU、LayerNorm）?

**回答**:
1. 将复杂算子分解为多个基本操作
2. 为每个基本操作选择对应ISA指令
3. 合理安排中间Tile缓冲区
4. 优化数据流减少搬运
5. 考虑使用近似计算简化实现

### Q5: Tile维度如何选择?

**回答**:
- 根据数据类型选择对齐维度
- 考虑片上存储容量限制
- 平衡计算效率和存储开销
- 使用constexpr计算对齐维度

### Q6: 如何验证ISA指令选择是否正确?

**回答**:
1. 检查指令功能是否匹配算子需求
2. 验证数据流完整性（GM → UB → GM）
3. 确认同步机制正确设置
4. 在CPU模拟器上测试
5. 与golden结果对比验证

### Q7: 标量参数如何处理?

**回答**:
- 标量参数统一使用float类型
- 在指令调用前转换为Tile数据类型: `(T)scalar`
- 使用标量指令(TADDS/TMULS等)而不是Tile指令

### Q8: 何时使用Vector数据流，何时使用Cube数据流?

**回答**:
根据算子类型选择合适的数据流:

**使用Vector数据流 (GM → UB → V → UB → GM)**:
- 逐元素操作: TADD, TSUB, TMUL, TDIV, TMAX, TMIN
- 数学函数: TEXP, TLOG, TSQRT, TPOW
- 激活函数: TRELU, TPRELU, TLRELU
- 标量操作: TADDS, TMULS, TDIVS
- 轴归约: TROWSUM, TCOLSUM, TROWMAX
- 广播操作: TROWEXPANDADD, TCOLEXPANDADD
- 类型转换: TCVT

**使用Cube数据流 (GM → L1 → L0A/L0B → L0C → GM)**:
- 矩阵乘法: TMATMUL, TMATMUL_ACC, TMATMUL_BIAS
- 矩阵向量乘: TGEMV, TGEMV_ACC, TGEMV_BIAS
- 需要使用TileType::Mat的矩阵操作

**判断方法**:
- 如果使用TileType::Vec → Vector数据流
- 如果使用TileType::Mat → Cube数据流

### Q9: 矩阵乘法中TMOV的作用是什么?

**回答**:
TMOV在矩阵乘法中用于数据搬运:

**数据流**: L1 → L0A/L0B

**具体作用**:
- TLOAD将矩阵数据加载到L1Buffer (MatTile)
- TMOV将MatTile数据搬运到L0Buffer (LeftTile和RightTile)
- TMATMUL在L0Buffer执行计算

**为什么需要TMOV**:
- L1Buffer和L0Buffer是不同的物理存储区域
- L1Buffer用于存储加载的原始数据
- L0Buffer是Cube计算单元的专用缓冲区 (L0A/L0B)
- TMOV将数据从L1搬运到L0，准备矩阵乘法计算

### Q10: 核间同步(TPUSH/TPOP)何时使用?

**回答**:
当算子涉及Vector核和Cube核之间的数据传输时，必须使用TPUSH/TPOP:

**使用场景**:
- Vector核计算结果需要传给Cube核进行矩阵乘法
- Cube核矩阵乘法结果需要传给Vector核进行逐元素操作
- 融合算子中Vector/Cube交替使用

**不使用TPUSH/TPOP的场景**:
- 同一核内部的数据搬运使用TMOV
- 纯Vector计算或纯Cube计算不需要核间同步

## 参考资料

- **ISA参考**: `pto-isa/docs/PTOISA_zh.md` - PTO指令索引
- **ISA详细文档**: `pto-isa/docs/isa/` - 各指令详细说明
  - `docs/isa/TMATMUL_zh.md` - 矩阵乘法指令
  - `docs/isa/TLOAD_zh.md` - 数据加载指令
  - `docs/isa/TMOV_zh.md` - 数据搬运指令
- **C++ API**: `include/pto/pto-inst.hpp` - PTO指令C++接口
- **常量定义**: `include/pto/common/constants.hpp` - 流水线、事件ID等常量
- **测试示例**: `tests/npu/a2a3/src/st/testcase/` - 算子实现示例
- **融合算子指南**: `vector-fusion-operator-generate` skill - 融合算子开发完整流程

## 总结

本skill提供了使用PTO-ISA实现指定算子功能的完整流程：

1. **步骤1**: 阅读PTOISA_zh.md，了解指令集
2. **步骤2**: 分析算子需求，列举ISA指令
3. **步骤3**: 按数据流顺序解释指令功能
   - Vector计算: GM → UB → V → UB → GM
   - Cube计算: GM → L1 → L0A/L0B → L0C → GM
4. **步骤4**: 输出完整kernel代码

通过遵循本指南，开发者可以系统性地选择ISA指令、理解两种数据流模式(Vector和Cube)、生成高质量kernel代码。

**关键要点**:
- **Vector计算**: 使用UB缓冲区，适用于逐元素操作，流水线 MTE2 → V → MTE3
- **Cube计算**: 使用L1和L0缓冲区，适用于矩阵乘法，流水线 MTE2 → MTE1 → M → FIX → MTE3
- **TMOV关键作用**: 在矩阵乘法中将L1数据搬运到L0，准备Cube计算