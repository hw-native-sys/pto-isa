---
name: PTO融合算子开发指南
description: 本指南为在PTO Tile Library中开发融合算子提供全面指导，涵盖Host-Device架构、数据流向(gm→ub→vector→ub→gm)、指令调用、融合模式、同步策略、内存管理、类型处理等核心概念和最佳实践
license: CANN Open Software License Agreement Version 2.0
---

# 融合算子开发指南

本指南为在 PTO Tile Library 中开发融合算子提供全面指导。

## 目录
1. [概述](#概述)
2. [文件结构](#文件结构)
3. [核心概念](#核心概念)
4. [融合模式](#融合模式)
5. [同步策略](#同步策略)
6. [内存管理](#内存管理)
7. [类型处理](#类型处理)
8. [最佳实践](#最佳实践)
9. [示例](#示例)

## 概述

融合算子将多个 PTO 指令组合成单个操作，以实现：
- 减少内存传输
- 提高流水线效率
- 最小化延迟
- 优化硬件资源利用率

### Host-Device 架构

PTO 融合算子采用 Host-Device 分离架构：
- **Host 侧**：负责调用 Device 函数，通过 `<<<1, nullptr, stream>>>` 启动
- **Device 侧**：使用 `__global__ AICORE` 标注，同时定义于 host 和 device 侧

### 数据流向

数据流向遵循 **gm → ub → vector → ub → gm** 模式：
- **gm (Global Memory)**：全局内存，通过 GlobalTensor 描述
- **ub (Unified Buffer)**：片上缓冲区，通过 Tile 描述
- **vector**：向量计算单元

### 指令调用

- **TLOAD**：gm → ub 数据搬运
- **TSTORE**：ub → gm 数据搬运
- **向量指令**：ub 上的计算操作
- **Event 同步**：指令间必须使用 event 进行数据搬运等待

常见的融合模式包括：
- **TAXPY**: TLOAD + TADD + TSTORE
- **TROWEXPANDADD**: TLOAD + TROWEXPANDADD + TSTORE
- **TCOLEXPANDADD**: TLOAD + TCOLEXPANDADD + TSTORE
- **TPARTADD**: TLOAD + TPARTADD + TSTORE
- **TMATMUL**: TLOAD + TMATMUL + TSTORE

### 近期 PR 复盘驱动的质量门

开发或生成融合算子时，先把以下检查写入实现和测试，而不是在评审后补救：

- **同步与跨核 FIFO**：PR #61、#77、#85、#93、#95、#100 反复暴露 `TPUSH`/`TPOP`/`TFREE`、hook 注入、`subblock_dim`、split lane 和 fine-grained sync 的组合风险。涉及这些路径时，至少覆盖 C2V/V2C、split/no-split、`subblock_dim=1/2`、hook/no-hook，以及目标后端（CPU-SIM、A2/A3、A5）。
- **地址、stride、layout**：PR #86 和 #95 的评审指出过 ColMajor DN stride、golden 数据布局、`entryOffset` 作用位置、split-N subblock 偏移问题。写 TLOAD/TSTORE 或 partial tile 搬运前，先明确 base offset、row offset、column offset、entry offset、split-lane offset 的公式，并用能区分 RowMajor/ColMajor/flattened rows/multi-column 的 golden case 验证。
- **模板契约**：PR #92 要求用命名常量和 `static_assert` 保护 constexpr tile-type dispatch。新增模板或重载时，必须明确 tile 类型、layout、dtype、backend-only 资源的约束，不要把 magic number 留在调度逻辑中。
- **测试闭环**：PR #85 暴露过 CMake 注册了不存在 testcase 目录的问题。新增 ST case 时检查：目录存在、局部 `CMakeLists.txt` 存在、父级 CMake 已注册、gtest 名称和 `gen_data.py` 输出一致、golden 数据能暴露目标 bug。
- **性能声明**：PR #100 的 chunked `TPUSH` benchmark 说明同步次数会直接影响性能。任何“更快”的融合路径都要给出 backend、shape、命令和 before/after 数字；热循环要分别审查 contiguous、strided、split、fallback 路径。

## 文件结构

### 文件命名规范

Kernel 文件命名遵循以下规范：

```
t<vec操作指令>_kernel.cpp
```

**命名规则**：
- 以 `t` 开头（表示 Tile 操作）
- 紧跟向量操作指令名称（大写）
- 以 `_kernel.cpp` 结尾

**示例**：
- `tadd_kernel.cpp`: 加法操作 kernel
- `tsub_kernel.cpp`: 减法操作 kernel
- `tmul_kernel.cpp`: 乘法操作 kernel
- `tdiv_kernel.cpp`: 除法操作 kernel
- `taddsub_kernel.cpp`: 加减融合操作 kernel
- `taddmuldiv_kernel.cpp`: 加乘除融合操作 kernel

### Kernel 文件结构

```cpp
/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to License for details. You may not use this file except in compliance with License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

namespace FusedOperatorName {

// Device 函数：使用 __global__ AICORE 标注，同时定义于 host 和 device 侧
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runFusedOp(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    // Device 侧实现：gm → ub → vector → ub → gm
}

// Host 函数：负责调用 Device 函数
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void launchFusedOp(T *out, T *src0, T *src1, void *stream)
{
    // Host 侧通过 <<<1, nullptr, stream>>> 启动 Device 函数
    runFusedOp<T, kTRows_, kTCTCols_, vRows, vCols><<1, nullptr, stream>>>(out, src0, src1);
}

// 模板实例化
template void launchFusedOp<float, 64, 64, 64, 64>(float *out, float *src0, float *src1, void *stream);

} // namespace FusedOperatorName
```

**关键点**：
- Device 函数使用 `__global__ AICORE` 标注
- Host 函数通过 `<<<1, nullptr, stream>>>` 启动 Device 函数
- `<<<1, nullptr, stream>>>` 是固定的 host→device 调用样式
- 第一个参数 `1` 表示 block 数量
- 第二个参数 `nullptr` 表示共享内存大小
- 第三个参数 `stream` 表示异步流

### 项目文件组织

所有测试用例文件都放在 `tests/npu/a2a3/src/st/testcase/` 目录下的同一文件夹中：

```
tests/npu/a2a3/src/st/testcase/
├── tadd/                          # 测试用例文件夹
│   ├── main.cpp                   # NPU 测试主文件
│   ├── gen_data.py                # Golden 数据生成脚本
│   └── testcases/                 # Golden 数据目录
│       ├── TADDTest.case_float_64x64_64x64/
│       │   ├── input1.bin
│       │   ├── input2.bin
│       │   └── golden.bin
│       └── ...
├── taddsub/                       # 加减融合测试用例
│   ├── main.cpp
│   ├── gen_data.py
│   └── testcases/
└── taddsubmuldiv/                # 加减乘除融合测试用例
    ├── main.cpp
    ├── gen_data.py
    └── testcases/
```

**文件组织说明**：
- **kernel.cpp**: 放在 `kernels/custom/` 目录下
- **main.cpp**: 放在 `tests/npu/a2a3/src/st/testcase/<testcase>/` 目录下
- **gen_data.py**: 放在 `tests/npu/a2a3/src/st/testcase/<testcase>/` 目录下
- **testcases/**: 放在 `tests/npu/a2a3/src/st/testcase/<testcase>/` 目录下

**优势**：
- 测试用例相关文件集中管理
- 便于查找和维护
- 符合项目组织规范

## 核心概念

### GlobalTensor 和 Tile 声明

GlobalTensor 用于描述 gm 上的数据布局，Tile 用于描述 ub 上的数据布局：

```cpp
// GlobalTensor：描述全局内存上的数据，使用 Shape 和 Stride 指定布局
using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
using DynStridDim5 = Stride<vRows * vCols, vRows * vCols, vRows * vCols, vCols, 1>;
using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;

// Tile：描述片上缓冲区，TileType::Vec 表示向量操作
using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
```

**关键点**：
- GlobalTensor 必须指定 Shape 和 Stride 来描述数据在 gm 上的排布
- Tile 使用 TileType::Vec 表示向量类型数据
- Tile 地址分配为紧密排布，通过 TASSIGN 指定偏移量

### Tile 对齐要求

Tile 的 BLayout 决定了对齐要求：
- **RowMajor**：cols 必须是 32byte 对齐
- **ColMajor**：rows 必须是 32byte 对齐

**对齐计算公式**：
```cpp
// RowMajor：cols 对齐
constexpr int alignedCols = ((cols * sizeof(T) + 31) / 32) * (32 / sizeof(T));

// ColMajor：rows 对齐
constexpr int alignedRows = ((rows * sizeof(T) + 31) / 32) * (32 / sizeof(T));
```

**示例**：
```cpp
// RowMajor：cols 必须对齐
constexpr int kTCols_ = ((64 * sizeof(float) + 31) / 32) * (32 / sizeof(float));  // 64
using TileData = Tile<TileType::Vec, float, 64, kTCols_, BLayout::RowMajor, -1, -1>;

// ColMajor：rows 必须对齐
constexpr int kTRows_ = ((16 * sizeof(half) + 31) / 32) * (32 / sizeof(half));  // 16
using TileData = Tile<TileType::Vec, half, kTRows_, 256, BLayout::ColMajor, -1, -1>;
```

### 使用 TASSIGN 分配缓冲区

Tile 地址分配为紧密排布，通过 TASSIGN 指定偏移量：

```cpp
TileData src0Tile(vRows, vCols);
TileData src1Tile(vRows, vCols);
TileData dstTile(vRows, vCols);

// 紧密排布：每个 tile 使用不同的偏移量
TASSIGN(src0Tile, 0x0);
TASSIGN(src1Tile, sizeof(T) * TileData::Numel);
TASSIGN(dstTile, 2 * sizeof(T) * TileData::Numel);
```

**关键点**：
- 每个 tile 必须有唯一的缓冲区地址
- 使用递增偏移量 sizeof(T) * TileData::Numel
- 缓冲区大小由 tile 维度和数据类型决定
- 紧密排布确保高效利用片上存储空间

### GlobalTensor 初始化

Host 传入的地址为 gm 上分配的空间，需要使用 GlobalTensor 来指定其分配的 shape 和 stride：

```cpp
// host 传入的指针是 gm 上的地址
// GlobalTensor 通过 shape 和 stride 描述该地址上的数据布局
GlobalData src0Global(src0);
GlobalData src1Global(src1);
GlobalData dstGlobal(out);
```

**关键点**：
- Host 传入的指针指向 gm 上的内存空间
- GlobalTensor 封装该指针，并通过 Shape 和 Stride 描述数据布局
- Shape 描述数据的维度大小
- Stride 描述数据在内存中的步长

## 融合模式

### 模式 1：基于事件的融合（推荐）

**使用场景**：具有清晰依赖关系的简单融合

**数据流向**：gm → ub (TLOAD) → vector (TADD) → ub → gm (TSTORE)

```cpp
Event<Op::TLOAD, Op::TADD> event0;
Event<Op::TADD, Op::TSTORE_VEC> event1;

// gm → ub：从全局内存加载到片上缓冲区
TLOAD(src0Tile, src0Global);
event0 = TLOAD(src1Tile, src1Global);

// vector：在片上缓冲区进行计算，等待 TLOAD 完成
event1 = TADD(dstTile, src0Tile, src1Tile, event0);

// ub → gm：从片上缓冲区存储到全局内存，等待 TADD 完成
TSTORE(dstGlobal, dstTile, event1);
```

**关键点**：
- 指令调用之间必须使用 event 进行数据搬运等待
- Event<Producer, Consumer> 定义生产者和消费者的依赖关系
- TLOAD 完成后才能执行 TADD
- TADD 完成后才能执行 TSTORE

**优势**：
- 自动依赖跟踪
- 编译器优化
- 代码更简洁
- 同时支持手动和自动模式

### 模式 2：手动标志同步

**使用场景**：具有自定义流水线控制的复杂融合

**数据流向**：gm → ub (TLOAD) → vector (TADD) → ub → gm (TSTORE)

```cpp
// gm → ub：从全局内存加载到片上缓冲区
TLOAD(src0Tile, src0Global);
TLOAD(src1Tile, src1Global);

// 等待 TLOAD 完成，然后执行 TADD
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

// vector：在片上缓冲区进行计算
TADD(dstTile, src0Tile, src1Tile);

// 等待 TADD 完成，然后执行 TSTORE
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

// ub → gm：从片上缓冲区存储到全局内存
TSTORE(dstGlobal, dstTile);
```

**关键点**：
- 指令调用之间必须使用 event 进行数据搬运等待
- set_flag/ wait_flag 实现流水线间的同步
- PIPE_MTE2/PIPE_MTE3：内存传输引擎
- PIPE_V：向量计算单元

**何时使用**：
- 需要对流水线阶段进行进行细粒度控制
- 复杂的多阶段融合
- 需要手动优化的性能关键代码

### 模式 3：标量操作融合

**示例：TAXPY (Y = a*X + Y)**

**数据流向**：gm → ub (TLOAD) → vector (TAXPY) → ub → gm (TSTORE)

```cpp
Event<Op::TLOAD, Op::TAXPY> event0;
Event<Op::TAXPY, Op::TSTORE_VEC> event1;

// gm → ub：加载 X 和 Y 到片上缓冲区
TLOAD(src0Tile, src0Global);
event0 = TLOAD(dstTile, dstGlobal);

// vector：原地操作 Y = a*X + Y，等待 TLOAD 完成
event1 = TAXPY(dstTile, src0Tile, (T)scalar, event0);

// ub → gm：存储结果到全局内存，等待 TAXPY 完成
TSTORE(dstGlobal, dstTile, event1);
```

**关键点**：
- 标量参数必须转换为 tile 数据类型
- 输出 tile 可以重用为输入（原地操作）
- 指令调用之间必须使用 event 进行数据搬运等待

### 模式 4：广播操作融合

**示例：TROWEXPANDADD**

**数据流向**：gm → ub (TLOAD) → vector (TROWEXPANDADD) → ub → gm (TSTORE)

```cpp
// 广播数据使用不同的形状/布局
using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, src1Row, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;
using TileDataSrc1 = Tile<TileType::Vec, T, src1Row, 1, BLayout::ColMajor, -1, -1>;

TileDataSrc1 src1Tile(validRow, 1);

// gm → ub：加载主数据和广播数据到片上缓冲区
TLOAD(src0Tile, src0Global);
TLOAD(src1Tile, src1Global);

// 等待 TLOAD 完成，然后执行 TROWEXPANDADD
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

// vector：行广播加法，将 src1 沿行广播加到 src0
TROWEXPANDADD(dstTile, src0Tile, src1Tile);

// 等待 TROWEXPANDADD 完成，然后执行 TSTORE
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

// ub → gm：存储结果到全局内存
TSTORE(dstGlobal, dstTile);
```

**关键点**：
- 广播数据使用不同的形状/布局
- 沿行广播使用 ColumnMajor 布局
- 全局张量使用 DN（Dense）布局
- 指令调用之间必须使用 event 进行数据搬运等待

### 模式 5：矩阵操作融合

**示例：TMATMUL**

**数据流向**：gm → ub (TLOAD) → mat (TMOV) → matrix (TMATMUL) → ub → gm (TSTORE)

```cpp
// 矩阵操作使用 TileType::Mat
using TileMatAData = Tile<TileType::Mat, U, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;

// 矩阵乘法特化类型
using LeftTile = TileLeft<U, M, K, validM, validK>;
using RightTile = TileRight<S, K, N, validK, validN>;
using AccTile = TileAcc<T, M, N, validM, validN>;

TileMatAData aMatTile;
TileMatBData bMatTile;
LeftTile aTile;
RightTile bTile;
AccTile cTile;

// gm → ub：加载矩阵 A 和 B 到片上缓冲区
TLOAD(aMatTile, src0Global);
TLOAD(bMatTile, src1Global);

// 等待 TLOAD 完成，然后执行 TMOV
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

// mat：将 Mat tile 转换为操作 tile
TMOV(aTile, aMatTile);
TMOV(bTile, bMatTile);

// 等待 TMOV 完成，然后执行 TMATMUL
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

// matrix：矩阵乘法
TMATMUL(cTile, aTile, bTile);

// 等待 TMATMUL 完成，然后执行 TSTORE
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

// ub → gm：存储结果到全局内存
TSTORE(dstGlobal, cTile);
```

**关键点**：
- 矩阵操作使用 TileType::Mat
- 需要 TileLeft、TileRight、TileAcc 特化
- 多个流水线阶段（MTE2 → MTE1 → M → FIX）
- TMOV 将 Mat tile 转换为操作 tile
- 指令调用之间必须使用 event 进行数据搬运等待

## 同步策略

### 基于事件的同步

```cpp
Event<Op::Producer, Op::Consumer> event;

event = PRODUCER(outputTile, inputTile);
CONSUMER(nextTile, outputTile, event);
```

**流水线阶段**：
- `Op::TLOAD`: 从全局内存加载
- `Op::TSTORE_VEC`: 存储到全局内存
- `Op::TADD`、`Op::TMUL` 等：向量操作
- `Op::TMATMUL`: 矩阵乘法

### 手动标志同步

```cpp
set_flag(srcPipe, dstPipe, eventId);
wait_flag(srcPipe, dstPipe, eventId);
```

**流水线线类型**：
- `PIPE_MTE1`、`PIPE_MTE2`、`PIPE_MTE3`: 内存传输引擎
- `PIPE_V`: 向量单元
- `PIPE_M`: 矩阵单元
- `PIPE_FIX`: 格式转换单元
- `PIPE_S`: 标量单元
- `PIPE_ALL`: 所有流水线

**事件 ID**：
- `EVENT_ID0` 到 `EVENT_ID7`: 可用的事件标识符

### 屏障同步

```cpp
pipe_barrier(PIPE_ALL);
```

**使用场景**：所有流水线阶段的全局同步

## 内存管理

### 缓冲区分配策略

```cpp
size_t size = Row * Col * sizeof(T);
TASSIGN(src0Tile, 0x0);
TASSIGN(src1Tile, size);
TASSIGN(dstTile, size * 2);
```

**最佳实践**：
- 计算实际大小（字节）
- 对多个 tile 使用累积偏移量
- 确保缓冲区不重叠

### 对齐考虑

```cpp
constexpr uint16_t alignedRows = ((validRows * sizeof(T) + 31) / 32) * (32 / sizeof(T));
```

**为什么需要对齐**：
- 硬件要求 32 字节对齐
- 防止性能下降
- 避免硬件错误

### 临时缓冲区

```cpp
TileDataTmp tmpTile(validRow, validCol);
TASSIGN(tmpTile, size + size1);
```

**使用场景**：
- 中间结果
- 复杂操作的临时空间
- 对齐填充

## 类型处理

### aclFloat16 到 half 转换

```cpp
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void launchFusedOp(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runFusedOp<half, kTRows_, kTCols_, vRows, vCols>
            <<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1);
    } else {
        runFusedOp<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
    }
}
```

**为什么使用此模式**：
- aclFloat16 是 API 类型
- half 是硬件类型
- NPU 执行需要转换

### 混合类型操作

```cpp
template <typename T, typename U, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runMixedOp(__gm__ T __out__ *out, __gm__ U __in__ *src0, float scalar)
{
    using SrcGlobalData = GlobalTensor<U, DynShapeDim5, DynStridDim5>;
    using SrcTileData = Tile<TileType::Vec, U, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    SrcTileData src0Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);

    event1 = TAXPY(dstTile, src0Tile, (U)scalar, event0);
}
```

**使用场景**：
- 操作期间类型转换
- 输入和输出使用不同精度
- 标量广播

## 最佳实践

### 1. 始终使用命名空间

```cpp
namespace FusedOperatorName {
}
```

**优势**：
- 避免符号冲突
- 更好的代码组织
- 更清晰的意图

### 2. 优先使用基于事件的融合

```cpp
Event<Op::TLOAD, Op::TADD> event0;
event0 = TLOAD(src1Tile, src1Global);
TADD(dstTile, src0Tile, src1Tile, event0);
```

**优势**：
- 自动依赖管理
- 更好的编译器优化
- 代码更简洁

### 3. 为手动模式使用条件编译

```cpp
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
```

**优势**：
- 两种模式使用单一代码库
- 编译器为每种模式优化
- 更易于维护

### 4. 验证 Tile 维度

```cpp
PTO_STATIC_ASSERT(kTRows_ <= 256, "Tile rows must not exceed 256");
PTO_STATIC_ASSERT(kTCols_ <= 256, "Tile cols must not exceed 256");
```

### 5. 记录缓冲区布局

```cpp
size_t size = Row * Col * sizeof(T);
TASSIGN(src0Tile, 0x0);           // Buffer 0: 0 to size
TASSIGN(src1Tile, size);           // Buffer 1: size to 2*size
TASSIGN(dstTile, size * 2);         // Buffer 2: 2*size to 3*size
```

### 6. 处理原地操作

```cpp
event0 = TLOAD(dstTile, dstGlobal);
event1 = TAXPY(dstTile, src0Tile, (T)scalar, event0);
TSTORE(dstGlobal, dstTile, event1);
```

**关键点**：
- 首先将输出作为输入加载
- 对同一 tile 使用相同角色
- 最小化内存传输

### 7. 为常见情况优化

```cpp
template void launchFusedOp<float, 64, 64, 64, 64>(float *out, float *src0, float *src1, void *stream);
template void launchFusedOp<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);
```

**常见尺寸**：
- float: 64x64, 32x32, 16x16
- aclFloat16: 16x256, 8x768, 4x1024
- int32_t: 64x64, 32x32
- int16_t: 64x64, 32x128

### 8. 使用 constexpr 进行编译时计算

```cpp
constexpr uint16_t alignedRows = ((validRows * sizeof(T) + 31) / 32) * (32 / sizeof(T));
```

**优势**：
- 在编译时计算
- 无运行时开销
- 更好的优化

### 9. 分离启动函数

```cpp
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void launchFusedOp(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runFusedOp<half, kTRows_, kTCols_, vRows, vCols>
            <<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1);
    } else {
        run fusedOp<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
    }
}
```

**优势**：
- 处理类型转换
- 提供清晰的 API
- 分离关注点

### 10. 优先在 CPU 模拟器上测试

始终在 NPU 硬件之前在 CPU 模拟器上测试

```bash
python3 tests/run_cpu.py --testcase <testcase> --gtest_filter '<test>'
```

**优势**：
- 更快的迭代
- 更好的调试
- 尽早发现错误

## 示例

### 示例 1：简单逐元素融合

```cpp
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTAdd(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

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
```

### 示例 2：标量融合 (TAXPY)

```cpp
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTAxpy(__gm__ T __out__ *out, __gm__ T __in__ *src0, float scalar)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, vCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x10000);

    GlobalData src0Global(src0);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TAXPY> event0;
    Event<Op::TAXPY, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(dstTile, dstGlobal);
    event1 = TAXPY(dstTile, src0Tile, (T)scalar, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}
```

### 示例 3：广播融合 (TROWEXPANDADD)

```cpp
template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
__global__ AICORE void runTRowExpandAdd(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    constexpr uint16_t src1Row = ((validRow * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, Row, Col>, Stride<1, 1, 1, Col, 1>>;
    using TileDataDst = Tile<TileType::Vec, T, Row, Col, BLayout::RowMajor, -1, -1>;
    using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, src1Row, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1Row, 1, BLayout::ColMajor, -1, -1>;

    TileDataDst src0Tile(validRow, validCol);
    TileDataSrc1 src1Tile(validRow, 1);
    TileDataDst dstTile(validRow, validCol);
    size_t size = Row * Col * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x0);
    TASSIGN(src1Tile, size);

    GlobalDataDst src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataDst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    if constexpr (src0eqdst) {
        TROWEXPANDADD(dstTile, src0Tile, src1Tile);
    } else {
        TROWEXPANDADD(dstTile, src1Tile, src0Tile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}
```

### 示例 4：矩阵融合 (TMATMUL)

```cpp
template <typename T, typename U, typename S, typename B, int validM, int validK, int validN, bool isBias>
__global__ AICORE void RunTMATMUL(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ B *src2)
{
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
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

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
```

## 测试和验证

### 单元测试

```cpp
TEST(FusedOpTest, BasicTest) {
    const int rows = 64;
    const int cols = 64;
    const int size = rows * cols;

    std::vector<float> src0(size, 1.0f);
    std::vector<float> src1(size, 2.0f);
    std::vector<float> dst(size);

    launchFusedOp<float, 64, 64, 64, 64>(dst.data(), src0.data(), src1.data(), stream);

    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(dst[i], src0[i] + src1[i]);
    }
}
```

### 性能测试

```bash
# CPU 模拟器
python3 tests/run_cpu.py --testcase fusedop --gtest_filter 'FusedOpTest.PerfTest'

# NPU 硬件
python3 tests/script/run_st.py -r npu -v a3 -t fusedop -g FusedOpTest.PerfTest
```

## 常见陷阱

### 1. 缓冲区偏移错误

**问题**：
```cpp
TASSIGN(src0Tile, 0x0);
TASSIGN(src1Tile, 0x0);  // Same offset!
```

**解决方案**：
```cpp
TASSIGN(src0Tile, 0x0);
TASSIGN(src1Tile, 0x10000);
```

### 2. 缺少类型转换

**问题**：
```cpp
TAXPY(dstTile, src0Tile, scalar, event0);  // scalar is float, tile is half
```

**解决方案**：
```cpp
TAXPY(dstTile, src0Tile, (T)scalar, event0);
```

### 3. 事件依赖错误

**问题**：
```cpp
TLOAD(src0Tile, src0Global);
TLOAD(src1Tile, src1Global);
TADD(dstTile, src0Tile, src1Tile);  // Missing event dependency
```

**解决方案**：
```cpp
Event<Op::TLOAD, Op::TADD> event0;
TLOAD(src0Tile, src0Global);
event0 = TLOAD(src1Tile, src1Global);
TADD(dstTile, src0Tile, src1Tile, event0);
```

### 4. 对齐问题

**问题**：
```cpp
using TileData = Tile<TileType::Vec, T, 63, 63, BLayout::RowMajor, -1, -1>;
```

**解决方案**：
```cpp
constexpr uint16_t alignedRows = ((63 * sizeof(T) + 31) / 32) * (32 / sizeof(T));
using TileData = Tile<TileType::Vec, T, alignedRows, alignedCols, BLayout::RowMajor, -1, -1>;
```

### 5. 忘记 aclFloat16 转换

**问题**：
```cpp
template void launchFusedOp<aclFloat16, 64, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);
// No conversion in launch function
```

**解决方案**：
```cpp
if constexpr (std::is_same_v<T, aclFloat16>) {
    runFusedOp<half, kTRows_, kTCols_, vRows, vCols>
        <<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1);
}
```

## 测试文件生成

### main.cpp 结构

NPU 测试的 main.cpp 遵循标准的 Google Test 结构，包含 ACL 初始化、内存管理、kernel 调用和结果验证。

### 基本结构

```cpp
/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to License for details. You may not use this file except in compliance with License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class FusedOpTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchFusedOp(T *out, T *src0, T *src1, void *stream);

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void test_fused_op()
{
    size_t fileSize = kTRows_ * kTCols_ * sizeof(T);

    // ACL 初始化
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 主机和设备内存分配
    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&src0Host), fileSize);
    aclrtMallocHost((void **)(&src1Host), fileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 读取输入数据
    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, fileSize);

    // 数据传输：Host → Device
    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // 调用 kernel，函数名需与kernel中的launch函数保持一致
    // 例如：kernel中定义了launchTAddSub，这里就调用launchTAddSub
    LaunchFusedOp<T, kTRows_, kTCols_, vRows, vCols>(dstDevice, src0Device, src1Device, stream);

    // 同步流
    aclrtSynchronizeStream(stream);

    // 数据传输：Device → Host
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // 写入输出结果
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    // 释放设备内存
    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    // 释放主机内存
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);

    // 销毁流和设备
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // 结果验证
    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(FusedOpTest, case_float_64x64_64x64)
{
    test_fused_op<float, 64, 64, 64, 64>();
}

TEST_F(FusedOpTest, case_half_16x256_16x256)
{
    test_fused_op<aclFloat16, 16, 256, 16, 256>();
}
```

### 关键组件说明

#### 1. GetGoldenDir 函数

获取当前测试用例的 golden 文件路径：

```cpp
std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}
```

**输出路径格式**：`../TestSuiteName.TestCaseName`

#### 2. ACL 初始化和清理

```cpp
// 初始化
aclInit(nullptr);
aclrtSetDevice(0);
aclrtCreateStream(&stream);

// 清理
aclrtDestroyStream(stream);
aclrtResetDevice(0);
aclFinalize();
```

#### 3. 内存管理

**主机内存分配**：
```cpp
aclrtMallocHost((void **)(&srcHost), fileSize);
aclrtFreeHost(srcHost);
```

**设备内存分配**：
```cpp
aclrtMalloc((void **)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtFree(srcDevice);
```

**重要**：所有输入和输出的内存大小必须保持一致
- `fileSize` 必须与 kernel 中的 tile 大小一致
- 例如：64x64 的 tile，fileSize = 64 * 64 * sizeof(T)
- 例如：16x256 的 tile，fileSize = 16 * 256 * sizeof(T)

#### 4. 数据传输

**Host → Device**：
```cpp
aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
```

**Device → Host**：
```cpp
aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
```

#### 5. Kernel 调用

```cpp
// 调用 kernel，函数名需与kernel中的launch函数保持一致
// 例如：kernel中定义了launchTMulDiv，这里就调用launchTMulDiv
// 模板参数必须与kernel中的模板参数完全一致
// 例如：kernel中定义了launchTMulDiv<T, 64, 64, 64, 64>，这里就调用launchTMulDiv<float, 64, 64, 64, 64>
LaunchTMulDiv<T, kTRows_, kTCols_, vRows, vCols>(dstDevice, src0Device, src1Device, stream);
aclrtSynchronizeStream(stream);
```

**重要**：
- **函数名一致性**：调用的 launch 函数名必须与 kernel 中定义的 launch 函数名完全一致
- **模板参数一致性**：模板参数 `<T, kTRows_, kTCols_, vRows, vCols>` 必须与 kernel 中的模板参数完全一致
- **示例**：如果 kernel 中定义了 `launchTAddSub<T, 64, 64, 64, 64>`，那么在 main.cpp 中就调用 `launchTAddSub<float, 64, 64, 64, 64>`

#### 6. 结果验证

```cpp
std::vector<T> golden(fileSize);
std::vector<T> devFinal(fileSize);
ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

bool ret = ResultCmp<T>(golden, devFinal, 0.001f);
EXPECT_TRUE(ret);
```

### Golden 文件结构

每个测试用例需要以下 golden 文件：

```
TestSuiteName.TestCaseName/
├── input1.bin      # 第一个输入数据
├── input2.bin      # 第二个输入数据
├── golden.bin      # 期望的输出结果
└── output.bin      # 实际输出结果（测试运行时生成）
```

### 测试用例命名规范

```cpp
TEST_F(FusedOpTest, case_<type>_<tile_size>_<valid_size>)
```

**示例**：
- `case_float_64x64_64x64`: float 类型，tile 64x64，valid 64x64
- `case_half_16x256_16x256`: half 类型，tile 16x256，valid 16x256

### 多输入处理

对于需要多个输入的融合算子，需要相应增加内存分配和数据传输：

```cpp
// 多个输入
T *dstHost, *src0Host, *src1Host, *src2Host, *src3Host;
T *dstDevice, *src0Device, *src1Device, *src2Device, *src3Device;

// 分配内存
aclrtMallocHost((void **)(&src0Host), fileSize);
aclrtMallocHost((void **)(&src1Host), fileSize);
aclrtMallocHost((void **)(&src2Host), fileSize);
aclrtMallocHost((void **)(&src3Host), fileSize);

aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc((void **)&src2Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc((void **)&src3Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

// 读取输入
ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, fileSize);
ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, fileSize);
ReadFile(GetGoldenDir() + "/input3.bin", fileSize, src2Host, fileSize);
ReadFile(GetGoldenDir() + "/input4.bin", fileSize, src3Host, fileSize);

// 传输数据
aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(src2Device, fileSize, src2Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(src3Device, fileSize, src3Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

// 调用 kernel
LaunchFusedOp<T, ...>(dstDevice, src0Device, src1Device, src2Device, src3Device, stream);
```

### 运行测试

```bash
# NPU 硬件测试
python3 tests/script/run_st.py -r npu -v a3 -t <testcase> -g <test_case_name>

# 示例
python3 tests/script/run_st.py -r npu -v a3 -t fused_arithmetic -g FusedArithmeticTest.case_float_64x64_64x64
```

## Golden 数据生成

### gen_data.py 结构

gen_data.py 用于生成测试用例的输入数据和 golden 结果（期望输出），支持多种数据类型和配置。

### 基本结构

```python
#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np
np.random.seed(19)


def gen_golden_data_fused_op(case_name, param):
    dtype = param.dtype

    h, w = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1 = np.random.randint(1, 10, size=[h, w]).astype(dtype)
    input2 = np.random.randint(1, 10, size=[h, w]).astype(dtype)
    input3 = np.random.randint(1, 10, size=[h, w]).astype(dtype)

    # Perform fused operation
    golden = (input1 + input2) - input3

    # Apply valid region constraints
    output = np.zeros([h, w]).astype(dtype)
    for h in range(h):
        for w in range(w):
            if h >= h_valid or w >= w_valid:
                golden[h][w] = output[h][w]

    # Save to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    input3.tofile("input3.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, input3, golden


class FusedOpParams:
    def __init__(self, dtype, gm_row, gm_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.gm_row = gm_row
        self.gm_col = gm_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16'
    }[param.dtype]
    return f"FusedOpTest.case_{dtype_str}_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}"


if __name__ == "__main__":
    # Get absolute path of script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        FusedOpParams(np.float32, 64, 64, 64, 64, 64, 64),
        FusedOpParams(np.int32, 64, 64, 64, 64, 64, 64),
        FusedOpParams(np.int16, 64, 64, 64, 64, 64, 64),
        FusedOpParams(np.float16, 16, 256, 16, 256, 16, 256),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_fused_op(case_name, param)
        os.chdir(original_dir)
```

### 关键组件说明

#### 1. 导入和随机种子

```python
import os
import numpy as np
np.random.seed(19)  # 固定随机种子，确保结果可复现
```

**关键点**：
- 使用 numpy 进行数组操作
- 固定随机种子确保测试结果可复现

#### 2. 参数类定义

```python
class FusedOpParams:
    def __init__(self, dtype, gm_row, gm_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.gm_row = gm_row
        self.gm_col = gm_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
```

**参数说明**：
- `dtype`: 数据类型（np.float32, np.int32, np.float16 等）
- `gm_row`, `gm_col`: 全局内存维度
- `tile_row`, `tile_col`: Tile 维度
- `valid_row`, `valid_col`: 有效数据区域维度

#### 3. Golden 数据生成函数

```python
def gen_golden_data_fused_op(case_name, param):
    dtype = param.dtype

    h, w = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1 = np.random.randint(1, 10, size=[h, w]).astype(dtype)
    input2 = np.random.randint(1, 10, size=[h, w]).astype(dtype)
    input3 = np.random.randint(1, 10, size=[h, w]).astype(dtype)

    # Perform fused operation
    golden = (input1 + input2) - input3

    # Apply valid region constraints
    output = np.zeros([h, w]).astype(dtype)
    for h in range(h):
        for w in range(w):
            if h >= h_valid or w >= w_valid:
                golden[h][w] = output[h][w]

    # Save to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    input3.tofile("input3.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, input3, golden
```

**关键点**：
- 使用 `np.random.randint` 生成随机数据
- 使用 `.astype(dtype)` 转换数据类型
- 使用 `.tofile()` 保存为二进制文件
- 处理 valid 区域外的数据

#### 4. 测试用例名称生成

```python
def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16'
    }[param.dtype]
    return f"FusedOpTest.case_{dtype_str}_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}"
```

**输出格式**：`FusedOpTest.case_<type>_<tile_row>x<tile_col>_<valid_row>x<valid_col>`

**示例**：
- `FusedOpTest.case_float_64x64_64x64`
- `FusedOpTest.case_half_16x256_16x256`

#### 5. 主函数逻辑

```python
if __name__ == "__main__":
    # Get absolute path of script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        FusedOpParams(np.float32, 64, 64, 64, 64, 64, 64),
        FusedOpParams(np.int32, 64, 64, 64, 64, 64, 64),
        FusedOpParams(np.int16, 64, 64, 64, 64, 64, 64),
        FusedOpParams(np.float16, 16, 256, 16, 256, 16, 256),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_fused_op(case_name, param)
        os.chdir(original_dir)
```

**关键点**：
- 获取脚本所在目录
- 创建 testcases 目录
- 遍历参数列表，为每个参数生成测试用例
- 切换到测试用例目录生成数据

#### 6. 数据类型映射

```python
dtype_str = {
    np.float32: 'float',
    np.float16: 'half',
    np.int8: 'int8',
    np.int32: 'int32',
    np.int16: 'int16'
}[param.dtype]
```

### 多输入处理

对于需要多个输入的融合算子，需要相应增加输入数据生成：

```python
def gen_golden_data_fused_op(case_name, param):
    dtype = param.dtype

    h, w = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate multiple input arrays
    input1 = np.random.randint(1, 10, size=[h, w]).astype(dtype)
    input2 = np.random.randint(1, 10, size=[h, w]).astype(dtype)
    input3 = np.random.randint(1, 10, size=[h, w]).astype(dtype)
    input4 = np.random.randint(1, 10, size=[h, w]).astype(dtype)

    # Perform fused operation
    golden = ((input1 + input2) - input3) * input4 / input4

    # Apply valid region constraints
    output = np.zeros([h, w]).astype(dtype)
    for h in range(h):
        for w in range(w):
            if h >= h_valid or w >= w_valid:
                golden[h][w] = output[h][w]

    # Save to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    input3.tofile("input3.bin")
    input4.tofile("input4.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, input3, input4, golden
```

### 运行 gen_data.py

```bash
# 进入测试用例目录
cd tests/npu/a2a3/src/st/testcase/fused_add_sub

# 生成 golden 数据
python3 gen_data.py

# 查看生成的目录结构
ls -la
```

**生成的目录结构**：
```
fused_add_sub/
├── gen_data.py
├── FusedAddSubTest.case_float_64x64_64x64/
│   ├── input1.bin
│   ├── input2.bin
│   ├── input3.bin
│   └── golden.bin
├── FusedAddSubTest.case_int32_64x64_64x64/
│   ├── input1.bin
│   ├── input2.bin
│   ├── input3.bin
│   └── golden.bin
├── FusedAddSubTest.case_int16_64x64_64x64/
│   ├── input1.bin
│   ├── input2.bin
│   ├── input3.bin
│   └── golden.bin
└── FusedAddSubTest.case_half_16x256_16x256/
    ├── input1.bin
    ├── input2.bin
    ├── input3.bin
    └── golden.bin
```

### 常见操作示例

#### 加法操作
```python
golden = input1 + input2
```

#### 减法操作
```python
golden = input1 - input2
```

#### 乘法操作
```python
golden = input1 * input2
```

#### 除法操作
```python
golden = input1 / input2
```

#### 融合操作
```python
golden = (input1 + input2) - input3
golden = ((input1 + input2) - input3) * input4
golden = ((input1 + input2) - input3) * input4 / input4
```

#### 激活函数
```python
golden = np.maximum(0, input1)  # ReLU
golden = 1.0 / (1.0 + np.exp(-input1))  # Sigmoid
```

### 注意事项

1. **随机种子**：始终设置固定的随机种子，确保测试结果可复现
2. **数据范围**：使用合理的数据范围，避免溢出和精度问题
3. `**Valid 区域**：正确处理 valid 区域外的数据
4. **数据类型**：确保使用正确的数据类型转换
5. **文件格式**：使用 `.tofile()` 保存为二进制格式
6. **目录切换**：生成数据后记得切换回原始目录

## 参考

- PTO 指令参考：`include/pto/pto-inst.hpp`
- 常量和类型：`include/pto/common/constants.hpp`
- 测试示例：`tests/npu/a2a3/src/st/testcase/`
- 构建系统：`tests/script/`

## 测试用例文件生成

### CMakeLists.txt 结构

CMakeLists.txt 是测试用例文件夹中的构建配置文件，用于定义和编译测试用例。

### 基本结构

```cmake
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

pto_vec_st(tadd)
```

### 文件组织

在测试用例文件夹中创建以下文件：

```
tests/npu/a2a3/src/st/testcase/tadd/
├── tadd_kernel.cpp           # Kernel 实现
├── main.cpp                   # NPU 测试主文件
├── gen_data.py                # Golden 数据生成脚本
└── testcases/                  # Golden 数据目录
    ├── TADDTest.case_float_64x64_64x64/
    │   ├── input1.bin
    │   ├── input2.bin
    │   └── golden.bin
    └── ...
```

### 文件生成步骤

#### 1. 创建测试用例目录

```bash
# 在 tests/npu/a2a3/src/st/testcase/ 下创建测试用例文件夹
mkdir -p tests/npu/a2a3/src/st/testcase/tadd
```

#### 2. 创建 CMakeLists.txt

```bash
# 在测试用例文件夹中创建 CMakeLists.txt
cat > tests/npu/a2a3/src/st/testcase/tadd/CMakeLists.txt << 'EOF'
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

pto_vec_st(tadd)
EOF
```

#### 3. 创建 kernel.cpp

参考"文件结构"章节中的 Kernel 文件结构，创建 `tadd_kernel.cpp`：

```cpp
/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to License for details. You may not use this is except in compliance with License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

namespace TADD {

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTAdd(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    // Device 侧实现：gm → ub → vector → ub → gm
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

}  // namespace TADD
```

#### 4. 创建 main.cpp

参考"测试文件生成"章节中的 main.cpp 结构，创建 `main.cpp`：

```cpp
/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to License for details. You may not use this file except in compliance with License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TADDTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTAdd(T *out, T *src0, T *src1, void *stream);

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void test_tadd()
{
    size_t fileSize = kTRows_ * kTCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&src0Host), fileSize);
    aclrtMallocHost((void **)(&src1Host), fileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, fileSize);

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTAdd<T, kTRows_, kTCols_, vRows, vCols>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TADDTest, case_float_64x64_64x64)
{
    test_tadd<float, 64, 64, 64, 64>();
}

TEST_F(TADDTest, case_int32_64x64_64x64)
{
    test_tadd<int32_t, 64, 64, 64, 64>();
}

TEST_F(TADDTest, case_int16_64x64_64x64)
{
    test_tadd<int16_t, 64, 64, 64, 64>();
}

TEST_F(TADDTest, case_half_16x256_16x256)
{
    test_tadd<aclFloat16, 16, 256, 16, 256>();
}
```

#### 5. 创建 gen_data.py

参考"Golden 数据生成"章节中的 gen_data.py 结构，创建 `gen_data.py`：

```python
#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np
np.random.seed(19)


def gen_golden_data_tadd(case_name, param):
    dtype = param.dtype

    h, w = [param.tile_row, param.tile_col]
       h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1 = np.random.randint(1, 10, size=[h, w]).astype(dtype)
    input2 = np.random.randint(1, 10, size=[h, w]).astype(dtype)

    # Perform addbtraction
    golden = input1 + input2

    # Apply valid region constraints
    output = np.zeros([h, w]).astype(dtype)
    for h in range(h):
        for w in range(w):
            if h >= h_valid or w >= w_valid:
                golden[h][w] = output[h][w]

    # Save to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, golden


class TAddParams:
    def __init__(self, dtype, gm_row, gm_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.gm_row = gm_row
        self.gm_col = gm_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16'
    }[param.dtype]
    return f"TADDTest.case_{dtype_str}_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}"


if __name__ == "__main__":
    # Get absolute path of script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TAddParams(np.float32, 64, 64, 64, 64, 64, 64),
        TAddParams(np.int32, 64, 64, 64, 64, 64),
        TAddParams(np.int16, 64, 64, 64, 64, 64),
        TAddParams(np.float16, 16, 256, 16, 256, 16, 256),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tadd(case_name, param)
        os.chdir(original_dir)
```

#### 6. 生成 Golden 数据

```bash
# 进入测试用例目录
cd tests/npu/a2a3/src/st/testcase/tadd

# 生成 golden 数据
python3 gen_data.py

# 查看生成的目录结构
ls -la
```

#### 7. 构建和测试

```bash
# 返回到项目根目录
cd ../../../../../..

# 构建
cmake --build build

# 运行测试
python3 tests/script/run_st.py -r npu -v a3 -t tadd
```

### 注意事项

1. **文件命名**：kernel.cpp 文件命名遵循 `t<操作指令>_kernel.cpp` 规范
`- `tadd_kernel.cpp`: 加法操作 kernel
- `taddsub_kernel.cpp`: 加减融合操作 kernel
- `taddsubmuldiv_kernel.cpp`: 加减乘除融合操作 kernel

2. **目录结构**：所有相关文件都放在同一个测试用例文件夹中
- `tests/npu/a2a3/src/st/testcase/<testcase>/`
- kernel.cpp、main.cpp、gen_data.py 都在测试用例文件夹中

3. **CMakeLists.txt**：调用 `pto_vec_st(<testcase>)` 函数
- 这个函数会自动处理 kernel 编译和链接

4. **Golden 数据**：gen_data.py 会生成 testcases 目录和 golden 数据
- 每个测试用例对应一个子目录
- 包含 input1.bin、input2.bin、golden.bin 等文件

5. **测试用例命名**：遵循 `case_<type>_<tile_size>_<valid_size>` 格式
- `case_float_64x64_64x64`: float 类型，tile 64x64，valid 64x64
- `case_half_16x256_16x256`: half 类型，tile 16x256，valid 16x256
