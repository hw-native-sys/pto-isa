---
name: PTO Costmodel Cycles 查询指南
description: 本指南介绍如何使用 PTO Costmodel 获取单条 PTO ISA 指令的仿真 cycles 数，涵盖 costmodel 两种使用场景、查询脚本用法、各指令类型的 C++ 模板以及编译运行方法
license: CANN Open Software License Agreement Version 2.0
---

# PTO Costmodel Cycles 查询指南

本指南介绍如何使用 PTO Costmodel 获取 PTO ISA 指令的仿真 cycles。

## 目录
1. [概述](#概述)
2. [查询前的输入信息确认](#查询前的输入信息确认)
3. [查询单条指令 Cycles](#查询单条指令-cycles)
4. [脚本不存在时的生成方法](#脚本不存在时的生成方法)
5. [架构参数](#架构参数)
6. [ST 测试套件](#st-测试套件)

## 概述

PTO costmodel 是 PTO ISA 的性能仿真模型，模拟 Ascend NPU (A2/A3, `__NPU_ARCH__=2201`) 上的指令执行耗时，纯 CPU 运行，不需要真实硬件。

核心原理：当编译时定义了 `__COSTMODEL` 宏后，`#include <pto/pto-inst.hpp>` 会自动将所有 PTO 指令替换为 costmodel 版本，指令执行时会通过 trace 系统（`BeginPtoInstr`/`EndPtoInstr` + `RecordCceCall`）记录周期数，用户可通过 API 查询。

**当前 costmodel 的能力边界：**
- **已支持**：单条 PTO ISA 指令级别的 cycles 查询（如 TADD、TMATMUL 等单条指令在指定 shape 和 dtype 下的仿真耗时）
- **未支持**：算子级（多个 PTO 指令组合）的性能仿真，该能力待未来开发

> **重要：当前 costmodel 仅支持 A2/A3 平台（`__NPU_ARCH__=2201`），仿真结果仅反映该平台的性能特征。** 其他平台的结果可能不同。

---

## 查询前的输入信息确认

在执行查询之前，**必须确认用户已提供以下信息**。如果缺少任何一项，请主动向用户询问：

### 通用必需信息

| 信息 | 说明 | 示例 |
|------|------|------|
| **指令名称** | 要查询的 PTO 指令 | `TADD`, `TMUL`, `TEXP`, `TMATMUL` 等 |
| **数据类型** | Tile 的元素类型 | `float`, `half`, `bf16`, `int32`, `int16`, `int8`, `uint8` |
| **Tile Shape** | Tile 的行数和列数 | `16x16`, `64x64`, `16x256` 等 |

### 特定指令的额外信息

| 指令类型 | 额外必需信息 | 示例 |
|----------|-------------|------|
| **TCVT** | 源类型 + 目标类型（两种类型不同） | 源=`float`, 目标=`half` |
| **TMATMUL** | A/B/Output 三种类型 + M/K/N 三个维度 | A=`half`, B=`half`, Out=`float`, M=128, K=64, N=128 |

### 缺少信息时的提示话术

- 缺少指令名称：`"请问您想查询哪条 PTO 指令的 cycles？（如 TADD、TMUL、TMATMUL 等）"`
- 缺少数据类型：`"请问数据类型是什么？支持 float/half/bf16/int32/int16/int8/uint8"`
- 缺少 shape：`"请问 Tile 的 shape 是多少？（如 16x16, 64x64, 32x256）"`
- TMATMUL 缺少维度：`"TMATMUL 需要 M、K、N 三个维度，请提供具体数值（如 M=128, K=64, N=128）"`
- TCVT 缺少类型：`"TCVT 需要指定源类型和目标类型（如 float 转 half）"`

---

## 查询单条指令 Cycles

### 使用查询脚本（推荐）

脚本路径：`tools/query_pto_cycles.py`

脚本会自动生成最小的 C++ 文件，编译运行，打印 cycles，无残留文件。

#### 用法

```bash
# Unary 指令 (TEXP, TNEG, TRECIP, TRSQRT, TSQRT, TABS)
python3 tools/query_pto_cycles.py <指令> <数据类型> -r <行> -c <列>

# Binary 指令 (TADD, TSUB, TMUL, TDIV, TMAX, TMIN, TADDS, TSUBS, TMULS, TDIVS, TMAXS, TMINS, TSEL, TSELS, TCMP)
python3 tools/query_pto_cycles.py <指令> <数据类型> -r <行> -c <列>

# Row Reduce 指令 (TROWSUM, TROWMAX, TROWMIN)
python3 tools/query_pto_cycles.py <指令> <数据类型> -r <行> -c <列>

# Col Reduce 指令 (TCOLSUM, TCOLMAX, TCOLMIN)
python3 tools/query_pto_cycles.py <指令> <数据类型> -r <行> -c <列>

# 类型转换 (TCVT): 第一个参数=目标类型, 第二个=源类型
python3 tools/query_pto_cycles.py TCVT <目标类型> <源类型> -r <行> -c <列>

# 矩阵乘法 (TMATMUL): dtype1=A矩阵类型, dtype2=B矩阵类型, dtype3=输出类型
python3 tools/query_pto_cycles.py TMATMUL <A类型> <B类型> <输出类型> --m <M> --k <K> --n <N>
```

#### 示例

```bash
python3 tools/query_pto_cycles.py TEXP float -r 16 -c 16
# => [CYCLE] TEXP.float_16x16 actual=53

python3 tools/query_pto_cycles.py TADD half -r 64 -c 64
# => [CYCLE] TADD.half_64x64 actual=98

python3 tools/query_pto_cycles.py TROWSUM float -r 64 -c 64
# => [CYCLE] TROWSUM.float_64x64 actual=51

python3 tools/query_pto_cycles.py TCOLSUM half -r 16 -c 256
# => [CYCLE] TCOLSUM.half_16x256 actual=500

python3 tools/query_pto_cycles.py TCVT half float -r 16 -c 64
# => [CYCLE] TCVT.half<-float_16x64 actual=66

python3 tools/query_pto_cycles.py TMATMUL half half float --m 128 --k 64 --n 128
# => [CYCLE] TMATMUL.float<half,half>_128x64x128 actual=262
```

#### 支持的数据类型

`float` / `fp32`, `half` / `fp16`, `bf16`, `int32`, `int16`, `int8`, `uint8`

#### 支持的指令

| 类别 | 指令 |
|------|------|
| Unary | TEXP, TNEG, TRECIP, TRSQRT, TSQRT, TABS |
| Binary | TADD, TSUB, TMUL, TDIV, TMAX, TMIN, TMAXS, TMINS, TADDS, TSUBS, TMULS, TDIVS, TSEL, TSELS, TCMP |
| Row Reduce | TROWSUM, TROWMAX, TROWMIN |
| Col Reduce | TCOLSUM, TCOLMAX, TCOLMIN |
| Convert | TCVT |
| Matmul | TMATMUL |

## 脚本不存在时的生成方法

脚本路径为项目根目录下 `tools/query_pto_cycles.py`。如果不存在，按以下说明创建。

脚本的核心逻辑：
1. 根据用户输入的指令名和数据类型，从预定义的指令分类（UNARY / BINARY / ROW_REDUCE / COL_REDUCE / CVT / MATMUL）确定指令签名
2. 用 Python 的 textwrap.dedent 生成对应的 C++ 源码，核心模板如下：

### 各类型指令的 C++ 模板

`{dtype}`, `{rows}`, `{cols}`, `{instr}` 为占位符。

**Unary（单输入）：**
```cpp
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/costmodel/trace.hpp>
#include <cstdio>
using namespace pto;
using namespace pto::mocker;
int main() {
    ResetTrace();
    using TileData = Tile<TileType::Vec, {dtype}, {rows}, {cols}, BLayout::RowMajor, -1, -1>;
    TileData srcTile({rows}, {cols});
    TileData dstTile({rows}, {cols});
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x8000);
    {instr}(dstTile, srcTile);  // 如: TEXP(dstTile, srcTile)
    uint64_t cycles = GetLastPtoInstrCycles();
    std::printf("[CYCLE] ... actual=%llu\n", (unsigned long long)cycles);
    return 0;
}
```

**Binary（双输入）：**
```cpp
    // 同上，但创建 src0Tile, src1Tile, dstTile
    {instr}(dstTile, src0Tile, src1Tile);  // 如: TADD(dstTile, src0Tile, src1Tile)
```

**Row Reduce（行规约，需 tmp）：**
```cpp
    // 创建 srcTile, tmpTile, dstTile（均为 rows x cols）
    {instr}(dstTile, srcTile, tmpTile);  // 如: TROWSUM(dstTile, srcTile, tmpTile)
```

**Col Reduce（列规约，dst 为 1xcols）：**
```cpp
    using SrcTile = Tile<TileType::Vec, {dtype}, {rows}, {cols}, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, {dtype}, 1, {cols}, BLayout::RowMajor, -1, -1>;
    // tmpTile shape = (rows/2 ? rows/2 : 1, cols)
    {instr}(dstTile, srcTile, tmpTile, false);  // 如: TCOLSUM(dstTile, srcTile, tmpTile, false)
```

**TCVT（类型转换）：**
```cpp
    using DstTile = Tile<TileType::Vec, {dst_dtype}, {rows}, {cols}, BLayout::RowMajor, -1, -1>;
    using SrcTile = Tile<TileType::Vec, {src_dtype}, {rows}, {cols}, BLayout::RowMajor, -1, -1>;
    TCVT(dstTile, srcTile, RoundMode::CAST_NONE);
```

**TMATMUL（矩阵乘法）：**
```cpp
    constexpr int blockAlign = (sizeof({a_dtype}) == 1) ? 32 : 16;
    constexpr int M = CeilAlign({m}, blockAlign);  // 同理 N, K
    using LeftTile  = TileLeft<{a_dtype}, M, K, {m}, {k}>;
    using RightTile = TileRight<{b_dtype}, K, N, {k}, {n}>;
    using AccTile   = TileAcc<{out_dtype}, M, N, {m}, {n}>;
    TMATMUL(cTile, aTile, bTile);
```

### 编译命令

将生成的 `.cpp` 写入临时目录，用以下命令编译：

```bash
clang++ <src.cpp> -o <exe> -std=c++23 -O2 \
    -I<repo>/include -I<repo>/include/common \
    -D__COSTMODEL -D__NPU_ARCH__=2201 -DPTO_COMM_NOT_SUPPORTED \
    -Wno-macro-redefined -Wno-ignored-attributes
```

macOS 额外需要：`-isystem$(xcrun --show-sdk-path)/usr/include/c++/v1`

运行可执行文件，解析 stdout 中的 `[CYCLE]` 行，输出结果。

## 架构参数

- **目标架构**: A2/A3 Ascend NPU (`__NPU_ARCH__=2201`)
- **主频**: 1.85 GHz
- **编译器要求**: clang++ >= 15 或 g++ >= 13，C++23
- **必须宏**: `-D__COSTMODEL -D__NPU_ARCH__=2201 -DPTO_COMM_NOT_SUPPORTED`
- **头文件路径**: `-I<repo>/include -I<repo>/include/common`

## ST 测试套件

正式的回归测试位于 `tests/costmodel/st/testcase/`，用于 costmodel 的正确性验证。

### 运行方式

```bash
python3 tests/run_costmodel.py --testcase <name> --build-type Release
python3 tests/run_costmodel.py --build-type Release  # 运行全部
```

### 可用测试用例

`tabs`, `tadd`, `tadds`, `tcmp`, `tcolmax`, `tcolmin`, `tcolsum`, `tcvt`, `tdiv`, `tdivs`, `texp`, `textract`, `tgather`, `tload`, `tloadconv`, `tmatmul`, `tmax`, `tmaxs`, `tmin`, `tmins`, `tmov`, `tmrgsort`, `tmul`, `tmuls`, `tneg`, `trecip`, `trowexpand`, `trowmax`, `trowmin`, `trowsum`, `trsqrt`, `tscatter`, `tsel`, `tsels`, `tsort32`, `tsqrt`, `tsub`, `tsubs`, `ttrans`
