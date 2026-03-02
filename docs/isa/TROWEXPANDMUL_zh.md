# TROWEXPANDMUL

## 指令示意图

![TROWEXPANDMUL tile operation](../figures/isa/TROWEXPANDMUL.svg)

## 简介

行广播乘法：将 `src0` 的每一行乘以一个每行标量向量 `src1`。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot \mathrm{src1}_{0,i} $$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = trowexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tcolexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMUL(TileDataDst& dst, TileDataDst& src0, TileDataSrc1& src1, WaitEvents&... events);
```

## 约束

- **实现检查**:
  - `TileDataDst::DType == TileDataSrc0::DType == TileDataSrc1::DType` (compile-time).
  - `TileDataDst::DType`, `TileDataSrc0::DType`, `TileDataSrc1::DType` must be one of: `half`, `float`.
  - Tile 形状/布局约束 (compile-time): `TileDataDst::isRowMajor`.
  - Mode 1: `src1` is expected to provide **one scalar per row** (i.e., its valid shape must cover `R` values).
  - Mode 2: `src1` is expected to provide **32 bytes data per row**.

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, half, 16, 16>;
  using RowVecT = Tile<TileType::Vec, half, 16, 1, BLayout::ColMajor, 1, DYNAMIC, SLayout::NoneBox>;

  TileT src0, dst;
  RowVecT src1(16);
  TROWEXPANDMUL(dst, src0, src1);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, half, 16, 16>;
  using RowVecT = Tile<TileType::Vec, half, 16, 1, BLayout::ColMajor, 1, DYNAMIC, SLayout::NoneBox>;

  TileT src0, dst;
  RowVecT src1(16);
  TASSIGN(src0, 0x1000);
  TASSIGN(dst,  0x2000);
  TASSIGN(src1, 0x3000);
  TROWEXPANDMUL(dst, src0, src1);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tcolexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcolexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = trowexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

