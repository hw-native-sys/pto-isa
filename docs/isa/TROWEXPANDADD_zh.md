# TROWEXPANDADD

## 指令示意图

![TROWEXPANDADD tile operation](../figures/isa/TROWEXPANDADD.svg)

## 简介

行广播加法：加上一个每行标量向量。

## 数学语义

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_i` be the per-row scalar taken from `src1` (one value per row).

For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + s_i
$$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = trowexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.trowexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents&... events);
```

## 约束

- `TileDataDst::DType == TileDataSrc0::DType == TileDataSrc1::DType`
- `TileDataDst::DType`, `TileDataSrc0::DType`, `TileDataSrc1::DType` must be one of: `half`, `float`.
- Tile 形状/布局约束 (compile-time): `TileDataDst::isRowMajor`.
- Mode 1: `src1` is expected to provide **one scalar per row** (i.e., its valid shape must cover `R` values).
- Mode 2: `src1` is expected to provide **32 bytes data per row**.
- Exact layout/fractal constraints are target-specific; see backend headers under `include/pto/npu/*/TRowExpand*.hpp`.

## 示例

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.trowexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = trowexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

