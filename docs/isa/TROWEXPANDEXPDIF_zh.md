# TROWEXPANDEXPDIF

## 指令示意图

![TROWEXPANDEXPDIF tile operation](../figures/isa/TROWEXPANDEXPDIF.svg)

## 简介

行指数差运算：计算exp(src0 - src1)，其中src1为每行标量。

## 数学语义

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。设 `s_i` 为从 `src1` 中获取的每行标量（每行一个值）。

对于 `0 <= i < R` 和 `0 <= j < C`：

$$
\mathrm{dst}_{i,j} = \exp(\mathrm{src0}_{i,j} - s_i)
$$

## 汇编语法

同步形式：

```text
%dst = trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束

- `TileDataDst::DType == TileDataSrc0::DType == TileDataSrc1::DType`
- `TileDataDst::DType`、`TileDataSrc0::DType`、`TileDataSrc1::DType` 必须是以下之一：`half`、`float`。
- Tile形状/布局约束（编译时）：`TileDataDst::isRowMajor`。
- 模式1：`src1` 预期提供**每行一个标量**（即，其有效形状必须覆盖 `R` 个值）。
- 模式2：`src1` 预期提供**每行32字节数据**。
- 确切的布局/分形约束是目标特定的；参见 `include/pto/npu/*/TRowExpand*.hpp` 下的后端头文件。

### 临时Tile

C++ API提供了显式传入 `TileDataTmp &tmp` 的重载。该重载仅支持**模式1**（ColMajor扩展操作数，每行标量）。内部实现中，`TROWEXPANDEXPDIF` 由 `TROWEXPANDSUB` 后接 `TEXP` 实现，因此tmp Tile用于SUB步骤的广播缓冲区。

- **Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品**：tmp Tile作为 `TROWEXPANDSUB` 步骤的广播缓冲区使用。ColMajor扩展操作数的每行标量值通过 `vbrcb` 指令广播到tmp缓冲区，为每行创建一个32字节块，然后在减法运算中作为扩展操作数使用。`vbrcb` 指令的repeat stride为8个块（256字节），每个repeat处理8行。最小tmp大小计算：
    - **公共参数**：
        - `R = dst.GetValidRow()`，`T = TileDataDst::DType`。
    - 当 `R < 256` 时：
        $$ \text{tmpSize} = \left\lceil\frac{R}{8}\right\rceil \times 256 \text{字节} $$
    - 当 `R >= 256` 时：
        - 操作采用循环方式，每次循环最多30个repeat（240行）。tmp缓冲区在各循环间复用，每次循环需要：
        $$ \text{tmpSize} = 30 \times 256 = 7680 \text{字节} $$
    - 对于任何模式1调用，一个紧凑的形状无关上界为 **8KB**（8192Byte）。
    - 不带 `tmp` 的3参数重载支持模式1和模式2。对于模式1，使用内部8KB缓冲区（`TMP_UB_OFFSET`）。对于模式2，不需要广播缓冲区。
- **Ascend 950PR/Ascend 950DT**：`tmp` Tile被接受但不使用（`[[maybe_unused]]`）。Ascend 950PR/Ascend 950DT硬件通过 `vlds` 指令的广播模式原生支持行广播，因此不需要临时缓冲区。

## 示例

参见 `docs/isa/` 和 `docs/coding/tutorials/` 中的相关示例。

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### PTO汇编形式

```text
%dst = trowexpandexpdif %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.trowexpandexpdif ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
