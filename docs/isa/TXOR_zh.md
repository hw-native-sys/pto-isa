# TXOR

## 指令示意图

![TXOR tile operation](../figures/isa/TXOR.svg)

## 简介

两个 Tile 的逐元素按位异或。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \oplus \mathrm{src1}_{i,j} $$

## 汇编语法

同步形式：

```text
%dst = txor %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.txor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.txor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TXOR(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束

- 该操作在 `dst.GetValidRow()` / `dst.GetValidCol()` 上迭代。
- **实现检查 (A5)**:
    - `dst`、`src0` 和 `src1` 的元素类型必须一致。
    - 支持的元素类型为 `uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`。
    - `dst`、`src0` 和 `src1` 必须是行主序。
    - `src0.GetValidRow()/GetValidCol()` 和 `src1.GetValidRow()/GetValidCol()` 必须与 `dst` 一致。
- **实现检查 (A2A3)**:
    - `dst`、`src0`、`src1` 和 `tmp` 的元素类型必须一致。
    - 支持的元素类型为 `uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`。
    - `dst`、`src0`、`src1` 和 `tmp` 必须是行主序。
    - `src0`、`src1` 和 `tmp` 的有效形状必须不小于 `dst`。
    - 在手动模式下，`dst`、`src0`、`src1` 和 `tmp` 的内存区域不得重叠。

## 临时空间

### A2A3

`tmp` **被使用**作为中间暂存存储。A2A3 实现通过分解计算 XOR：`XOR(a,b) = AND(NOT(AND(a,b)), OR(a,b))`，需要 `tmp` 来保存中间结果 `OR(a,b)`。

- `tmp` 必须与 `dst`/`src0`/`src1` 具有相同的元素类型。
- `tmp` 必须是行主序。
- `tmp.GetValidRow() >= dst.GetValidRow()` 且 `tmp.GetValidCol() >= dst.GetValidCol()`。
- 在手动模式下，`tmp` 的内存区域不得与 `dst`、`src0` 或 `src1` 重叠。

### A5

`tmp` 被接口接受但 A5 实现**不使用**。A5 后端直接使用 `vxor` 向量指令，不需要暂存 Tile 存储。`tmp` 仅为了与 A2A3 的 API 兼容性而保留在 C++ 内建接口签名中。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileSrc0 = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileSrc1 = Tile<TileType::Vec, uint32_t, 16, 16>;
  using TileTmp = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileDst dst;
  TileSrc0 src0;
  TileSrc1 src1;
  TileTmp tmp;
  TXOR(dst, src0, src1, tmp);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.txor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.txor %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = txor %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.txor ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
