# TPRELU

## 指令示意图

![TPRELU tile operation](../../../../figures/isa/TPRELU.svg)

## 简介

带逐元素斜率 Tile 的逐元素参数化 ReLU (PReLU)。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = (\mathrm{src0}_{i,j} > 0) ? \mathrm{src0}_{i,j} : (\mathrm{src0}_{i,j} \cdot \mathrm{src1}_{i,j}) $$

## 汇编语法

同步形式：

```text
%dst = tprelu %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tprelu ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TPRELU(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束

- 该操作在 `dst.GetValidRow()` / `dst.GetValidCol()` 上迭代。
- **实现检查 (A2A3)**:
    - `dst`、`src0` 和 `src1` 的元素类型必须一致。支持的类型：`half`、`float`。
    - `tmp` 的元素类型必须是 `uint8_t`（用作比较掩码缓冲区）。
    - 所有 Tile 必须是行主序。
    - `src0` 和 `src1` 的有效形状必须与 `dst` 一致。
    - `tmp.GetValidRow() > dst.GetValidRow()`（tmp 需要额外行用于掩码存储）。
    - 在手动模式下，`src0`、`src1`、`dst` 和 `tmp` 的内存区域不得重叠。
- **实现检查 (A5)**:
    - `dst`、`src0` 和 `src1` 的元素类型必须一致。支持的类型：`half`、`float`。
    - 所有 Tile 必须是行主序。
    - `src0` 和 `src1` 的有效形状必须与 `dst` 一致。

## 临时空间

### A2A3

`tmp` **被使用**作为比较掩码存储。A2A3 实现将 PReLU 分解为：`dst = src0 > 0 ? src0 : src0 * src1`，使用 `TCMPS` 将比较掩码写入 `tmp`，然后使用 `TSEL` 在 `src0` 和 `dst`（= `src0 * src1`）之间进行选择。

- `tmp` 的元素类型必须是 `uint8_t`。
- `tmp.GetValidRow() > dst.GetValidRow()`（有效行之后的额外行区域通过 `TSUBVIEW` 用于 `TSEL` 掩码）。
- `tmp` 必须是行主序。
- 在手动模式下，`tmp` 的内存区域不得与 `dst`、`src0` 或 `src1` 重叠。

### A5

`tmp` 被接口接受但 A5 实现**不使用**。A5 后端直接使用 `vprelu` 向量指令，不需要暂存 Tile 存储。`tmp` 仅为了与 A2A3 的 API 兼容性而保留在 C++ 内建接口签名中。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, slope, out, tmp;
  TPRELU(out, x, slope, tmp);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tprelu %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tprelu %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.tprelu ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

