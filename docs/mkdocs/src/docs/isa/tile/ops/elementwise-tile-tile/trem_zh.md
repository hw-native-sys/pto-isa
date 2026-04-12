<!-- Generated from `docs/isa/tile/ops/elementwise-tile-tile/trem_zh.md` -->

# TREM

## 指令示意图

![TREM tile operation](../figures/isa/TREM.svg)

## 简介

两个 Tile 的逐元素余数运算。结果符号与除数相同。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \bmod \mathrm{src1}_{i,j}$$

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = trem %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trem ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TREM(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束

- **实现检查 (A2A3)**:
    - `dst`、`src0` 和 `src1` 必须使用相同的元素类型。
    - 支持的元素类型：`float` 和 `int32_t`。
    - `dst`、`src0` 和 `src1` 必须是向量 Tile。
    - `dst`、`src0` 和 `src1` 必须是行主序。
    - 运行时：`dst.GetValidRow() == src0.GetValidRow() == src1.GetValidRow() > 0` 且 `dst.GetValidCol() == src0.GetValidCol() == src1.GetValidCol() > 0`。
    - **tmp 缓冲区要求**：
      - `tmp.GetValidCol() >= dst.GetValidCol()`（至少与 dst 相同的列数）
      - `tmp.GetValidRow() >= 1`（至少 1 行）
      - 数据类型必须与 `TileDataDst::DType` 匹配。
- **实现检查 (A5)**:
    - `dst`、`src0` 和 `src1` 必须使用相同的元素类型。
    - 支持的元素类型：`float`、`int32_t`、`uint32_t`、`half`、`int16_t` 和 `uint16_t`。
    - `dst`、`src0` 和 `src1` 必须是向量 Tile。
    - 静态有效边界：所有 Tile 都必须满足 `ValidRow <= Rows` 且 `ValidCol <= Cols`。
    - 运行时：`dst.GetValidRow() == src0.GetValidRow() == src1.GetValidRow()` 且 `dst.GetValidCol() == src0.GetValidCol() == src1.GetValidCol()`。
    - 注意：tmp 参数在 A5 上被接受但不进行验证或使用。
- **除零**:
    - 行为由目标定义；CPU 模拟器在调试构建中会断言。
- **有效区域**:
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域。
- **对于 `int32_t` 输入（仅 A2A3）**：`src0` 和 `src1` 的所有元素必须在 `[-2^24, 2^24]` 范围内（即 `[-16777216, 16777216]`），以确保在计算过程中能精确转换为 float32。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT out, a, b;
  Tile<TileType::Vec, int32_t, 16, 16> tmp;
  TREM(out, a, b, tmp);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trem %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = trem %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.trem ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
