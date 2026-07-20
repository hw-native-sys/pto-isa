# TANDS

## 指令示意图

![TANDS tile operation](../figures/isa/TANDS.svg)

## 简介

Tile与标量的逐元素按位与。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \;\&\; \mathrm{scalar} $$

## 汇编语法

同步形式：

```text
%dst = tands %src, %scalar : !pto.tile<...>, i32
```

### AS Level 1（SSA）

```text
%dst = pto.tands %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tands ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TANDS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar, WaitEvents &... events);
```

## 约束

- **实现检查 (Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品)**:
    - 适用于整数元素类型。
    - `dst` 和 `src` 必须使用相同的元素类型。
    - `dst` 和 `src` 必须是向量Tile。
    - 运行时：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`。
    - 在手动模式下，不支持将源Tile和目标Tile设置为相同的内存。
- **实现检查 (Ascend 950PR/Ascend 950DT)**:
    - 适用于 `TANDS` 支持的整数元素类型。
    - `dst` 和 `src` 必须使用相同的元素类型。
    - `dst` 和 `src` 必须是向量Tile。
    - 运行时：`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`。
    - 在手动模式下，不支持将源Tile和目标Tile设置为相同的内存。
- **有效区域**:
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint16_t, 16, 16>;
  using TileSrc = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileDst dst;
  TileSrc src;
  TANDS(dst, src, 0xffu);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tands %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tands %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### PTO汇编形式

```text
%dst = tands %src, %scalar : !pto.tile<...>, i32
# AS Level 2 (DPS)
pto.tands ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```
