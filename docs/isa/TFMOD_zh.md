# TFMOD

## 指令示意图

![TFMOD tile operation](../figures/isa/TFMOD.svg)

## 简介

两个 Tile 的逐元素余数，余数符号与被除数相同。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$\mathrm{dst}_{i,j} = \mathrm{fmod}(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j})$$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = tfmod %src0, %src1 : !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tfmod %src0, %src1 : !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tfmod ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TFMOD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents&... events);
```

## 约束

- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.
- Division-by-zero behavior is target-defined; the CPU simulator asserts in debug builds.
- Temporary space is required by A3 for calculation, while not used by A5.

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT out, a, b;
  TFMOD(out, a, b);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tfmod %src0, %src1 : !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tfmod %src0, %src1 : !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tfmod %src0, %src1 : !pto.tile<...>
# IR Level 2 (DPS)
pto.tfmod ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

