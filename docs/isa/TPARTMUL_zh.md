# TPARTMUL

## 指令示意图

![TPARTMUL tile operation](../figures/isa/TPARTMUL.svg)

## 简介

部分逐元素乘法，对有效区域不一致的处理为实现定义。

## 数学语义

对每个元素 `(i, j)` in the destination valid region:

$$
\mathrm{dst}_{i,j} =
egin{cases}
\mathrm{src0}_{i,j} \cdot \mathrm{src1}_{i,j} & 	ext{if both inputs are defined at } (i,j) \
\mathrm{src0}_{i,j} & 	ext{if only src0 is defined at } (i,j) \
\mathrm{src1}_{i,j} & 	ext{if only src1 is defined at } (i,j)
\end{cases}
$$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tpartmul ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## 约束

- Element type/layout legality follows backend checks and is analogous to `TPARTADD` / `TPARTMAX` / `TPARTMIN`.
- Destination valid region defines the result domain.
- Partial-validity handling is implementation-defined for unsupported shape combinations.

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TPARTMUL(dst, src0, src1);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TPARTMUL(dst, src0, src1);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tpartmul %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tpartmul ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

