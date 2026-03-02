# TSHLS

## 指令示意图

![TSHLS tile operation](../figures/isa/TSHLS.svg)

## 简介

Tile 按标量逐元素左移。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \ll \mathrm{scalar} $$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = tshls %src, %scalar : !pto.tile<...>, i32
```

### IR Level 1（SSA）

```text
%dst = pto.tshls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tshls ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSHLS(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, WaitEvents&... events);
```

## 约束

- Intended for integral element types.
- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.
- Scalar only support zero and positive value.

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint16_t, 16, 16>;
  using TileSrc = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileDst dst;
  TileSrc src;
  TSHLS(dst, src, 0x2);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tshls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tshls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tshls %src, %scalar : !pto.tile<...>, i32
# IR Level 2 (DPS)
pto.tshls ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

