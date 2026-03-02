# TCOLSUM

## 指令示意图

![TCOLSUM tile operation](../figures/isa/TCOLSUM.svg)

## 简介

通过对行求和来归约每一列。

## 数学语义

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= j < C`:

$$ \mathrm{dst}_{0,j} = \sum_{i=0}^{R-1} \mathrm{src}_{i,j} $$

`isBinary` selects the implementation path (binary-tree accumulation vs. sequential accumulation).

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = tcolsum %src {isBinary = false} : !pto.tile<...> -> !pto.tile<...>
```
Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### IR Level 1（SSA）

```text
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
%dst = pto.tcolsum %src, %tmp {isBinary = false} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tcolsum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tcolsum ins(%src, %tmp {isBinary = false} : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, bool isBinary,
                             WaitEvents&... events);
```

## 约束

实现检查 (NPU):

- Tile location: `dst`, `src`, `tmp` must be `TileType::Vec`.
- Tile 布局: all tiles must be ND fractal (`isRowMajor` and `SLayout::NoneBox`).
- 数据类型一致性:
  - A2A3: `src.DType` must be one of `half`, `float`, `int16_t`, `int32_t`, and `dst.DType == tmp.DType == src.DType`.
  - A5: `dst.DType == src.DType` is required by `TColReduceCheck`; the exact supported `src.DType` set is target-defined (see `include/pto/npu/a5/TColReduceOps.hpp`).
- 运行期有效区域检查:
  - A2A3: `src.GetValidCol() == dst.GetValidCol()`; returns early if `src.GetValidRow() == 0` or `src.GetValidCol() == 0`.
  - A5: `srcValidRow` and `srcValidCol` must be non-zero; `srcValidCol == dstValidCol` is asserted by `TColReduceCheck`.

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TCOLSUM(dst, src, tmp, /*isBinary=*/false);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TCOLSUM(dst, src, tmp, /*isBinary=*/false);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tcolsum %src {isBinary = false} : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tcolsum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

