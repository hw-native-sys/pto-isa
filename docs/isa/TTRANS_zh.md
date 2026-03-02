# TTRANS

## 指令示意图

![TTRANS tile operation](../figures/isa/TTRANS.svg)

## 简介

使用实现定义的临时 Tile 进行转置。

## 数学语义

For a 2D tile, over the effective transpose domain:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{j,i} $$

Exact shape/layout and the transpose domain depend on the target (see Constraints).

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = ttrans %src : !pto.tile<...> -> !pto.tile<...>
```
Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### IR Level 1（SSA）

```text
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TTRANS(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp, WaitEvents&... events);
```

## 约束

- **实现检查 (A2A3)**:
  - `sizeof(TileDataSrc::DType) == sizeof(TileDataDst::DType)`.
  - Source layout must be row-major (`TileDataSrc::isRowMajor`).
  - Element size must be `1`, `2`, or `4` bytes.
  - Supported element types are restricted per element width:
    - 4 bytes: `uint32_t`, `int32_t`, `float`
    - 2 bytes: `uint16_t`, `int16_t`, `half`, `bfloat16_t`
    - 1 byte: `uint8_t`, `int8_t`
  - The transpose size is taken from `src.GetValidRow()` / `src.GetValidCol()`.
- **实现检查 (A5)**:
  - `sizeof(TileDataSrc::DType) == sizeof(TileDataDst::DType)`.
  - 32-byte alignment constraints are enforced on the major dimension of both input and output (row-major checks `Cols * sizeof(T) % 32 == 0`, col-major checks `Rows * sizeof(T) % 32 == 0`).
  - Supported element types are restricted per element width:
    - 4 bytes: `uint32_t`, `int32_t`, `float`
    - 2 bytes: `uint16_t`, `int16_t`, `half`, `bfloat16_t`
    - 1 byte: `uint8_t`, `int8_t`
  - The implementation operates over the static tile shape (`TileDataSrc::Rows/Cols`) and does not consult `GetValidRow/GetValidCol`.
- **Temporary tile**:
  - The C++ API requires `tmp`, but some implementations may not use it.

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TTRANS(dst, src, tmp);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TTRANS(dst, src, tmp);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = ttrans %src : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

