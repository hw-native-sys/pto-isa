# TSCATTER

## 指令示意图

![TSCATTER tile operation](../figures/isa/TSCATTER.svg)

## 简介

使用逐元素行索引将源 Tile 的行散播到目标 Tile 中。

## 数学语义

对每个源元素 `(i, j)`, write:

$$ \mathrm{dst}_{\mathrm{idx}_{i,j},\ j} = \mathrm{src}_{i,j} $$

If multiple elements map to the same destination location, the final value is implementation-defined (last writer wins in the current implementation).

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = tscatter %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataD, typename TileDataS, typename TileDataI, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(TileDataD& dst, TileDataS& src, TileDataI& indexes, WaitEvents&... events);
```

## 约束

- **实现检查 (A2A3)**:
  - `TileDataD::Loc`, `TileDataS::Loc`, `TileDataI::Loc` must be `TileType::Vec`.
  - `TileDataD::DType`, `TileDataS::DType` must be one of: `int32_t`, `int16_t`, `int8_t`, `half`, `float32_t`, `uint32_t`, `uint16_t`, `uint8_t`, `bfloat16_t`.
  - `TileDataI::DType` must be one of: `int16_t`, `int32_t`, `uint16_t` or `uint32_t`.
  - No bounds checks are enforced on `indexes` values.
  - Static valid bounds: `TileDataD::ValidRow <= TileDataD::Rows`, `TileDataD::ValidCol <= TileDataD::Cols`, `TileDataS::ValidRow <= TileDataS::Rows`, `TileDataS::ValidCol <= TileDataS::Cols`, `TileDataI::ValidRow <= TileDataI::Rows`, `TileDataI::ValidCol <= TileDataI::Cols`.
  - `TileDataD::DType` and `TileDataS::DType` must be the same.
  - When size of `TileDataD::DType` is 4 bytes, the size of `TileDataI::DType` must be 4 bytes.
  - When size of `TileDataD::DType` is 2 bytes, the size of `TileDataI::DType` must be 2 bytes.
  - When size of `TileDataD::DType` is 1 bytes, the size of `TileDataI::DType` must be 2 bytes.
- **实现检查 (A5)**:
  - `TileDataD::Loc`, `TileDataS::Loc`, `TileDataI::Loc` must be `TileType::Vec`.
  - `TileDataD::DType`, `TileDataS::DType` must be one of: `int32_t`, `int16_t`, `int8_t`, `half`, `float32_t`, `uint32_t`, `uint16_t`, `uint8_t`, `bfloat16_t`.
  - `TileDataI::DType` must be one of: `int16_t`, `int32_t`, `uint16_t` or `uint32_t`.
  - No bounds checks are enforced on `indexes` values.
  - Static valid bounds: `TileDataD::ValidRow <= TileDataD::Rows`, `TileDataD::ValidCol <= TileDataD::Cols`, `TileDataS::ValidRow <= TileDataS::Rows`, `TileDataS::ValidCol <= TileDataS::Cols`, `TileDataI::ValidRow <= TileDataI::Rows`, `TileDataI::ValidCol <= TileDataI::Cols`.
  - `TileDataD::DType` and `TileDataS::DType` must be the same.
  - When size of `TileDataD::DType` is 4 bytes, the size of `TileDataI::DType` must be 4 bytes.
  - When size of `TileDataD::DType` is 2 bytes, the size of `TileDataI::DType` must be 2 bytes.
  - When size of `TileDataD::DType` is 1 bytes, the size of `TileDataI::DType` must be 2 bytes.

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TSCATTER(dst, src, idx);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(idx, 0x3000);
  TSCATTER(dst, src, idx);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tscatter %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

