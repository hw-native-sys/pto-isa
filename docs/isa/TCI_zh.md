# TCI

## 指令示意图

![TCI tile operation](../figures/isa/TCI.svg)

## 简介

生成连续整数序列到目标 Tile 中。

## 数学语义

For a linearized index `k` over the valid elements:

- Ascending:

  $$ \mathrm{dst}_{k} = S + k $$

- Descending:

  $$ \mathrm{dst}_{k} = S - k $$

The linearization order depends on the tile layout (implementation-defined).

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = tci %S {descending = false} : !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData& dst, T S, WaitEvents&... events);
```

## 约束

- **实现检查 (A2A3/A5)**:
  - `TileData::DType` must be exactly the same type as the scalar template parameter `T`.
  - `dst/scalar` element types must be identical, and must be one of: `int32_t`, `uint32_t`, `int16_t`, `uint16_t`.
  - `TileData::Cols != 1` (this is the condition enforced by the implementation).
- **有效区域**:
  - The implementation uses `dst.GetValidCol()` as the sequence length and does not consult `dst.GetValidRow()`.

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, int32_t, 1, 16>;
  TileT dst;
  TCI<TileT, int32_t, /*descending=*/0>(dst, /*S=*/0);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, int32_t, 1, 16>;
  TileT dst;
  TASSIGN(dst, 0x1000);
  TCI<TileT, int32_t, /*descending=*/1>(dst, /*S=*/100);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tci %S {descending = false} : !pto.tile<...>
# IR Level 2 (DPS)
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```

