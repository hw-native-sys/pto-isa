# TSORT32

## 指令示意图

![TSORT32 tile operation](../figures/isa/TSORT32.svg)

## 简介

对固定大小的 32 元素块进行排序并生成索引映射。

## 数学语义

Sorts values from `src` into `dst` and produces an index mapping in `idx`. Conceptually, for each row `i`:

$$ \mathrm{dst}_{i,k} = \mathrm{src}_{i,\pi_i(k)} $$

where $\pi_i$ is a permutation of the indices in the row. Sort order and stability are target-defined.

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst, %idx = tsort32 %src : !pto.tile<...> -> (!pto.tile<...>, !pto.tile<...>)
```

### IR Level 1（SSA）

```text
%dst, %idx = pto.tsort32 %src : !pto.tile<...> -> (!pto.tile<...>, !pto.tile<...>)
```

### IR Level 2（DPS）

```text
pto.tsort32 ins(%src : !pto.tile_buf<...>) outs(%dst, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INST RecordEvent TSORT32(DstTileData& dst, SrcTileData& src, IdxTileData& idx);

template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData>
PTO_INST RecordEvent TSORT32(DstTileData& dst, SrcTileData& src, IdxTileData& idx, TmpTileData& tmp);
```

## 约束

- `TSORT32` does not take `WaitEvents&...` and does not call `TSYNC(...)` internally; synchronize explicitly if needed.
- **实现检查 (A2A3/A5)**:
  - `DstTileData::DType` must be `half` or `float`.
  - `SrcTileData::DType` must match `DstTileData::DType`.
  - `IdxTileData::DType` must be `uint32_t`.
  - `dst/src/idx` tile location must be `TileType::Vec`, and all must be row-major (`isRowMajor`).
- **有效区域**:
  - The implementation uses `dst.GetValidRow()` as the number of rows and uses `src.GetValidCol()` to determine how many 32-element blocks to sort per row.

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 1, 32>;
  using DstT = Tile<TileType::Vec, float, 1, 32>;
  using IdxT = Tile<TileType::Vec, uint32_t, 1, 32>;
  SrcT src;
  DstT dst;
  IdxT idx;
  TSORT32(dst, src, idx);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 1, 32>;
  using DstT = Tile<TileType::Vec, float, 1, 32>;
  using IdxT = Tile<TileType::Vec, uint32_t, 1, 32>;
  SrcT src;
  DstT dst;
  IdxT idx;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(idx, 0x3000);
  TSORT32(dst, src, idx);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst, %idx = pto.tsort32 %src : !pto.tile<...> -> (!pto.tile<...>, !pto.tile<...>)
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst, %idx = pto.tsort32 %src : !pto.tile<...> -> (!pto.tile<...>, !pto.tile<...>)
```

### PTO 汇编形式

```text
%dst, %idx = tsort32 %src : !pto.tile<...> -> (!pto.tile<...>, !pto.tile<...>)
# IR Level 2 (DPS)
pto.tsort32 ins(%src : !pto.tile_buf<...>) outs(%dst, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

