# TSORT32

## 指令示意图

![TSORT32 tile operation](../figures/isa/TSORT32.svg)

## 简介

对固定大小的 32 元素块进行排序并生成索引映射。

## 数学语义

将 `src` 中的值排序后写入 `dst`，并在 `idx` 中生成索引映射。概念上，对每一行 `i`：

$$ \mathrm{dst}_{i,k} = \mathrm{src}_{i,\pi_i(k)} $$

其中 $\pi_i$ 是该行索引的一个排列。排序顺序和稳定性由目标定义。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst, %idx = tsort32 %src : !pto.tile<...> -> (!pto.tile<...>, !pto.tile<...>)
```

### AS Level 1（SSA）

```text
%dst, %idx = pto.tsort32 %src : !pto.tile<...> -> (!pto.tile<...>, !pto.tile<...>)
```

### AS Level 2（DPS）

```text
pto.tsort32 ins(%src : !pto.tile_buf<...>) outs(%dst, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INST RecordEvent TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx);

template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData>
PTO_INST RecordEvent TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx, TmpTileData &tmp);
```

## 约束

- `TSORT32` 不接受 `WaitEvents&...` 参数，也不在内部调用 `TSYNC(...)`；如有需要请显式同步。
- **实现检查 (A2A3/A5)**:
    - `DstTileData::DType` 必须是 `half` 或 `float`。
    - `SrcTileData::DType` 必须与 `DstTileData::DType` 匹配。
    - `IdxTileData::DType` 必须是 `uint32_t`。
    - `dst`/`src`/`idx` Tile 位置必须是 `TileType::Vec`，且都必须是行主序（`isRowMajor`）。
- **有效区域**:
    - 实现使用 `dst.GetValidRow()` 作为行数，并使用 `src.GetValidCol()` 确定每行需要排序的 32 元素块的数量。

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
# AS Level 2 (DPS)
pto.tsort32 ins(%src : !pto.tile_buf<...>) outs(%dst, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

