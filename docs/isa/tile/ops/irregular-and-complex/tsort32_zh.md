# pto.tsort32

`pto.tsort32` 属于[不规则与复杂指令](../../irregular-and-complex_zh.md)集。

## 概述

对 `src` 的每个 32 元素块，与 `idx` 中对应的索引一起进行排序，并将排序后的值-索引对写入 `dst`。`idx` 是输入 Tile 而非输出 Tile，提供与 `src` 一起参与重排的索引。`dst` 保存的是排序后的值-索引对，而不只是排序后的值。在 CPU 仿真实现中，按值降序排序；当值相同时，索引较小者优先。

对每一行，TSORT32 按独立的 32 元素块处理 `src`。设第 `b` 个块覆盖列 `32b ... 32b+31`，该块的有效元素数为 `n_b = min(32, C - 32b)`。对于块中的每个有效元素，先构造二元组 `(v_k, i_k) = (\mathrm{src}_{r,32b+k}, \mathrm{idx}_{r,32b+k})`，然后按值对这些二元组排序。

## 语法

### PTO-AS

```text
%dst = tsort32 %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tsort32 %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tsort32 ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INST RecordEvent TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx);

template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData>
PTO_INST RecordEvent TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx, TmpTileData &tmp);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `dst` | 输出 | 目标 Tile，接收排序后的值-索引对 |
| `src` | 输入 | 源 Tile，包含待排序的值 |
| `idx` | 输入 | 索引 Tile，提供与 src 一起参与重排的索引 |
| `tmp` | 临时 | 可选，用于支持非 32 对齐尾块 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 排序后的值-索引对 |

## 副作用

该指令可能会写入 Tile 的有效区域标记。

## 约束

- TSORT32 不接受 `WaitEvents&...` 参数，也不在内部调用 `TSYNC(...)`；如有需要请显式同步。
- `idx` 在两个重载中都是必需的输入操作数，提供与 `src` 一起参与重排的索引。
- `DstTileData::DType` 必须是 `half` 或 `float`。
- `SrcTileData::DType` 必须与 `DstTileData::DType` 匹配。
- `IdxTileData::DType` 必须是 `uint32_t`。
- `dst`/`src`/`idx` Tile 类型必须是 `TileType::Vec`，且都必须是行主序（`isRowMajor`）。
- 实现使用 `dst.GetValidRow()` 作为行数，使用 `src.GetValidCol()` 确定每行参与排序的元素数量。
- 排序按独立的 32 元素块进行；4 参数重载额外通过 `tmp` 支持非 32 对齐尾块。

## 异常与非法情形

- 输入 Tile 类型不是 `TileType::Vec` 时行为未定义。
- 元素类型不匹配时行为未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 1, 32>;
  using IdxT = Tile<TileType::Vec, uint32_t, 1, 32>;
  using DstT = Tile<TileType::Vec, float, 1, 64>;
  SrcT src;
  IdxT idx;
  DstT dst;
  TSORT32(dst, src, idx);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 1, 32>;
  using IdxT = Tile<TileType::Vec, uint32_t, 1, 32>;
  using DstT = Tile<TileType::Vec, float, 1, 64>;
  SrcT src;
  IdxT idx;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(idx, 0x2000);
  TASSIGN(dst, 0x3000);
  TSORT32(dst, src, idx);
}
```

### PTO-AS

```text
%dst = pto.tsort32 %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tsort32 ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[不规则与复杂指令](../../irregular-and-complex_zh.md)
- 相关指令：[TMRGSORT](./tmrgsort_zh.md)
