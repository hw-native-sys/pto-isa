# pto.tmrgsort

`pto.tmrgsort` 属于[不规则与复杂指令](../../irregular-and-complex_zh.md)集。

## 概述

TMRGSORT 用于把多个已经排好序的列表按目标定义的键顺序做归并。它不是对一个无序 Tile 排序，而是归并多个有序输入。该指令支持两类接口：多列表归并（2/3/4 路输入）和单列表块归并（将源 Tile 中连续放置的 4 个已排序块归并成一个更大的有序结果）。

TMRGSORT 的输入并不是任意数组，而是按固定结构组织的记录流。当前实现里，一个记录按 8 字节结构处理：float 类型时每条记录占 2 个元素，half 类型时占 4 个元素。CPU 模拟器按每条结构的第一个元素作为排序键，并优先选更大的键值；NPU backend 通过 `vmrgsort4` 完成硬件归并。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.tmrgsort %src, %blockLen : (!pto.tile<...>, dtype) -> !pto.tile<...>
%dst, %executed = pto.tmrgsort %src0, %src1, %src2, %src3 {exhausted = false}
 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> (!pto.tile<...>, vector<4xi16>)
```

### AS Level 2（DPS）

```mlir
pto.tmrgsort ins(%src, %blockLen : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
pto.tmrgsort ins(%src0, %src1, %src2, %src3 {exhausted = false} : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
outs(%dst, %executed : !pto.tile_buf<...>, vector<4xi16>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, typename Src3TileData, bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                              Src0TileData &src0, Src1TileData &src1, Src2TileData &src2, Src3TileData &src3,
                              WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, SrcTileData &src, uint32_t blockLen, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `dst` | 输出 | 归并结果 Tile |
| `src` | 输入 | 单列表块归并模式下的源 Tile |
| `src0~src3` | 输入 | 多列表归并模式下的源 Tile（最多4路） |
| `tmp` | 临时 | 多列表归并所需的临时 Tile |
| `blockLen` | 标量输入 | 单列表块归并中每个块的长度 |
| `executedNumList` | 输出 | 每个输入列表实际消费的记录数 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 归并后的有序结果 |
| `executed` | vector<4xi16> | 每路输入消费的记录数（多列表归并） |

## 副作用

该指令可能会读写 Tile 的有效区域标记，并使用临时存储。

## 约束

### 通用约束

- 所有参与 Tile 都必须是 `TileType::Vec`，`Rows == 1`，`BLayout::RowMajor`。
- 支持的数据类型是 `half` 或 `float`，且 `dst/tmp/src*` 的元素类型必须一致。

### 多列表归并

- 2/3/4 路版本都要求显式传入 `tmp`。
- `executedNumList` 返回每个输入列表实际消费的记录数。
- 模板参数 `exhausted` 决定某路输入先耗尽时是否提前停止归并。
- UB 使用量必须满足各 backend 的限制。

### 单列表块归并

- 假设 `src` 中顺序摆放了 4 个已排序块。
- `blockLen` 表示每个块的长度，包含记录值和索引/负载。
- `blockLen` 必须是 64 的倍数。
- `src.GetValidCol()` 必须是 `blockLen * 4` 的整数倍。
- `repeatTimes = src.GetValidCol() / (blockLen * 4)` 必须在 [1, 255] 范围内。

## 异常与非法情形

- 输入 Tile 类型不是 `TileType::Vec` 时行为未定义。
- `blockLen` 不是 64 的倍数时行为未定义。
- `repeatTimes` 超出 [1, 255] 范围时行为未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |

## 示例

### C++ 单列表块归并

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 1, 256>;
  using DstT = Tile<TileType::Vec, float, 1, 256>;
  SrcT src;
  DstT dst;
  TMRGSORT(dst, src, /*blockLen=*/64);
}
```

### C++ 双列表归并

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_merge2() {
  using TileT = Tile<TileType::Vec, float, 1, 256>;
  TileT src0, src1, tmp, dst;
  MrgSortExecutedNumList executed{};
  TMRGSORT<TileT, TileT, TileT, TileT, false>(dst, executed, tmp, src0, src1);
}
```

## 相关页面

- 指令集总览：[不规则与复杂指令](../../irregular-and-complex_zh.md)
- 相关指令：[TSORT32](./tsort32_zh.md)
