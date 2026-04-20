# pto.tsubview

`pto.tsubview` 属于[同步与配置指令](../../sync-and-config_zh.md)集。

## 概述

`TSUBVIEW` 表达一个 Tile 是另一个 Tile 的子视图，通过指定起始行和列偏移量来定义源 Tile 内的一个区域作为目标 Tile 的数据来源。

## 机制

对于 `dst` 中有效区域内的每一个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{rowIdx} + i,\mathrm{colIdx} + j} $$

其中 `rowIdx` 是源 Tile 有效区域内的起始行索引，`colIdx` 是源 Tile 有效区域内的起始列索引。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.tsubview %src, %rowIdx, %colIdx : !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tsubview ins(%src, %rowIdx, %colIdx : !pto.tile<...>, ui16, ui16)
             outs(%dst : !pto.tile<...>)
```

## C++ 内建接口

定义在 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSUBVIEW(TileDataDst &dst, TileDataSrc &src, uint16_t rowIdx, uint16_t colIdx, WaitEvents&... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `dst` | 输出 | 目标 Tile（子视图） |
| `src` | 输入 | 源 Tile |
| `rowIdx` | 输入 | 起始行偏移量 |
| `colIdx` | 输入 | 起始列偏移量 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 绑定到源 Tile 指定区域的子视图 |

## 副作用

将 `dst` 绑定为 `src` 的子视图，后续对 `dst` 的操作实际作用于源 Tile 的对应区域。

## 约束

- Tile 类型必须相同：`TileDataSrc::Loc == TileDataDst::Loc`。
- 输入和输出 Tile 的静态 shape 必须相同：`TileDataSrc::Rows == TileDataDst::Rows` 且 `TileDataSrc::Cols == TileDataDst::Cols`。
- 输入和输出 Tile 的 BLayout 必须相同：`TileDataSrc::BFractal == TileDataDst::BFractal`。
- 源 Tile 的 validRow 和 validCol 必须大于等于目标 Tile 的 validRow 和 validCol。

## 异常与非法情形

- 当 `rowIdx` 或 `colIdx` 超出源 Tile 有效区域时，行为未定义。
- 当源和目标 Tile 类型不匹配时，编译错误。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 子视图支持 | 是 | 是 | 是 |

## 示例

### C++

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using Src = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, 4, 64>;
  using Dst = Tile<TileType::Vec, float, 4, 64, BLayout::RowMajor, 2, 32>;

  Src src;
  Dst dst0;
  Dst dst1;
  Dst dst2;
  Dst dst3;

  // e.g. split into four 2x32 subtiles
  TSUBVIEW(dst0, src, 0, 0);
  TSUBVIEW(dst1, src, 0, 32);
  TSUBVIEW(dst2, src, 2, 0);
  TSUBVIEW(dst3, src, 2, 32);
}
```

### PTO-AS

```text
%dst = pto.tsubview %src, %rowIdx, %colIdx : !pto.tile<...>
```

## 相关页面

- 指令集总览：[同步与配置](../../sync-and-config_zh.md)
- [TASSIGN](./tassign_zh.md)
- [TSUBVIEW](./tsubview_zh.md)
