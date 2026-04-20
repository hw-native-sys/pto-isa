# pto.trowargmax

`pto.trowargmax` 属于[行归约](./tile/ops/reduce-and-expand/trowargmax_zh.md)指令集。

## 概述

获取每行最大值对应列索引。对源 tile 的每一行，计算该行最大元素的列索引，写入目标 tile 对应行的第一个位置。

## 机制

设 `R = src.GetValidRow()`，`C = src.GetValidCol()`。对 `0 <= i < R`：

$$ \mathrm{dst}_{i,0} = \underset{0 \le j < C}{\operatorname{argmax}} \; \mathrm{src}_{i,j} $$

## 语法

### PTO-AS

```text
%dst = trowargmax %src : !pto.tile<...> -> !pto.tile<...>
```
Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### AS Level 1（SSA）

```mlir
%dst = pto.trowargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.trowargmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWARGMAX(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源 tile，支持 `half`、`float` 元素类型 |
| `tmp` | 临时 | A3 临时 tile，取决于 `srcValidCol` 与 `ElementPerRepeat` 的关系 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | 输出 | 每行最大值的列索引，`uint32_t` 或 `int32_t` 类型 |

## 副作用

指令执行完成后，`dst` 有效行数与 `src` 相同，有效列数为 1。

## 约束

- `dst` 和 `src` 必须为 `TileType::Vec`
- 支持的源元素类型：`half`、`float`
- 支持的目标元素类型：`uint32_t`、`int32_t`
- `src.GetValidRow() != 0`
- `src.GetValidCol() != 0`
- `src.GetValidRow() == dst.GetValidRow()`
- `src` 必须使用标准 ND 布局：行主且非分形（`BLayout::RowMajor`、`SLayout::NoneBox`）
- `dst` 可使用单列 DN 布局（`BLayout::ColMajor`、`Cols == 1`）或有效列数为 1 的 ND 布局

### A3 `tmp`临时Tile相关说明

- `tmp`临时Tile在`srcValidCol <= ElementPerRepeat`时不使用，`srcValidCol > ElementPerRepeat`时需要使用
- `tmp` tile的行数和`src` tile的行数相同
- 按以下公式根据`src` tile的`validCol`算出`tmp` tile所需stride：

```text
repeats = ceil(validCol / elementPerRepeat)
stride = ceil(repeats * 2 / elementPerBlock) * elementPerBlock + ceil(repeats / elementPerBlock) * elementPerBlock
```

## 异常与非法情形

- 运行时检查失败时，行为由具体实现定义

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 支持 | 是 | 是 | 是 |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint32_t, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWARGMAX(dst, src, tmp);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint32_t, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TROWARGMAX(dst, src, tmp);
}
```

### PTO-AS

```text
# 自动模式
%dst = pto.trowargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

## 相关页面

- 指令集总览：[行归约](./tile/ops/reduce-and-expand/trowargmax_zh.md)
