# pto.trowprod

`pto.trowprod` 属于[行归约](./tile/ops/reduce-and-expand/trowprod_zh.md)指令集。

## 概述

对每行元素进行乘积归约。将源 tile 每行的所有元素相乘，结果写入目标 tile 对应行的第一个位置。

## 机制

设 `R = src.GetValidRow()` 且 `C = src.GetValidCol()`。对于 `0 <= i < R`：

$$ \mathrm{dst}_{i,0} = \prod_{j=0}^{C-1} \mathrm{src}_{i,j} $$

## 语法

### PTO-AS

```text
%dst = trowprod %src : !pto.tile<...> -> !pto.tile<...>
```
降级可能引入内部临时 tile；C++ 内建函数需要显式的 `tmp` 操作数。

### AS Level 1（SSA）

```mlir
%dst = pto.trowprod %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.trowprod ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWPROD(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源 tile，支持 `half`、`float`、`int32_t`、`int16_t` 元素类型 |
| `tmp` | 临时 | 临时 tile，接口保留但当前实现路径无特定约束 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | 输出 | 每行元素的乘积结果，类型与 `src` 一致 |

## 副作用

指令执行完成后，`dst` 有效行数与 `src` 相同，有效列数为 1。

## 约束

- `dst` 和 `src` 必须均为 `TileType::Vec`
- `src` 必须使用标准 ND 布局：行主且非分形（`BLayout::RowMajor`、`SLayout::NoneBox`）
- `dst` 必须使用以下两种非分形布局之一：
    - ND 布局（`BLayout::RowMajor`、`SLayout::NoneBox`）
    - 列数严格为 1 的 DN 布局（`BLayout::ColMajor`、`SLayout::NoneBox`、`Cols == 1`）
- `dst` 和 `src` 的元素类型必须一致
- `src.GetValidRow() != 0`
- `src.GetValidCol() != 0`
- `src.GetValidRow() == dst.GetValidRow()`
- 内建接口签名要求显式传入 `tmp` 操作数

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
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWPROD(dst, src, tmp);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TROWPROD(dst, src, tmp);
}
```

### PTO-AS

```text
# 自动模式
%dst = pto.trowprod %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

## 相关页面

- 指令集总览：[行归约](./tile/ops/reduce-and-expand/trowprod_zh.md)
