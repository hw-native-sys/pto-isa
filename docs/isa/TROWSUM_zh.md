# pto.trowsum

`pto.trowsum` 属于[归约指令](./tile/reduce-and-expand_zh.md)集。

## 概述

通过对列求和来归约每一行。

## 机制

设 `R = src.GetValidRow()`，`C = src.GetValidCol()`。对 `0 <= i < R`：

$$ \mathrm{dst}_{i,0} = \sum_{j=0}^{C-1} \mathrm{src}_{i,j} $$

## 语法

### PTO-AS

参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = trowsum %src : !pto.tile<...> -> !pto.tile<...>
```

降低时可能引入内部临时 Tile；C++ 内建接口需要显式传入 `tmp` 操作数。

### AS Level 1（SSA）

```mlir
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src` | 源 Tile | 输入 Tile |
| `tmp` | 临时 Tile | 用于内部计算的临时 Tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 按行归约求和后的目标 Tile |

## 副作用

无。

## 约束

- `dst` 和 `src` 必须均为 `TileType::Vec`。
- `src` 必须使用标准 ND 布局：行主且非分形（`BLayout::RowMajor`、`SLayout::NoneBox`）。
- `dst` 必须使用以下两种非分形布局之一：
    - ND 布局（`BLayout::RowMajor`、`SLayout::NoneBox`），或
    - 列数严格为 1 的 DN 布局（`BLayout::ColMajor`、`SLayout::NoneBox`、`Cols == 1`）。
- `dst` 和 `src` 的元素类型必须一致。
- 运行时有效区域检查：
    - `src.GetValidRow() != 0`
    - `src.GetValidCol() != 0`
    - `src.GetValidRow() == dst.GetValidRow()`
- 内建接口签名要求显式传入 `tmp` 操作数。

## 异常与非法情形

- 未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 支持的元素类型 | - | `half`、`float`、`int32_t`、`int16_t` | - |
| 输出布局 | - | ND 或 `Cols==1` DN | - |

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
  TROWSUM(dst, src, tmp);
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
  TROWSUM(dst, src, tmp);
}
```

### PTO-AS

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>

# 手动模式：先显式绑定资源，再发射指令。
# pto.tassign %src, @tile(0x1000)
# pto.tassign %dst, @tile(0x2000)
# pto.tassign %tmp, @tile(0x3000)
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

## 相关页面

- 指令集总览：[归约指令](./tile/reduce-and-expand_zh.md)
