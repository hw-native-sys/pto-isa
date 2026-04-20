# pto.tcolargmin

`pto.tcolargmin` 属于[归约与扩展](./tile/reduce-and-expand_zh.md)指令集。

## 概述

获取每列最小值对应行索引。

## 机制

设 `R = src.GetValidRow()`，`C = src.GetValidCol()`。对 `0 <= j < C`：

$$ \mathrm{dst}_{0,j} = \underset{0 \le i < R}{\operatorname{argmin}} \; \mathrm{src}_{i,j} $$

输出 tile 中每个元素为源 tile 对应列中最小值的行索引（0-based）。`tmp` 操作数用于存储中间结果（当前行索引、argmin 索引、当前最小值元素）。若 `src.GetValidRow() == 0` 或 `src.GetValidCol() == 0`，实现会直接返回。

A2A3 实现中 `tmp` 始终被使用；A5 实现中 `tmp` 保留但不使用。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.tcolargmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tcolargmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMIN(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | 输入 tile，源元素类型支持数值类型 |
| `%dst` | 目标 tile | 接收 argmin 索引结果，目标元素类型为 `uint32_t` 或 `int32_t` |
| `%tmp` | 临时 tile | A2A3 必需（存储中间结果）；A5 保留但不使用 |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<1, C>` | 每列最小值的行索引（0-based） |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 和 `src` 必须为 `TileType::Vec`。
- 由于辅助检查仅要求 `SLayout::NoneBox`，`src` 可使用 ND 或 DN 的非分形布局。
- `dst` 必须使用标准 ND 布局：行主且非分形（`BLayout::RowMajor`、`SLayout::NoneBox`）。
- 目标元素类型：`uint32_t`、`int32_t`。
- 编译时检查：`TileDataIn::ValidCol == 1 || TileDataIn::ValidCol == -1`。
- 运行时检查：`src.GetValidRow() != 0`、`src.GetValidCol() != 0`、`dst.GetValidRow() == 1`、`src.GetValidCol() == dst.GetValidCol()`。
- A2A3：`tmp` 元素类型必须与 `src` 一致，`tmp` 用作索引跟踪和当前比较值的临时存储。
- A5：`tmp` 保留但不使用，A5 使用基于向量寄存器的计算方式。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 源 `float` | Simulated | Supported | Supported |
| 源 `half` | Simulated | Supported | Supported |
| 源 `uint16_t` / `uint32_t` | — | Supported | Supported |
| 源 `int8_t` / `uint8_t` / `int16_t` / `int32_t` | — | — | Supported |
| 目标 `uint32_t` / `int32_t` | Simulated | Supported | Supported |
| `tmp` 临时 tile | — | Required | Unused (API compat) |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1>;
  using DstT = Tile<TileType::Vec, uint32_t, 1, 256, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1>;
  SrcT src(16, 255);
  DstT dst(1, 255);
  TmpT tmp(1, 32);
  TCOLARGMIN(dst, src, tmp);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1>;
  using DstT = Tile<TileType::Vec, uint32_t, 1, 256, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1>;
  SrcT src(16, 255);
  DstT dst(1, 255);
  TmpT tmp(1, 32);
  TASSIGN(src, 0x0);
  TASSIGN(dst, 0x1000);
  TASSIGN(tmp, 0x2000);
  TCOLARGMIN(dst, src, tmp);
}
```

### PTO-AS

```text
# 自动模式
%dst = pto.tcolargmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>

# 手动模式
pto.tassign %arg0, @tile(0x1000)
pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcolargmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>

# PTO 汇编形式
%dst = tcolargmin %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcolargmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[归约与扩展](./tile/reduce-and-expand_zh.md)
