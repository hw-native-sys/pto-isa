# pto.tci

`pto.tci` 属于[不规则与复杂指令](../../irregular-and-complex_zh.md)集。

## 概述

TCI 在目标 Tile 的有效区域内生成连续整数序列，序列的起始值由标量参数 `S` 指定。当 `descending = false` 时序列递增，当 `descending = true` 时序列递减。对于线性化索引 `k`，升序时 $\mathrm{dst}_{k} = S + k$，降序时 $\mathrm{dst}_{k} = S - k$。线性化顺序取决于 Tile 的布局（实现定义）。

## 语法

### PTO-AS

```text
%dst = tci %S {descending = false} : !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData &dst, T start, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `dst` | 输出 | 目标 Tile，接收生成的整数序列 |
| `S` | 标量输入 | 序列的起始值 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 包含连续整数序列的 Tile |

## 副作用

该指令可能会读取或写入 Tile 的有效区域标记。

## 约束

- `TileData::DType` 必须与标量模板参数 `T` 类型完全一致。
- `dst`/`scalar` 元素类型必须相同，且必须是以下类型之一：`int32_t`、`uint32_t`、`int16_t`、`uint16_t`。
- `TileData::Cols != 1`（这是实现强制执行的条件）。
- 实现使用 `dst.GetValidCol()` 作为序列长度，不使用 `dst.GetValidRow()`。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, int32_t, 1, 16>;
  TileT dst;
  TCI<TileT, int32_t, /*descending=*/0>(dst, /*S=*/0);
}
```

### C++ 手动模式

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

## 相关页面

- 指令集总览：[不规则与复杂指令](../../irregular-and-complex_zh.md)
