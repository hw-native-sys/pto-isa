# TCI

## 指令示意图

![TCI tile operation](../figures/isa/TCI.svg)

## 简介

生成连续整数序列到目标 Tile 中。

## 数学语义

对于有效元素上的线性化索引 `k`：

- 升序：

  $$ \mathrm{dst}_{k} = S + k $$

- 降序：

  $$ \mathrm{dst}_{k} = S - k $$

线性化顺序取决于 Tile 布局（实现定义）。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = tci %S {descending = false} : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData &dst, T start, WaitEvents &... events);
```

## 约束

- **实现检查 (A2A3/A5)**:
    - `TileData::DType` 必须与标量模板参数 `T` 的类型完全相同。
    - `dst`/`scalar` 元素类型必须相同，且必须是以下之一：`int32_t`、`uint32_t`、`int16_t`、`uint16_t`。
    - `TileData::Cols != 1`（此为实现强制执行的条件）。
- **有效区域**:
    - 实现使用 `dst.GetValidCol()` 作为序列长度，不参考 `dst.GetValidRow()`。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, int32_t, 1, 16>;
  TileT dst;
  TCI<TileT, int32_t, /*descending=*/0>(dst, /*S=*/0);
}
```

### 手动（Manual）

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

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tci %scalar {descending = false} : dtype -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tci %S {descending = false} : !pto.tile<...>
# AS Level 2 (DPS)
pto.tci ins(%scalar {descending = false} : dtype) outs(%dst : !pto.tile_buf<...>)
```

