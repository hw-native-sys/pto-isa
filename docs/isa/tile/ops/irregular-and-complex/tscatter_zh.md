# pto.tscatter

`pto.tscatter` 属于[不规则与复杂](../../irregular-and-complex_zh.md)指令集。

## 概述

`TSCATTER` 按索引 Tile 给出的目标偏移，把源 Tile 中的元素分散写入目标 Tile。它适合表达“不规则写回到本地 Tile”这类模式：数据仍然留在 tile 空间里，但目的位置不再由规则的行列映射决定。和规则搬运不同，`TSCATTER` 的关键输入不是另一个 shape，而是 `indexes`。每个索引元素都表示目标 Tile 在线性存储视角下的一个元素偏移。

## 机制

设 `R = indexes.GetValidRow()`，`C = indexes.GetValidCol()`。当前实现会先把整个 `dst` Tile 清零，然后对 `0 <= i < R`、`0 <= j < C` 执行：

$$ \mathrm{dst\_flat}_{\mathrm{indexes}_{i,j}} = \mathrm{src}_{i,j} $$

其中 `dst_flat` 表示把目标 Tile 按其存储顺序看成一段线性序列。对标准 row-major Tile 来说，这就是“写到扁平化后的第 `k` 个元素”。

有三个读者容易忽略的语义点：`TSCATTER` 按 `indexes` 的 valid region 遍历，而不是按 `dst` 的 valid region 遍历；没有被任何索引命中的目标位置，在当前实现里保持为零值，而不是保留调用前的 `dst` 内容；如果多个源元素落到同一个目标位置，最终值属于实现定义行为，当前实现按行优先顺序遍历，因此后写入的元素会覆盖前值。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = tscatter %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataD, typename TileDataS, typename TileDataI, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(TileDataD &dst, TileDataS &src, TileDataI &indexes, WaitEvents & ... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| dst | 输出 Tile | 目标 Tile |
| src | 输入 Tile | 源 Tile |
| indexes | 输入 Tile | 索引 Tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| dst | Tile | 按索引分散写入后的 Tile，先被清零 |

## 副作用

当前 backend 不会对索引做越界检查。超出目标 Tile 线性范围的索引不属于合法使用域。

## 约束

通用约束：`dst`、`src`、`indexes` 都必须是 `TileType::Vec`；`dst` 与 `src` 的元素类型必须完全一致；`indexes` 必须是整型 Tile，且索引元素宽度必须和数据元素宽度匹配（数据为 4 字节时索引也必须是 4 字节，数据为 2 字节时索引也必须是 2 字节，数据为 1 字节时索引必须是 2 字节）；实现按 `indexes.GetValidRow()` / `indexes.GetValidCol()` 遍历，可移植代码应保证 `src` 在同一坐标域内可读；当前 backend 不会对索引做越界检查，超出目标 Tile 线性范围的索引不属于合法使用域；编译期 valid bounds 必须满足 `TileDataD::ValidRow <= TileDataD::Rows`、`TileDataD::ValidCol <= TileDataD::Cols`、`TileDataS::ValidRow <= TileDataS::Rows`、`TileDataS::ValidCol <= TileDataS::Cols`、`TileDataI::ValidRow <= TileDataI::Rows`、`TileDataI::ValidCol <= TileDataI::Cols`。

A2/A3 实现检查：`dst/src` 的元素类型必须属于 `int32_t`、`int16_t`、`int8_t`、`uint32_t`、`uint16_t`、`uint8_t`、`half`、`float32_t`、`bfloat16_t`；`indexes` 的元素类型必须属于 `int16_t`、`int32_t`、`uint16_t`、`uint32_t`。

A5 实现检查：A5 上的类型限制与 A2/A3 相同，`dst/src` 支持 `int32_t`、`int16_t`、`int8_t`、`uint32_t`、`uint16_t`、`uint8_t`、`half`、`float32_t`、`bfloat16_t`，`indexes` 支持 `int16_t`、`int32_t`、`uint16_t`、`uint32_t`。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| dst/src 数据类型 | - | int32_t、int16_t、int8_t、uint32_t、uint16_t、uint8_t、half、float32_t、bfloat16_t | 同 A2/A3 |
| indexes 数据类型 | - | int16_t、int32_t、uint16_t、uint32_t | 同 A2/A3 |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TSCATTER(dst, src, idx);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(idx, 0x3000);
  TSCATTER(dst, src, idx);
}
```

### PTO-AS

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# 手动模式：先显式绑定资源，再发射指令。
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- [不规则与复杂指令集](../../irregular-and-complex_zh.md)
- [布局参考](../../../state-and-types/layout_zh.md)
- [数据格式](../../../state-and-types/data-format_zh.md)

![TSCATTER tile operation](../../../../figures/isa/TSCATTER.svg)
