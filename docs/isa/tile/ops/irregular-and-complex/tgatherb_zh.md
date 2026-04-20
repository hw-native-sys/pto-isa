# pto.tgatherb

`pto.tgatherb` 属于[不规则与复杂](../../irregular-and-complex_zh.md)指令集。

## 概述

使用字节偏移量收集元素。对每个元素在有效区域内，满足 `dst_{i,j} = *(srcBase + offset_{i,j})`。确切的边界行为由实现定义。

## 机制

对每个元素在有效区域内：

$$ \mathrm{dst}_{i,j} = *\left(\mathrm{srcBase} + \mathrm{offset}_{i,j}\right) $$

确切的边界行为由实现定义。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = tgatherb %src, %offsets : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tgatherb %src, %offsets : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tgatherb ins(%src, %offsets : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset, typename... WaitEvents>
PTO_INST RecordEvent TGATHERB(TileDataDst &dst, TileDataSrc &src, TileDataOffset &offset, WaitEvents & ... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| dst | 输出 Tile | 目标 Tile |
| src | 输入 Tile | 数据源 Tile（基地址） |
| offset | 输入 Tile | 字节偏移量 Tile，元素类型为 `uint32_t` |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| dst | Tile | 按字节偏移收集后的 Tile |

## 副作用

偏移量边界不通过显式运行时断言进行验证；超出范围的偏移行为由目标定义。

## 约束

A2/A3：目标布局必须为行主序（`TileDataDst::isRowMajor`），目标元素大小必须是 `1`、`2` 或 `4` 字节，`SrcTileData::DType`/`DstTileData::DType` 必须是 `int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t` 或 `float` 之一。

A5：目标元素大小必须是 `1`、`2` 或 `4` 字节，`SrcTileData::DType`/`DstTileData::DType` 必须是 `int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t` 或 `float` 之一。

偏移解释：偏移量被实现解释为 `uint32_t` 值（字节偏移），偏移量边界不通过显式运行时断言进行验证；超出范围的偏移由目标定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 目标布局要求 | - | 行主序 | - |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, uint8_t, 1, 256>;
  using OffT = Tile<TileType::Vec, uint32_t, 1, 256>;
  using DstT = Tile<TileType::Vec, uint8_t, 1, 256>;
  SrcT src;
  OffT off;
  DstT dst;
  TGATHERB(dst, src, off);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, uint8_t, 1, 256>;
  using OffT = Tile<TileType::Vec, uint32_t, 1, 256>;
  using DstT = Tile<TileType::Vec, uint8_t, 1, 256>;
  SrcT src;
  OffT off;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(off, 0x2000);
  TASSIGN(dst, 0x3000);
  TGATHERB(dst, src, off);
}
```

### PTO-AS

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tgatherb %src, %offsets : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# 手动模式：先显式绑定资源，再发射指令。
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tgatherb %src, %offsets : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tgatherb ins(%src, %offsets : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- [不规则与复杂指令集](../../irregular-and-complex_zh.md)
- [TGATHER](./tgather_zh.md)

![TGATHERB tile operation](../../../../figures/isa/TGATHERB.svg)
