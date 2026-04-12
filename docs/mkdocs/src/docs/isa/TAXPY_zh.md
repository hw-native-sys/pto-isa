<!-- Generated from `docs/isa/TAXPY_zh.md` -->

# TAXPY

## 指令示意图

![TAXPY tile operation](../figures/isa/TAXPY.svg)

## 简介

AXPY 风格融合更新：将 Tile 乘以标量并累加到目标 Tile。

## 数学语义

语义随指令而变化。 Unless stated otherwise, behavior is defined over the destination valid region.

## 汇编语法

PTO-AS 形式：参见 `docs/assembly/PTO-AS.md`.

### AS Level 1（SSA）

```text
%dst = pto.taxpy ...
```

### AS Level 2（DPS）

```text
pto.taxpy ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`.

## 约束

Refer to backend-specific legality checks for data type/layout/location/shape constraints.

## 示例

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.
