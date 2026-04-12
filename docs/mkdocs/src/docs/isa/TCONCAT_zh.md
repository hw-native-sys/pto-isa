<!-- Generated from `docs/isa/TCONCAT_zh.md` -->

# TCONCAT

## 指令示意图

![TCONCAT tile operation](../figures/isa/TCONCAT.svg)

## 简介

沿列维将两个源 Tile 拼接到目标 Tile。

## 数学语义

语义随指令而变化。 Unless stated otherwise, behavior is defined over the destination valid region.

## 汇编语法

PTO-AS 形式：参见 `docs/assembly/PTO-AS.md`.

### AS Level 1（SSA）

```text
%dst = pto.tconcat ...
```

### AS Level 2（DPS）

```text
pto.tconcat ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`.

## 约束

Refer to backend-specific legality checks for data type/layout/location/shape constraints.

## 示例

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.
