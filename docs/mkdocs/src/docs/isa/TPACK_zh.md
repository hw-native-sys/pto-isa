<!-- Generated from `docs/isa/TPACK_zh.md` -->

# TPACK

## 指令示意图

![TPACK tile operation](../figures/isa/TPACK.svg)

## 简介

将 Tile 元素打包或转换为更窄的目标表示。

## 数学语义

语义随指令而变化。 Unless stated otherwise, behavior is defined over the destination valid region.

## 汇编语法

PTO-AS 形式：参见 `docs/assembly/PTO-AS.md`.

### AS Level 1（SSA）

```text
%dst = pto.tpack ...
```

### AS Level 2（DPS）

```text
pto.tpack ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`.

## 约束

Refer to backend-specific legality checks for data type/layout/location/shape constraints.

## 示例

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.
