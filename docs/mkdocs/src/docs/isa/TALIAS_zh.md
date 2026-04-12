<!-- Generated from `docs/isa/TALIAS_zh.md` -->

# TALIAS

## 指令示意图

![TALIAS tile operation](../figures/isa/TALIAS.svg)

## 简介

创建一个与原始 Tile 共享底层存储的别名视图。

## 数学语义

语义随指令而变化。 Unless stated otherwise, behavior is defined over the destination valid region.

## 汇编语法

PTO-AS 形式：参见 `docs/assembly/PTO-AS.md`.

### AS Level 1（SSA）

```text
%dst = pto.talias ...
```

### AS Level 2（DPS）

```text
pto.talias ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`.

## 约束

Refer to backend-specific legality checks for data type/layout/location/shape constraints.

## 示例

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.
