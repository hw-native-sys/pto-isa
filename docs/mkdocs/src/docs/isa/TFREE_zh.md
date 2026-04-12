<!-- Generated from `docs/isa/TFREE_zh.md` -->

# TFREE

## 指令示意图

![TFREE tile operation](../figures/isa/TFREE.svg)

## 简介

将当前占用的 pipe 或 FIFO 槽位释放回生产者。

## 数学语义

语义随指令而变化。 Unless stated otherwise, behavior is defined over the destination valid region.

## 汇编语法

PTO-AS 形式：参见 `docs/assembly/PTO-AS.md`.

### AS Level 1（SSA）

```text
%dst = pto.tfree ...
```

### AS Level 2（DPS）

```text
pto.tfree ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`.

## 约束

Refer to backend-specific legality checks for data type/layout/location/shape constraints.

## 示例

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.
