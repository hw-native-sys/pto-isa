<!-- Generated from `docs/isa/TPOP_zh.md` -->

# TPOP

## 指令示意图

![TPOP tile operation](../figures/isa/TPOP.svg)

## 简介

从 pipe 或 FIFO 的消费者端弹出一个 Tile。

## 数学语义

语义随指令而变化。 Unless stated otherwise, behavior is defined over the destination valid region.

## 汇编语法

PTO-AS 形式：参见 `docs/assembly/PTO-AS.md`.

### AS Level 1（SSA）

```text
%dst = pto.tpop ...
```

### AS Level 2（DPS）

```text
pto.tpop ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`.

## 约束

Refer to backend-specific legality checks for data type/layout/location/shape constraints.

## 示例

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.
