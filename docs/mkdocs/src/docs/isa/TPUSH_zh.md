<!-- Generated from `docs/isa/TPUSH_zh.md` -->

# TPUSH

## 指令示意图

![TPUSH tile operation](../figures/isa/TPUSH.svg)

## 简介

将 Tile 推入 pipe 或 FIFO 的生产者端。

## 数学语义

语义随指令而变化。 Unless stated otherwise, behavior is defined over the destination valid region.

## 汇编语法

PTO-AS 形式：参见 `docs/assembly/PTO-AS.md`.

### AS Level 1（SSA）

```text
%dst = pto.tpush ...
```

### AS Level 2（DPS）

```text
pto.tpush ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`.

## 约束

Refer to backend-specific legality checks for data type/layout/location/shape constraints.

## 示例

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.
