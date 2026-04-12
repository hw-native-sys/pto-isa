<!-- Generated from `docs/isa/tile/ops/memory-and-data-movement/tprefetch_zh.md` -->

# TPREFETCH

## 指令示意图

![TPREFETCH tile operation](../figures/isa/TPREFETCH.svg)

## 简介

将数据从全局内存预取到 Tile 本地缓存/缓冲区（提示）。

## 数学语义

除非另有说明, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## 汇编语法

PTO-AS 形式：参见 [PTO-AS Specification](../assembly/PTO-AS.md).

同步形式：

```text
%dst = tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

### AS Level 1（SSA）

```text
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData &dst, GlobalData &src);
```

## 约束

- Semantics and caching behavior are target/implementation-defined.
- Some targets may ignore prefetches or treat them as hints.

## 示例

See related examples in `docs/isa/` and `docs/coding/tutorials/`.
