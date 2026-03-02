# TPREFETCH

## 指令示意图

![TPREFETCH tile operation](../figures/isa/TPREFETCH.svg)

## 简介

将数据从全局内存预取到 Tile 本地缓存/缓冲区（提示）。

## 数学语义

除非另有说明, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData &dst, GlobalData &src);
```

## 约束

- Semantics and caching behavior are target/implementation-defined.
- Some targets may ignore prefetches or treat them as hints.

## 示例

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tprefetch %src : !pto.global<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

