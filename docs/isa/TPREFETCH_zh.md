# TPREFETCH

## 指令示意图

![TPREFETCH tile operation](../figures/isa/TPREFETCH.svg)

## 简介

将数据从全局内存预取到 Tile 本地缓存/缓冲区（提示）。

## 数学语义

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = tprefetch %src : !pto.global<...> -> !pto.tile<...>
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

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData &dst, GlobalData &src);
```

## 约束

- 语义和缓存行为由目标/实现定义。
- 某些目标可能会忽略预取，将其视为提示。

## 示例

参见 `docs/isa/` 和 `docs/coding/tutorials/` 中的相关示例。

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
# AS Level 2 (DPS)
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

