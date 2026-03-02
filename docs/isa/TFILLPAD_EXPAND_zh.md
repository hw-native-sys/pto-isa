# TFILLPAD_EXPAND

## 指令示意图

![TFILLPAD_EXPAND tile operation](../figures/isa/TFILLPAD_EXPAND.svg)

## 简介

填充/填充时允许目标大于源。

## 数学语义

除非另有说明, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

### IR Level 1（SSA）

```text
%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tfillpad_expand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_EXPAND(DstTileData &dst, SrcTileData &src,
                            WaitEvents&... events);
```

## 约束

Type/layout/location/shape legality is backend-dependent; treat implementation-specific notes as normative for that backend.

## 示例

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.tfillpad_expand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

