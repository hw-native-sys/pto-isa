# TTRI

## 指令示意图

![TTRI tile operation](../figures/isa/TTRI.svg)

## 简介

生成三角（下/上）掩码 Tile。

## 数学语义

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `d = diagonal`.

Lower-triangular (`isUpperOrLower=0`) conceptually produces:

$$
\mathrm{dst}_{i,j} = egin{cases}1 & j \le i + d \ 0 & 	ext{otherwise}\end{cases}
$$

Upper-triangular (`isUpperOrLower=1`) conceptually produces:

$$
\mathrm{dst}_{i,j} = egin{cases}0 & j < i + d \ 1 & 	ext{otherwise}\end{cases}
$$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

### IR Level 1（SSA）

```text
%dst = pto.ttri %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.ttri ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileData, int isUpperOrLower, typename... WaitEvents>
PTO_INST RecordEvent TTRI(TileData &dst, int diagonal, WaitEvents&... events);
```

## 约束

- `isUpperOrLower` must be `0` (lower) or `1` (upper).
- Destination tile must be row-major on some targets (see `include/pto/npu/*/TTri.hpp`).

## 示例

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.ttri %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.ttri %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = pto.ttri %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# IR Level 2 (DPS)
pto.ttri ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

