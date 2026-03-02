# MSCATTER

## 指令示意图

![MSCATTER tile operation](../figures/isa/MSCATTER.svg)

## 简介

使用逐元素索引将 Tile 中的元素散播存储到全局内存。

## 数学语义

对每个元素 `(i, j)` in the source valid region:

$$ \mathrm{mem}[\mathrm{idx}_{i,j}] = \mathrm{src}_{i,j} $$

If multiple elements map to the same destination location, the final value is implementation-defined (CPU simulator: last writer wins in row-major iteration order).

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
mscatter %src, %mem, %idx : !pto.memref<...>, !pto.tile<...>, !pto.tile<...>
```

### IR Level 1（SSA）

```text
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### IR Level 2（DPS）

```text
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData& dst, TileSrc& src, TileInd& indexes, WaitEvents&... events);
```

## 约束

- Index interpretation is target-defined. The CPU simulator treats indices as linear element indices into `dst.data()`.
- No bounds checks are enforced on `indexes` by the CPU simulator.

## 示例

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### PTO 汇编形式

```text
mscatter %src, %mem, %idx : !pto.memref<...>, !pto.tile<...>, !pto.tile<...>
# IR Level 2 (DPS)
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

