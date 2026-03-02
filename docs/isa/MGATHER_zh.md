# MGATHER

## 指令示意图

![MGATHER tile operation](../figures/isa/MGATHER.svg)

## 简介

使用逐元素索引从全局内存收集加载元素到 Tile 中。

## 数学语义

对每个元素 `(i, j)` in the destination valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{mem}[\mathrm{idx}_{i,j}] $$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = mgather %mem, %idx : !pto.memref<...>, !pto.tile<...> -> !pto.tile<...>
```

### IR Level 1（SSA）

```text
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
-> !pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

### IR Level 2（DPS）

```text
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalData& src, TileInd& indexes, WaitEvents&... events);
```

## 约束

- Index interpretation is target-defined. The CPU simulator treats indices as linear element indices into `src.data()`.
- No bounds checks are enforced on `indexes` by the CPU simulator.

## 示例

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
```

### PTO 汇编形式

```text
%dst = mgather %mem, %idx : !pto.memref<...>, !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

