<!-- Generated from `docs/isa/tile/ops/memory-and-data-movement/mgather_zh.md` -->

# MGATHER

## 指令示意图

![MGATHER tile operation](../figures/isa/MGATHER.svg)

## 简介

使用逐元素索引从全局内存收集加载元素到 Tile 中。

## 数学语义

对目标有效区域中的每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{mem}[\mathrm{idx}_{i,j}] $$

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = mgather %mem, %idx : !pto.memref<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNxdtype>, pto.tile<...>)
-> !pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

### AS Level 2（DPS）

```text
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src, TileInd &indexes, WaitEvents &... events);
```

## 约束

- **支持的数据类型**：
    - `dst`/`src` 的元素类型必须是以下之一：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`、`half`、`bfloat16_t`、`float`。
    - 在 AICore 目标上，还支持 `float8_e4m3_t` 和 `float8_e5m2_t`。
    - `indexes` 的元素类型必须是 `int32_t` 或 `uint32_t`。
- **Tile 与内存类型约束**：
    - `dst` 必须是向量 Tile（`TileType::Vec`）。
    - `indexes` 必须是向量 Tile（`TileType::Vec`）。
    - `dst` 和 `indexes` 必须使用行主序布局。
    - `src` 必须是位于 GM 内存中的 `GlobalTensor`。
    - `src` 必须使用 `ND` 布局。
- **形状约束**：
    - `dst.Rows == indexes.Rows`。
    - `indexes` 的形状必须为 `[N, 1]`（按行 gather）或 `[N, M]`（按元素 gather）。
    - `dst` 的行宽必须满足 32 字节对齐，即 `dst.Cols * sizeof(DType)` 必须是 32 的倍数。
    - `src` 的静态 shape 必须满足 `Shape<1, 1, 1, TableRows, RowWidth>`。
- **索引解释**：
    - 索引解释由目标定义。CPU 模拟器将索引视为 `src.data()` 中的线性元素索引。
    - CPU 模拟器不对 `indexes` 强制执行边界检查。

## 示例

参见 `docs/isa/` 和 `docs/coding/tutorials/` 中的相关示例。

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
# AS Level 2 (DPS)
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNxdtype>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

