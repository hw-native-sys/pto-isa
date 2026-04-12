<!-- Generated from `docs/isa/tile/ops/memory-and-data-movement/mscatter_zh.md` -->

# MSCATTER

## 指令示意图

![MSCATTER tile operation](../figures/isa/MSCATTER.svg)

## 简介

使用逐元素索引将 Tile 中的元素散播存储到全局内存。

## 数学语义

对源有效区域中的每个元素 `(i, j)`：

$$ \mathrm{mem}[\mathrm{idx}_{i,j}] = \mathrm{src}_{i,j} $$

如果多个元素映射到同一目标位置，最终值由实现定义（CPU 模拟器：按行主序迭代顺序，最后写入者获胜）。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
mscatter %src, %mem, %idx : !pto.memref<...>, !pto.tile<...>, !pto.tile<...>
```

### AS Level 1（SSA）

```text
pto.mscatter %src, %idx, %mem : (!pto.tile<...>, !pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### AS Level 2（DPS）

```text
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &... events);
```

## 约束

- **支持的数据类型**：
    - `src`/`dst` 的元素类型必须是以下之一：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。
    - 在 AICore 目标上，还支持 `float8_e4m3_t` 和 `float8_e5m2_t`。
    - `indexes` 的元素类型必须是 `int32_t` 或 `uint32_t`。
- **Tile 与内存类型约束**：
    - `src` 必须是向量 Tile（`TileType::Vec`）。
    - `indexes` 必须是向量 Tile（`TileType::Vec`）。
    - `src` 和 `indexes` 必须使用行主序布局。
    - `dst` 必须是位于 GM 内存中的 `GlobalTensor`。
    - `dst` 必须使用 `ND` 布局。
- **原子操作约束**：
    - 非原子 scatter 对所有受支持元素类型都可用。
    - `Add` 原子模式要求元素类型为 `int32_t`、`uint32_t`、`float` 或 `half`。
    - `Max`/`Min` 原子模式要求元素类型为 `int32_t` 或 `float`。
- **形状约束**：
    - `src.Rows == indexes.Rows`。
    - `indexes` 的形状必须为 `[N, 1]`（按行 scatter）或 `[N, M]`（按元素 scatter）。
    - `src` 的行宽必须满足 32 字节对齐，即 `src.Cols * sizeof(DType)` 必须是 32 的倍数。
    - `dst` 的静态 shape 必须满足 `Shape<1, 1, 1, TableRows, RowWidth>`。
- **索引解释**：
    - 索引解释由目标定义。CPU 模拟器将索引视为 `dst.data()` 中的线性元素索引。
    - CPU 模拟器不对 `indexes` 强制执行边界检查。

## 示例

参见 `docs/isa/` 和 `docs/coding/tutorials/` 中的相关示例。

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
# AS Level 2 (DPS)
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

