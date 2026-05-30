# TSCATTER

## 指令示意图

![TSCATTER tile operation](../../../../figures/isa/TSCATTER.svg)

## 简介

`TSCATTER` 提供两种操作模式：

- 索引散播：按索引 Tile 给出的目标偏移，把源 Tile 中的元素分散写入目标 Tile。
- 掩码散播：按掩码模式把源元素写入更宽的目标 Tile，并把未选中的位置填零。

索引散播适合表达“不规则写回到本地 Tile”这类模式：数据仍然留在 tile 空间里，但目的位置不再由规则的行列映射决定。

和规则搬运不同，`TSCATTER` 的关键输入不是另一个 shape，而是 `indexes`。每个索引元素都表示目标 Tile 在线性存储视角下的一个元素偏移。

## 数学语义

设：

- `R = indexes.GetValidRow()`
- `C = indexes.GetValidCol()`

当前实现会先把整个 `dst` Tile 清零，然后对 `0 <= i < R`、`0 <= j < C` 执行：

$$ \mathrm{dst\_flat}_{\mathrm{indexes}_{i,j}} = \mathrm{src}_{i,j} $$

其中 `dst_flat` 表示把目标 Tile 按其存储顺序看成一段线性序列。对标准 row-major Tile 来说，这就是“写到扁平化后的第 `k` 个元素”。

这条语义有三个读者容易忽略的点：

- `TSCATTER` 按 `indexes` 的 valid region 遍历，而不是按 `dst` 的 valid region 遍历。
- 没有被任何索引命中的目标位置，在当前实现里保持为零值，而不是保留调用前的 `dst` 内容。
- 如果多个源元素落到同一个目标位置，最终值属于实现定义行为；当前实现按行优先顺序遍历，因此后写入的元素会覆盖前值。

### 掩码散播

掩码散播是 A5-only 重载。它把每个源元素写入扩展后目标分组中的一个选中 lane，并把同一分组中其他 lane 填零。

对于源元素 `(i, j)` 和掩码模式 `P`：

$$ \mathrm{dst}_{i,\ F_P \cdot j + \mathrm{pos}_P} = \mathrm{src}_{i,j} $$

其中 `F_P` 是扩展倍数：`P1111` 为 1，`P0101`/`P1010` 为 2，`P0001`/`P0010`/`P0100`/`P1000` 为 4。同一扩展分组中未选中的列会被写为 0。

## 汇编语法

PTO-AS 形式：参见 [汇编写法与操作数](../../../syntax-and-operands/assembly-model_zh.md)。

同步形式：

```text
%dst = tscatter %src, %idx : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tscatter %src, %idx : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataD, typename TileDataS, typename TileDataI, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(TileDataD &dst, TileDataS &src, TileDataI &indexes, WaitEvents &... events);

template <MaskPattern maskPattern = MaskPattern::P1111, typename DstTileData, typename SrcTileData,
          typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

`MaskPattern` 定义于 `include/pto/common/type.hpp`：

| 值 | 模式 | 选中 lane | 扩展倍数 |
| --- | --- | --- | --- |
| `P0101` | `0101...` | 每 2-lane 分组的第一个 lane | x2 |
| `P1010` | `1010...` | 每 2-lane 分组的第二个 lane | x2 |
| `P0001` | `0001...` | 每 4-lane 分组的第一个 lane | x4 |
| `P0010` | `0010...` | 每 4-lane 分组的第二个 lane | x4 |
| `P0100` | `0100...` | 每 4-lane 分组的第三个 lane | x4 |
| `P1000` | `1000...` | 每 4-lane 分组的第四个 lane | x4 |
| `P1111` | `1111...` | 全部 lane，等同于 `TMOV` | x1 |

## 约束

!!! warning "约束"
    ### 通用约束

    - `dst`、`src`、`indexes` 都必须是 `TileType::Vec`。
    - `dst` 与 `src` 的元素类型必须完全一致。
    - `indexes` 必须是整型 Tile，且索引元素宽度必须和数据元素宽度匹配：
      - 数据为 4 字节时，索引也必须是 4 字节；
      - 数据为 2 字节时，索引也必须是 2 字节；
      - 数据为 1 字节时，索引必须是 2 字节。
    - 实现按 `indexes.GetValidRow()` / `indexes.GetValidCol()` 遍历。可移植代码应保证 `src` 在同一坐标域内可读。
    - 当前 backend 不会对索引做越界检查。超出目标 Tile 线性范围的索引不属于合法使用域。
    - 编译期 valid bounds 必须满足：
      - `TileDataD::ValidRow <= TileDataD::Rows`
      - `TileDataD::ValidCol <= TileDataD::Cols`
      - `TileDataS::ValidRow <= TileDataS::Rows`
      - `TileDataS::ValidCol <= TileDataS::Cols`
      - `TileDataI::ValidRow <= TileDataI::Rows`
      - `TileDataI::ValidCol <= TileDataI::Cols`

    ### A2/A3 实现检查

    - `dst/src` 的元素类型必须属于：
      `int32_t`、`int16_t`、`int8_t`、`uint32_t`、`uint16_t`、`uint8_t`、`half`、`float32_t`、`bfloat16_t`。
    - `indexes` 的元素类型必须属于：
      `int16_t`、`int32_t`、`uint16_t`、`uint32_t`。

    ### A5 实现检查

    - A5 上的类型限制与 A2/A3 相同：
      - `dst/src` 支持 `int32_t`、`int16_t`、`int8_t`、`uint32_t`、`uint16_t`、`uint8_t`、`half`、`float32_t`、`bfloat16_t`
      - `indexes` 支持 `int16_t`、`int32_t`、`uint16_t`、`uint32_t`

    ### 掩码散播检查（仅 A5）

    - `DstTileData::Loc` 与 `SrcTileData::Loc` 必须是 `TileType::Vec`。
    - `DstTileData::DType` 与 `SrcTileData::DType` 必须一致，且属于 TSCATTER 支持的数据类型。
    - `maskPattern` 必须是 `P0101`、`P1010`、`P0001`、`P0010`、`P0100`、`P1000` 或 `P1111`。
    - `SrcTileData::ValidRow` 必须等于 `DstTileData::ValidRow`。
    - `DstTileData::ValidCol` 必须等于 `SrcTileData::ValidCol * F_P`，其中 `F_P` 为对应掩码的扩展倍数。

!!! warning "掩码散播填零"
    A5 掩码散播在写入源元素前，会把目标 Tile 的整个 UB 分配区域初始化为 0，而不只限于 valid region。代码不得让该目标 UB 分配区域与其他仍然活跃的数据重叠。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TSCATTER(dst, src, idx);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(idx, 0x3000);
  TSCATTER(dst, src, idx);
}
```

### 掩码散播

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_mask_scatter() {
  using SrcTileT = Tile<TileType::Vec, half, 16, 64>;
  using DstTileT = Tile<TileType::Vec, half, 16, 128>;
  SrcTileT src;
  DstTileT dst;
  TSCATTER<MaskPattern::P1010>(dst, src);
}
```

## 相关页面

- [不规则与复杂指令集](../../irregular-and-complex_zh.md)
- [布局参考](../../../state-and-types/layout_zh.md)
- [数据格式](../../../state-and-types/data-format_zh.md)
