# pto.tdequant

`pto.tdequant` 属于[不规则与复杂](../../irregular-and-complex_zh.md) tile 指令族。

## 概述

使用按行广播的 `scale` 与 `offset` tile，把整数量化 tile 反量化为浮点 tile。

## 机制

`pto.tdequant` 把量化整数源 tile 恢复为浮点数值表示。在当前仓内实现里，目标 tile 是浮点型，源 tile 是整型，`scale` / `offset` 为“每行一个值”的参数 tile，并沿列广播。

对目标 valid region 内的每个 `(r, c)`：

$$ \mathrm{dst}_{r,c} = \left(\mathrm{src}_{r,c} - \mathrm{offset}_r\right) \cdot \mathrm{scale}_r $$

## 语法

### AS Level 1（SSA）

```text
%dst = pto.tdequant %src, %scale, %offset : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tdequant ins(%src, %scale, %offset : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
              outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent TDEQUANT(TileDataDst &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara &offset,
                              WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
|--------|------|------|
| `dst` | 目标 tile | 浮点输出 tile |
| `src` | 源 tile | 量化整数 tile |
| `scale` | 参数 tile | 每行一个 scale 值，并沿列广播 |
| `offset` | 参数 tile | 每行一个 offset 值，并沿列广播 |

## 预期输出

`dst` 在其 valid region 内保存反量化后的浮点结果。

## 副作用

除产生目标 tile 外，没有额外架构副作用。不会隐式建立与无关流量的 fence。

## 约束

- `dst` 与 `src` 的 valid row / valid col 必须一致。
- `scale` 和 `offset` 的行数必须与 `dst.GetValidRow()` 一致。
- 当前仓内实现中，`dst` 为 `float` / `float32_t`。
- 当前仓内实现中，`src` 为 `int8_t` 或 `int16_t`。
- 当前仓内实现中，`scale` / `offset` 使用与 `dst` 相同的浮点元素类型。
- 当前具体实现要求 row-major tile。

## 不允许的情形

- 使用不支持的类型组合、layout 或不匹配的 valid region。
- 把 `scale` / `offset` 当作按列变化的参数，除非目标 profile 另行文档化。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
|------|:-------------:|:-----:|:--:|
| `float <- int8_t` | Yes | Yes | Yes |
| `float <- int16_t` | Yes | Yes | Yes |
| row-major tile | Yes | Yes | Yes |

当前仓内在 CPU 仿真、A2/A3 和 A5 上都能找到具体实现。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using DstTile = Tile<TileType::Vec, float, 64, 64, BLayout::RowMajor>;
  using SrcTile = Tile<TileType::Vec, int8_t, 64, 64, BLayout::RowMajor>;
  using ParaTile = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor>;

  DstTile dst;
  SrcTile src;
  ParaTile scale;
  ParaTile offset;

  TDEQUANT(dst, src, scale, offset);
}
```

## 相关页面

- [不规则与复杂](../../irregular-and-complex_zh.md)
- [pto.tquant](./tquant_zh.md)
