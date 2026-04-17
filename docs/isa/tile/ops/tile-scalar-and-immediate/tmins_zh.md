# pto.tmins

`pto.tmins` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述

对 tile 和标量逐元素取最小值。

## 机制

对目标 tile 的有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \min(\mathrm{src}_{i,j}, \mathrm{scalar}) $$

标量会广播到整个 valid region。

## 语法

同步形式：

```text
%dst = tmins %src, %scalar : !pto.tile<...>, f32
```

### AS Level 1（SSA）

```text
%dst = pto.tmins %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tmins ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TMINS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar, WaitEvents &... events);
```

## 输入

- `src`：源 tile
- `scalar`：广播到所有元素的标量
- `dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出

- `dst`：逐元素最小值结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 与 `src` 必须使用相同元素类型。
- 标量类型必须与 tile 数据类型匹配。
- tile 位置必须是向量 tile。
- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- `TileData::DType` 必须属于：`int32_t`、`int`、`int16_t`、`half`、`float16_t`、`float`、`float32_t`
- 运行时要求：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`

### A5

- `TileData::DType` 必须属于：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`、`half`、`float`、`bfloat16_t`
- 运行时要求：`src.GetValidCol() == dst.GetValidCol()`

## 性能

### A2A3

仓内 `include/pto/costmodel/pto_isa_costmodel.hpp` 直接给出 `TMINS -> vmins` 对应桶：

| 指标 | INT16 / INT32 | FP16 / FP32 |
| --- | --- | --- |
| 启动时延 | 14 (`A2A3_STARTUP_BINARY`) | 14 (`A2A3_STARTUP_BINARY`) |
| 完成时延 | 17 (`A2A3_COMPL_INT_BINOP`) | 17 (`A2A3_COMPL_INT_BINOP`) |
| 每次 repeat 吞吐 | 1 (`A2A3_RPT_1`) | 1 (`A2A3_RPT_1`) |
| 流水间隔 | 18 (`A2A3_INTERVAL`) | 18 (`A2A3_INTERVAL`) |

### A5

当前手册未单列 `tmins` 的独立周期条目，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TMINS(dst, src, 0.0f);
}
```

## 相关页面

- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.tsels](./tsels_zh.md)
- 下一条指令：[pto.tadds](./tadds_zh.md)
