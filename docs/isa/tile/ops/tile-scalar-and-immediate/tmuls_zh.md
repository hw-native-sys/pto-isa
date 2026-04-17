# pto.tmuls

`pto.tmuls` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述

把一个标量逐元素乘到 tile 上。

## 机制

对目标 tile 的有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \cdot \mathrm{scalar} $$

标量逻辑上广播到整个 valid region。

## 语法

同步形式：

```text
%dst = tmuls %src, %scalar : !pto.tile<...>, f32
```

### AS Level 1（SSA）

```text
%dst = pto.tmuls %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tmuls ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TMULS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar, WaitEvents &... events);
```

## 输入

- `src`：源 tile
- `scalar`：广播到所有元素的标量
- `dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出

- `dst`：逐元素乘法后的结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- `TileData::DType` 必须属于：`int32_t`、`int`、`int16_t`、`half`、`float16_t`、`float`、`float32_t`
- tile 位置必须是向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`src0.GetValidRow() == dst.GetValidRow()` 且 `src0.GetValidCol() == dst.GetValidCol()`
- tile 布局必须是行主序

### A5

- `TileData::DType` 必须属于：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`、`half`、`float`、`bfloat16_t`
- tile 位置必须是向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`src0.GetValidCol() == dst.GetValidCol()`
- tile 布局必须是行主序

## 性能

### A2A3

仓内 `include/pto/costmodel/pto_isa_costmodel.hpp` 直接给出 `TMULS -> vmuls` 对应桶：

| 指标 | INT16 / INT32 | FP16 / FP32 |
| --- | --- | --- |
| 启动时延 | 14 (`A2A3_STARTUP_BINARY`) | 14 (`A2A3_STARTUP_BINARY`) |
| 完成时延 | 18 (`A2A3_COMPL_INT_MUL`) | 20 (`A2A3_COMPL_FP_MUL`) |
| 每次 repeat 吞吐 | 1 (`A2A3_RPT_1`) | 1 (`A2A3_RPT_1`) |
| 流水间隔 | 18 (`A2A3_INTERVAL`) | 18 (`A2A3_INTERVAL`) |

### A5

当前手册未单列 `tmuls` 的独立周期条目，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TMULS(dst, src, 2.0f);
}
```

## 相关页面

- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.tdivs](./tdivs_zh.md)
- 下一条指令：[pto.tfmods](./tfmods_zh.md)
