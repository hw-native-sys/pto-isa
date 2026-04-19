# pto.texpands

`pto.texpands` 属于[逐元素 Tile-标量](./tile/tile-scalar-and-immediate_zh.md)指令集。

## 概述

将标量广播到目标 Tile 中所有有效位置。

## 机制

对有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{scalar} $$

对于向量 Tile，迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定；对于 Mat Tile，迭代域由 `TileData::Rows` / `TileData::Cols` 决定。

## 语法

### PTO-AS

```text
%dst = texpands %scalar : f32, !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.texpands %scalar : dtype -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.texpands ins(%scalar : dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TEXPANDS(TileData &dst, typename TileData::DType scalar, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%scalar` | 标量 | 广播到目标 tile 的值 |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | 有效区域内所有元素等于标量值 |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- Tile 位置可以是向量或 Mat。
- A2/A3 向量支持：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。
- A5 向量支持：`uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`uint32_t`、`int32_t`、`half`、`float`。
- A2/A3 Mat 要求：`TileData::Rows * TileData::Cols * sizeof(T) / 32` 在 `[1, 32767]` 范围内。
- A5 向量静态有效边界：`TileData::ValidRow <= TileData::Rows` 且 `TileData::ValidCol <= TileData::Cols`。
- A5 Mat 约束因布局而异。

## 异常与非法情形

- 不支持的元素类型会被 verifier 拒绝。
- 所选 target profile 不支持的形状/布局约束会被后端拒绝。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `f32` | Simulated | Supported | Supported |
| `f16` | Simulated | Supported | Supported |
| `bf16` | Simulated | Supported | No |
| `i32 / u32` | Simulated | Supported | Supported |
| `i16 / u16` | Simulated | Supported | Supported |
| `i8 / u8` | Simulated | Supported | Supported |
| Vec Layout | Any | Any | RowMajor |
| Mat Layout | Any | Supported | Supported |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT dst;
  TEXPANDS(dst, 0.0f);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT dst;
  TASSIGN(dst, 0x1000);
  TEXPANDS(dst, 0.0f);
}
```

### PTO-AS

```text
%dst = texpands %scalar : f32, !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.texpands ins(%scalar : dtype) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[逐元素 Tile-标量](./tile/tile-scalar-and-immediate_zh.md)
