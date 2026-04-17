# pto.tfmods

`pto.tfmods` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述

对 tile 和标量逐元素执行 `fmod`。

## 机制

对目标 tile 的有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{fmod}(\mathrm{src}_{i,j}, \mathrm{scalar}) $$

它作用在 tile payload 上，标量会广播到整个 valid region。

## 语法

同步形式：

```text
%dst = tfmods %src, %scalar : !pto.tile<...>, f32
```

### AS Level 1（SSA）

```text
%dst = pto.tfmods %src, %scalar : !pto.tile<...>, f32
```

### AS Level 2（DPS）

```text
pto.tfmods ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TFMODS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar, WaitEvents &... events);
```

## 输入

- `src`：源 tile
- `scalar`：广播到所有元素的标量
- `dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出

- `dst`：逐元素 `fmod` 结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 除零行为由目标平台定义；CPU 模拟器在调试构建下会断言。
- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

### A2A3

- `dst` 与 `src` 必须使用相同元素类型
- 支持元素类型：`float`、`float32_t`
- `dst` 与 `src` 必须是向量 tile 且为行主序
- 运行时要求：`dst.GetValidRow() == src.GetValidRow() > 0` 且 `dst.GetValidCol() == src.GetValidCol() > 0`

### A5

- `dst` 与 `src` 必须使用相同元素类型
- 支持元素类型是目标实现支持的 2 字节或 4 字节类型（包括 `half`、`float`）
- `dst` 与 `src` 必须是向量 tile
- 静态 valid 边界必须合法
- 运行时要求：`dst.GetValidRow() == src.GetValidRow()` 且 `dst.GetValidCol() == src.GetValidCol()`

## 性能

当前仓内没有为 `tfmods` 单列公开 cost bucket。若代码依赖具体延迟，应把它视为目标 profile 相关的余数 / 模路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TFMODS(out, x, 3.0f);
}
```

## 相关页面

- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.tmuls](./tmuls_zh.md)
- 下一条指令：[pto.trems](./trems_zh.md)
