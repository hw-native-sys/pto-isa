# pto.tands

`pto.tands` 属于[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)指令集。

## 概述

对 tile 和标量逐元素做按位与。

## 机制

`pto.tands` 作用在 tile payload 上，而不是标量控制状态。对目标 tile 的有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \;\&\; \mathrm{scalar} $$

标量会广播到整个 valid region。它本质上是 tile 版的按位掩码操作，适合做位域裁剪、标志位过滤和逐元素掩码化。

## 语法

同步形式：

```text
%dst = tands %src, %scalar : !pto.tile<...>, i32
```

### AS Level 1（SSA）

```text
%dst = pto.tands %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tands ins(%src, %scalar : !pto.tile_buf<...>, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TANDS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar, WaitEvents &... events);
```

## 输入

- `src`：源 tile
- `scalar`：广播到所有元素的标量掩码
- `dst`：目标 tile
- 迭代域：`dst` 的 valid row / valid col

## 预期输出

- `dst`：逐元素按位与结果 tile

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 操作迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 这条指令面向整数元素类型，不适用于浮点 tile。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。
- 程序不能依赖文档合法域之外的行为。

## Target-Profile 限制

### A2A3

- 面向整型元素类型。
- `dst` 与 `src` 必须使用相同元素类型。
- `dst` 与 `src` 必须是向量 tile。
- 运行时要求：`src.GetValidRow() == dst.GetValidRow()` 且 `src.GetValidCol() == dst.GetValidCol()`。
- 手动模式下，不支持把源 tile 与目标 tile 绑定到同一片内存。

### A5

- 面向 `TEXPANDS` 与 `TAND` 支持的整型元素类型。
- `dst` 与 `src` 必须使用相同元素类型。
- `dst` 与 `src` 必须是向量 tile。
- 手动模式下，不支持把源 tile 与目标 tile 绑定到同一片内存。

## 性能

当前仓内没有为 `tands` 单列公开 cost bucket。若代码依赖具体延迟，应把它视为目标 profile 相关的 tile 按位逻辑路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint16_t, 16, 16>;
  using TileSrc = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileDst dst;
  TileSrc src;
  TANDS(dst, src, 0xffu);
}
```

## 相关页面

- 指令集总览：[Tile-标量与立即数](../../tile-scalar-and-immediate_zh.md)
- 上一条指令：[pto.tmaxs](./tmaxs_zh.md)
- 下一条指令：[pto.tors](./tors_zh.md)
