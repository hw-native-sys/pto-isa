# pto.tors

`pto.tors` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对源 tile 的每个元素与一个立即数（标量）做按位或，结果写入目标 tile。迭代域由目标 tile 的 valid region 决定。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \;|\; \mathrm{scalar} $$

标量值在发射时广播到所有参与 lane；超出源 tile valid region 的坐标读到的值属于 implementation-defined。

## 语法

### PTO-AS

```text
%dst = tors %src, %scalar : !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tors ins(%src, %scalar : !pto.tile_buf<...>, dtype)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TORS(TileDataDst &dst, TileDataSrc &src,
                          typename TileDataDst::DType scalar, WaitEvents &... events);
```

## 输入

|| 操作数 | 角色 | 说明 |
|| --- | --- | --- |
|| `%src` | 源 tile | 在 `dst` valid region 上逐坐标读取 |
|| `%scalar` | 立即数 | 广播到所有 lane 的整数标量 |
|| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

|| 结果 | 类型 | 说明 |
|| --- | --- | --- |
|| `%dst` | `!pto.tile<...>` | `dst` valid region 内每个元素等于 `src | scalar` |

## 副作用

除产生目标 tile 外，没有额外架构副作用，不会隐式为无关 tile 流量建立栅栏。

## 约束

- **类型约束**：源 tile 和目标 tile 必须有相同元素类型，且均为整数类型。
- **布局约束**：源 tile 和目标 tile 必须有兼容布局。
- **有效区域**：迭代域总是 `dst.GetValidRow() × dst.GetValidCol()`。
- **手动模式**：不支持将源 tile 和目标 tile 设置为同一地址（禁止 in-place）。

## 异常与非法情形

- Verifier 拒绝类型不匹配。
- 后端拒绝不支持的元素类型、布局或目标 profile。
- 程序不能依赖 `dst` valid region 之外的值。

## Target-Profile 限制

|| 特性 | CPU Simulator | A2/A3 | A5 |
|| --- | :---: | :---: | :---: |
|| 整数类型 | Simulated | Supported | Supported |
|| 布局 | Any | RowMajor | RowMajor |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint16_t, 16, 16>;
  using TileSrc = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileDst dst;
  TileSrc src;
  TORS(dst, src, 0xffu);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_manual() {
  using TileDst = Tile<TileType::Vec, uint16_t, 16, 16>;
  using TileSrc = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileDst dst;
  TileSrc src;
  TASSIGN(src, 0x1000);
  TASSIGN(dst,  0x3000);
  TORS(dst, src, 0xffu);
}
```

### PTO-AS

```text
%dst = pto.tors %src, %scalar : (!pto.tile<...>, dtype) -> !pto.tile<...>
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 上一条指令：[pto.txors](./txors_zh.md)
- 下一条指令：[pto.tshls](./tshl_zh.md)
- 类似指令：[pto.tands](./tands_zh.md)、[pto.tor](./tor_zh.md)
