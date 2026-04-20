# pto.tfmod

![TFMOD tile operation](../figures/isa/TFMOD.svg)

`pto.tfmod` 属于[逐元素 Tile-Tile](./tile/elementwise-tile-tile_zh.md)指令集。

## 概述

对两个 tile 做逐元素 `fmod` 运算，结果写入目标 tile。迭代域由目标 tile 的 valid region 决定。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{fmod}(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) $$

它表示浮点余数语义，余数符号与被除数（`src0`）相同。常用于周期折返、相位归一化和浮点余数路径。除零行为由目标 profile 定义。

## 语法

### PTO-AS

```text
%dst = tfmod %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tfmod %src0, %src1 : !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tfmod ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TFMOD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src0` | 左 tile | 被除数 tile，在 `dst` valid region 上逐坐标读取 |
| `%src1` | 右 tile | 除数 tile，在 `dst` valid region 上逐坐标读取 |
| `%dst` | 目标 tile | 接收逐元素 fmod 结果 |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | `dst` valid region 内的每个元素都等于 `fmod(src0, src1)` |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 除零行为由目标 profile 定义；CPU 模拟器在调试构建下会断言。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `int32_t` | Simulated | Supported | Supported |
| `float` | Simulated | Supported | Supported |
| `half` | Simulated | Supported | Supported |
| 布局 | Any | RowMajor only | RowMajor only |

`pto.tfmod` 在 CPU 仿真、A2/A3 和 A5 上保留相同的 PTO 可见语义，但具体支持子集仍取决于 profile。

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT out, a, b;
  TFMOD(out, a, b);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT a, b, dst;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(dst, 0x3000);
  TFMOD(dst, a, b);
}
```

### PTO-AS

```text
# 自动模式
%dst = pto.tfmod %src0, %src1 : !pto.tile<...>

# 手动模式
pto.tassign %arg0, @tile(0x1000)
pto.tassign %arg1, @tile(0x2000)
%dst = pto.tfmod %src0, %src1 : !pto.tile<...>

# PTO 汇编形式
%dst = tfmod %src0, %src1 : !pto.tile<...>
# AS Level 2 (DPS)
pto.tfmod ins(%src0, %src1 : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](./tile/elementwise-tile-tile_zh.md)
- 规范页：[pto.tfmod](./tile/ops/elementwise-tile-tile/tfmod_zh.md)
