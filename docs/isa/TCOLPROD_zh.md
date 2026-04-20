# pto.tcolprod

`pto.tcolprod` 属于[归约与扩展](./tile/reduce-and-expand_zh.md)指令集。

## 概述

通过跨行乘积来归约每一列。

## 机制

设 `R = src.GetValidRow()`，`C = src.GetValidCol()`。对 `0 <= j < C`：

$$ \mathrm{dst}_{0,j} = \prod_{i=0}^{R-1} \mathrm{src}_{i,j} $$

迭代域由 `src` 的 valid region 决定。若 `src.GetValidRow() == 0` 或 `src.GetValidCol() == 0`，实现会直接返回。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.tcolprod %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tcolprod ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLPROD(TileDataOut &dst, TileDataIn &src, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src` | 源 tile | 输入 tile |
| `%dst` | 目标 tile | 接收按列乘积结果 |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<1, C>` | 每列的 `R` 个元素乘积 |

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 和 `src` 必须为 `TileType::Vec`。
- `dst` 和 `src` 必须使用标准 ND 布局：行主且非分形（`BLayout::RowMajor`、`SLayout::NoneBox`）。
- `dst` 和 `src` 的元素类型必须一致。
- 运行时检查：`src.GetValidCol() == dst.GetValidCol()`。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `float` | Simulated | Supported | Supported |
| `half` | Simulated | Supported | Supported |
| `bfloat16_t` | — | — | Supported |
| `int16_t` / `uint16_t` | — | Supported | Supported |
| `int32_t` / `uint32_t` | — | Supported | Supported |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  SrcT src;
  DstT dst;
  TCOLPROD(dst, src);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TCOLPROD(dst, src);
}
```

### PTO-AS

```text
# 自动模式
%dst = pto.tcolprod %src : !pto.tile<...> -> !pto.tile<...>

# 手动模式
pto.tassign %arg0, @tile(0x1000)
pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcolprod %src : !pto.tile<...> -> !pto.tile<...>

# PTO 汇编形式
%dst = tcolprod %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcolprod ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[归约与扩展](./tile/reduce-and-expand_zh.md)
