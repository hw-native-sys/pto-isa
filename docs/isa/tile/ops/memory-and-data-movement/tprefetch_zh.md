# pto.tprefetch

`pto.tprefetch` 属于[内存与数据搬运](../../memory-and-data-movement_zh.md)指令集。

## 概述

把后续可能会访问的一段 `GlobalTensor` 数据提前搬进 tile 本地缓冲，用作预取或预热。

## 机制

若把 `src` 的当前视图看成二维切片，则：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{r_0 + i,\; c_0 + j} $$

和 `TLOAD` 不同，这条指令的重点不是复杂布局转换，而是把“稍后会用到的数据”尽早拉近。它在当前仓库实现里并不是“可以完全忽略的 hint bit”，而是真会填充 `dst`。

## 语法

同步形式：

```text
%dst = tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData &dst, GlobalData &src);
```

与大多数 PTO C++ 接口不同，这条封装不会自动执行 `TSYNC(events...)`。

## 输入

- `src`：源 `GlobalTensor`
- `dst`：目标 tile 缓冲

## 预期输出

- `dst`：被预取进本地路径的数据

## 副作用

这条指令可能会从 GM 读取。某些 target 可能会把它当提示，也可能会直接完成缓冲填充。

## 约束

- 可移植代码应把它用于“提前搬运即将访问的数据”，而不要把它当 `TLOAD` 的语义等价替代品。
- 在当前仓库实现里，预取范围仍然受 `dst` 大小和 `src` 切片大小影响。

## Target-Profile 限制

### CPU

- CPU 会直接按 `dst.GetValidRow()` / `dst.GetValidCol()` 做逐元素拷贝

### A2A3 / A5

- 若单个切片能放进 `dst`，则一次性预取
- 若放不下，则按 `dst` 容量分块预取

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能

当前手册未单列 `tprefetch` 的公开周期表。它的真正收益更依赖后续访存局部性，而不是单条指令本身的独立算术成本。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T, typename GT>
void example(Tile<TileType::Vec, T, 16, 16>& tileBuf, GT& globalView) {
  TPREFETCH(tileBuf, globalView);
}
```

## 相关页面

- [TLOAD](./tload_zh.md)
- [内存与数据搬运](../../memory-and-data-movement_zh.md)
