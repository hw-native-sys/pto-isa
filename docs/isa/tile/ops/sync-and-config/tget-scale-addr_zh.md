# pto.tget_scale_addr

`pto.tget_scale_addr` 属于[同步与配置指令](../../sync-and-config_zh.md)集。

## 概述

`TGET_SCALE_ADDR` 将输入 Tile 的片上地址数值按比例扩展，将其结果数值绑定为输出 Tile 的片上地址。这个扩展因子是由 `include/pto/npu/a5/utils.hpp` 中的右移值 `SHIFT_MX_ADDR` 定义的。

## 机制

地址映射关系为：

$$ \mathrm{Address}(\mathrm{dst}) = \mathrm{Address}(\mathrm{src}) \gg \mathrm{SHIFT\_MX\_ADDR} $$

该指令主要用于在自动模式下处理不同粒度的地址映射。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.tget_scale_addr %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tget_scale_addr ins(%src : !pto.tile<...>) outs(%dst : !pto.tile<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TGET_SCALE_ADDR(TileDataDst &dst, TileDataSrc &src, WaitEvents&... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `dst` | 输出 | 目标 Tile（缩放后地址） |
| `src` | 输入 | 源 Tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 片上地址按比例缩放后的 Tile |

## 副作用

将目标 Tile 的片上地址绑定为源 Tile 地址经右移 `SHIFT_MX_ADDR` 位后的值。

## 约束

- 输入和输出都必须为 Tile 对象。
- 目前只能用在自动模式下。

## 异常与非法情形

- 在手动模式下使用此指令，行为未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 自动模式支持 | - | - | 是 |

## 示例

### C++

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T, int ARows, int ACols, int BRows, int BCols>
void example() {
    using LeftTile = TileLeft<T, ARows, ACols>;
    using RightTile = TileRight<T, BRows, BCols>;

    using LeftScaleTile = TileLeftScale<T, ARows, ACols>;
    using RightScaleTile = TileRightScale<T, BRows, BCols>;

    LeftTile aTile;
    RightTile bTile;
    LeftScaleTile aScaleTile;
    RightScaleTile bScaleTile;

    TGET_SCALE_ADDR(aScaleTile, aTile);
    TGET_SCALE_ADDR(bScaleTile, bTile);
}
```

### PTO-AS

```text
%dst = pto.tget_scale_addr %src : !pto.tile<...> -> !pto.tile<...>
```

## 相关页面

- 指令集总览：[同步与配置](../../sync-and-config_zh.md)
- [TGET_SCALE_ADDR](./tget-scale-addr_zh.md)
