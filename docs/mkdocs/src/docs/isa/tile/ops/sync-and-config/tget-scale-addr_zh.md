<!-- Generated from `docs/isa/tile/ops/sync-and-config/tget-scale-addr_zh.md` -->

# TGET_SCALE_ADDR

## Tile Operation Diagram

![TGET_SCALE_ADDR tile operation](../figures/isa/TGET_SCALE_ADDR.svg)

## Introduction

将输入Tile的片上地址数值按比例扩展，将其结果数值绑定为输出Tile的片上地址。

这个扩展因子是由`include/pto/npu/a5/utils.hpp`中的右移值`SHIFT_MX_ADDR`定义的。

## 数学语义

Address(`dst`) = Address(`src`) >> `SHIFT_MX_ADDR`

## 汇编语法

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

### IR Level 1 (SSA)

TODO

### IR Level 2 (DPS)

TODO

## C++ 内建接口

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TGET_SCALE_ADDR(TileDataDst &dst, TileDataSrc &src, aitEvents&... events);
```

## 约束

- **输入和输出都必须为Tile对象**
- **目前只能用在auto模式下**（以后会将支持manual模式下的实现）

## 示例

```cpp
#include <pto/pto-inst.hpp>

> wa
using namespace pto;

template <typename T, int ARows, int ACols, BRows, BCols>
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

## asm form examples

### Auto Mode

TODO

### Manual Mode

TODO

### PTO Assembly Form

TODO
