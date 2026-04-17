# pto.tadd

`pto.tadd` 属于[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)指令集。

## 概述

对两个源 tile 做逐元素加法，结果写入目标 tile。迭代域由目标 tile 的 valid region 决定。

## 机制

对目标 tile 的 valid region 中每个 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{src1}_{i,j} $$

真正决定遍历范围的是目标 tile，不是两个源 tile。源 tile 会在相同坐标被读取；如果某个源 tile 在该坐标超出了自己的 valid region，那么这个位置读到的值属于 implementation-defined。

### 微操作映射

从 Tile Register File 的视角，这条指令可以理解为：

```text
TRF_READ(src0, i, j) -> A
TRF_READ(src1, i, j) -> B
A + B                -> C
TRF_WRITE(dst, i, j, C)
```

这一级不会直接暴露给手册读者，但它解释了为什么布局、stride 和目标 pipeline 会影响吞吐。

## 语法

### PTO-AS

```text
%dst = tadd %src0, %src1 : !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src0` | 左 tile | 在 `dst` valid region 上逐坐标读取 |
| `%src1` | 右 tile | 在 `dst` valid region 上逐坐标读取 |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | `dst` valid region 内的每个元素都等于 `src0 + src1` |

## 副作用

除产生目标 tile 外，没有额外架构副作用，不会隐式为无关 tile 流量建立栅栏。

## 约束

- `src0`、`src1` 和 `dst` 必须有相同元素类型。
- 布局必须彼此兼容，见[Tile 与 Valid Region](../../../programming-model/tiles-and-valid-regions_zh.md)。
- 迭代域总是 `dst.GetValidRow() × dst.GetValidCol()`。
- 目标 tile 的 TileType 决定这条指令落在哪类 pipeline 上执行。

## 异常与非法情形

- verifier 会拒绝源 / 目标类型不匹配。
- 后端会拒绝所选 target profile 不支持的元素类型、布局或 shape。
- 程序不能依赖 `dst` valid region 之外的值。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `f32` | Simulated | Supported | Supported |
| `f16` | Simulated | Supported | Supported |
| `bf16` | Simulated | Supported | Supported |
| `i32` | Simulated | Supported | Supported |
| `i16` | Simulated | Supported | Supported |
| `i8 / u8` | Simulated | No | Supported |
| `i64 / u64` | Simulated | No | No |
| `f8e4m3 / f8e5m2` | Simulated | No | Supported |
| 布局 | Any | RowMajor only | RowMajor only |

A2/A3 与 A5 当前都要求行主序布局才能走正式实现路径；A5 支持的元素类型更宽。

## 性能

### A2/A3

`TADD` 会落到 CCE 向量二元算术路径：

| 指标 | 数值 | 常量 |
| --- | --- | --- |
| 启动时延 | 14 cycles | `A2A3_STARTUP_BINARY` |
| 完成时延 | 19（FP）/ 17（INT） | `A2A3_COMPL_FP_BINOP` / `A2A3_COMPL_INT_BINOP` |
| 每次 repeat 吞吐 | 2 cycles | `A2A3_RPT_2` |
| 流水间隔 | 18 cycles | `A2A3_INTERVAL` |
| 周期模型 | `14 + C + 2R + (R-1)×18` | `R` 为 repeats |

连续路径下可粗略按 `R = validRow × validCol / 8` 估算 repeats。

### A5

当前手册未单列 `tadd` 的独立周期表，应视为目标 profile 相关。

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void add_tiles(Tile<Vec, float, 16, 16>& dst,
               Tile<Vec, float, 16, 16>& src0,
               Tile<Vec, float, 16, 16>& src1) {
    TADD(dst, src0, src1);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void add_tiles_manual(Tile<Vec, float, 16, 16>& dst,
                      Tile<Vec, float, 16, 16>& src0,
                      Tile<Vec, float, 16, 16>& src1) {
    TASSIGN(src0, 0x1000);
    TASSIGN(src1, 0x2000);
    TASSIGN(dst,  0x3000);
    TADD(dst, src0, src1);
}
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](../../elementwise-tile-tile_zh.md)
- 下一条指令：[pto.tabs](./tabs_zh.md)
- [Tile 与 Valid Region](../../../programming-model/tiles-and-valid-regions_zh.md)
