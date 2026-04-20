# pto.tprint

`pto.tprint` 属于[不规则与复杂](../../irregular-and-complex_zh.md)指令集。

## 概述

调试/打印 Tile 中的元素（实现定义）。从设备代码直接打印 Tile 或 GlobalTensor 的内容以用于调试目的。`TPRINT` 指令输出存储在 Tile 或 GlobalTensor 中的数据的逻辑视图。它支持常见的数据类型（例如 `float`、`half`、`int8`、`uint32`）和多种内存布局（GlobalTensor 的 `ND`、`DN`、`NZ`；片上缓冲区的向量 tiles）。

> **重要**：
> - 此指令**仅用于开发和调试**。
> - 它会产生**显著的运行时开销**，**不得在生产 kernel 中使用**。
> - 如果输出超过内部打印缓冲区，可能会被**截断**。可以通过在编译选项中添加 `-DCCEBlockMaxSize=16384` 来修改打印缓冲区，默认为 16KB。
> - **需要 CCE 编译选项 `-D_DEBUG --cce-enable-print`**。

## 机制

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

```text
tprint %src : !pto.tile<...> | !pto.global<...>
```

### AS Level 1（SSA）

```mlir
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
```

### AS Level 2（DPS）

```mlir
pto.tprint ins(%src : !pto.tile_buf<...> | !pto.partition_tensor_view<MxNxdtype>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
// 适用于打印 GlobalTensor 或 Vec 类型 Tile
template <typename TileData>
PTO_INST void TPRINT(TileData &src);

// 适用于打印 Acc 类型 Tile 和 Mat 类型 Tile（Mat 打印仅适用于 A3，A5 暂不支持）
template <typename TileData, typename GlobalData>
PTO_INTERNAL void TPRINT(TileData &src, GlobalData &tmp);
```

支持的 T 类型：Tile 的 TileType 必须是 `Vec`、`Acc`、`Mat`（仅 A3 支持），并具有支持的元素类型；GlobalTensor 必须使用布局 `ND`、`DN` 或 `NZ`，并具有支持的元素类型。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| src | 输入 Tile/GlobalTensor | 要打印的 Tile 或 GlobalTensor |
| tmp | 临时空间（仅 Mat/Acc） | 打印 Mat 或 Acc 类型 Tile 时需要传入 GM 上的临时空间 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| - | 控制台输出 | Tile 或 GlobalTensor 数据的逻辑视图 |

## 副作用

输出存储在 Tile 或 GlobalTensor 中的数据到控制台，产生显著的运行时开销。

## 约束

支持的元素类型：浮点数为 `float`、`half`；有符号整数为 `int8_t`、`int16_t`、`int32_t`；无符号整数为 `uint8_t`、`uint16_t`、`uint32_t`。

对于 GlobalTensor：布局必须是 `Layout::ND`、`Layout::DN` 或 `Layout::NZ` 之一。

对于临时空间：打印 `TileType` 为 `Mat` 或 `Acc` 的 Tile 时需要传入 GM 上的临时空间，临时空间不得小于 `TileData::Numel * sizeof(T)`。

A5 暂不支持 `TileType` 为 `Mat` 的 Tile 打印。

回显信息：`TileType` 为 `Mat` 时，布局将按照 `Layout::ND` 进行打印，其他布局可能会导致信息错位。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| Vec Tile 打印 | 支持 | 支持 | 支持 |
| Acc Tile 打印 | 支持 | 支持 | 支持 |
| Mat Tile 打印 | 支持 | 支持 | 不支持 |

## 示例

### C++ 打印 Tile

```cpp
#include <pto/pto-inst.hpp>

PTO_INTERNAL void DebugTile(__gm__ float *src) {
  using ValidSrcShape = TileShape2D<float, 16, 16>;
  using NDSrcShape = BaseShape2D<float, 32, 32>;
  using GlobalDataSrc = GlobalTensor<float, ValidSrcShape, NDSrcShape>;
  GlobalDataSrc srcGlobal(src);

  using srcTileData = Tile<TileType::Vec, float, 16, 16>;
  srcTileData srcTile;
  TASSIGN(srcTile, 0x0);

  TLOAD(srcTile, srcGlobal);
  TPRINT(srcTile);
}
```

### C++ 打印 GlobalTensor

```cpp
#include <pto/pto-inst.hpp>

PTO_INTERNAL void DebugGlobalTensor(__gm__ float *src) {
  using ValidSrcShape = TileShape2D<float, 16, 16>;
  using NDSrcShape = BaseShape2D<float, 32, 32>;
  using GlobalDataSrc = GlobalTensor<float, ValidSrcShape, NDSrcShape>;
  GlobalDataSrc srcGlobal(src);

  TPRINT(srcGlobal);
}
```

### PTO-AS

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
# 手动模式：先显式绑定资源，再发射指令。
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tprint %src : !pto.tile<...> | !pto.partition_tensor_view<MxNxdtype> -> ()
# AS Level 2 (DPS)
pto.tprint ins(%src : !pto.tile_buf<...> | !pto.partition_tensor_view<MxNxdtype>)
```

## 相关页面

- [不规则与复杂指令集](../../irregular-and-complex_zh.md)

![TPRINT tile operation](../../../../figures/isa/TPRINT.svg)
