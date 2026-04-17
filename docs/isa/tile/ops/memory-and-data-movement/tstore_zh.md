# pto.tstore

`pto.tstore` 属于[内存与数据搬运](../../memory-and-data-movement_zh.md)指令集。

## 概述

把 tile 中的数据写回 `GlobalTensor`（GM）。普通写回、原子写回以及当前实现中面向 `TileType::Acc` 的量化写回都在这一类接口里。

## 机制

若用带基址偏移的二维视角表示，可写成：

$$ \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{src}_{i,j} $$

真正的写回范围由源 tile 的 valid region 决定。

## 语法

同步形式：

```text
tstore %t1, %sv_out[%c0, %c0]
```

### AS Level 1（SSA）

```text
pto.tstore %src, %mem : (!pto.tile<...>, !pto.partition_tensor_view<MxNxdtype>) -> ()
```

### AS Level 2（DPS）

```text
pto.tstore ins(%src : !pto.tile_buf<...>) outs(%mem : !pto.partition_tensor_view<MxNxdtype>)
```

## C++ 内建接口

```cpp
template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &... events);

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, uint64_t preQuantScalar, WaitEvents &... events);

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData &dst, TileData &src, FpTileData &fp, WaitEvents &... events);
```

## 输入

- `src`：源 tile
- `dst`：目标 `GlobalTensor`
- `atomicType`：可选原子写回模式
- `preQuantScalar` / `fp`：当前实现中仅对 `TileType::Acc` 合法的量化 / fix-pipe 路径附加参数

## 预期输出

- 数据从 `src` 写回 `dst`
- 若使用原子模式，则在 GM 侧执行累加或比较类原子写回
- 若使用 `TSTORE_FP`，则通过 fix-pipe sideband state 路径写回

## 副作用

这条指令会写 GM。原子模式下，并发写入的结果还会依赖实现和内存一致性规则。

## 约束

- 写回范围由 `src.GetValidRow()` / `src.GetValidCol()` 决定
- 目标 `GlobalTensor` 的 shape / stride 必须允许这次写回

## Target-Profile 限制

### A2A3

- 源 tile 位置必须是 `TileType::Vec`、`TileType::Mat` 或 `TileType::Acc`
- 运行时要求：所有 `dst.GetShape(dim)` 和 `src.GetValidRow()/GetValidCol()` 都大于 0
- 对 `Vec/Mat`：
  - 支持类型：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、`half`、`bfloat16_t`、`float`
  - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`
  - 布局必须匹配 ND / DN / NZ，或满足单行 / 单列特殊情形
  - `int64_t/uint64_t` 只支持 `ND→ND` 与 `DN→DN`
- 对 `TileType::Acc`：
  - 目标布局必须为 ND 或 NZ
  - 源 dtype 必须为 `int32_t` 或 `float`
  - 不量化时，目标 dtype 必须为 `__gm__ int32_t/float/half/bfloat16_t`
  - shape 受 `Cols <= 4095`、`Rows` 上限等约束

### A5

- 源 tile 位置必须为 `TileType::Vec` 或 `TileType::Acc`
- `Vec` 路径支持更宽的元素类型集合，包括部分 float8 / packed float4 形式
- `Vec` 路径要求 `sizeof(TileData::DType) == sizeof(GlobalData::DType)`
- 布局需匹配 ND / DN / NZ，且额外存在行宽 / 列高字节对齐约束
- `Acc` 路径与 A2A3 类似，但 `AtomicAdd` 还会进一步限制目标 dtype

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能

当前仓内没有把 `tstore` 单列成完整公开周期表，但在 A2A3 的搬运带宽模型里，Vec → Vec / GM 路径通常按带宽估算。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename T>
void example_auto(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TSTORE(gout, t);
}
```

## 相关页面

- 指令集总览：[内存与数据搬运](../../memory-and-data-movement_zh.md)
- [一致性基线](../../../memory-model/consistency-baseline_zh.md)
