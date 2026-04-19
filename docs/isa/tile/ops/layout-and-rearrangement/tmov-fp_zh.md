# pto.tmov.fp

`pto.tmov.fp` 属于[布局与重排](../../layout-and-rearrangement_zh.md)指令集。

## 概述

`TMOV_FP` 是 `TMOV` 家族里的向量量化版本：它从累加器 Tile 读取源数据，同时使用额外的 `fp` Tile 提供量化参数，把结果移动到目标 Tile。这条指令解决的是"单纯按 dtype cast 不够"的问题。很多量化路径不仅需要源/目标类型，还需要一组外部缩放参数。`TMOV_FP` 把这组参数显式建模成一个 Tile 操作数。

## 机制

对有效区域中的每个元素，`TMOV_FP` 可以概念化为：

$$\mathrm{dst}_{i,j} = \mathrm{Convert}\!\left(\mathrm{src}_{i,j};\ \mathrm{fp}\right)$$

这里的 `Convert` 并不是单纯的 C++ 类型转换，而是"由 `fp` 指定量化 / 反量化参数"的目标相关转换。架构层可见的合同是：`src` 来自 `Acc` Tile，`fp` 提供向量量化所需的缩放信息，`dst` 接收量化或反量化后的结果。

## 语法

### PTO-AS

```text
%dst = tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tmov.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents &... events);
```

如果目标是 Vec Tile，并且还需要显式选择 `AccToVecMode`，则使用 `TMOV` 的同族模板重载而不是这个命名接口。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src` | 输入 Tile | 来自 `Acc` Tile 的源数据 |
| `fp` | 量化参数 Tile | 提供向量量化所需的缩放信息，应为 `TileType::Scaling` |
| `dst` | 输出 Tile | 量化或反量化后的结果 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 经过向量量化转换后的 Tile |

## 副作用

CPU 模拟器当前会接受 `TMOV_FP` 接口，但不会真正消费 `fp` 参数，而是退化为普通 `TMOV` 路径。依赖 `fp` 具体数值的行为应以 NPU backend 和目标 profile 为准。

## 约束

- `src` 必须来自 `Acc` Tile
- `fp` 的设计意图是承载量化参数，可移植代码应将它建成 `TileType::Scaling`
- 目标 dtype 的支持集由具体 backend 上的向量量化模式决定
- A2/A3 的 `TMOV_FP` 实际对应 `Acc -> Mat` 路径，不支持 `Acc -> Vec`
- A2/A3 要求 `FpTileData::Loc` 必须是 `TileType::Scaling`，源必须是 `Acc`，目标必须是 `Mat`，并且满足 `CheckTMovAccToMat(...)`：目标 fractal size 必须是 `512`、目标列宽字节数必须是 `32` 的倍数、源 dtype 必须是 `float` 或 `int32_t`
- A2/A3 量化支持集为 `float Acc -> int8_t Mat` 和 `int32_t Acc -> int8_t / uint8_t / half / int16_t Mat`
- A5 的 `TMOV_FP` 由 `CheckTMovAccValid(..., true)` 约束，`FpTileData::Loc` 必须是 `TileType::Scaling`
- A5 命名接口 `TMOV_FP(dst, src, fp)` 支持 `Acc -> Vec` 和 `Acc -> Mat`
- 对 `Acc -> Vec/Mat`，源 dtype 必须是 `float` 或 `int32_t`，目标布局只允许 `nz2nz`、`nz2nd`、`nz2dn`，目标 stride 对应字节数必须是 `32` 的倍数
- A5 量化支持集为 `float Acc -> int8_t / uint8_t / hifloat8_t / half / bfloat16_t / float8_e4m3_t / float` 和 `int32_t Acc -> int8_t / uint8_t / half / bfloat16_t`

## 异常与非法情形

- 如果目标不是 `Mat` 类型或量化参数格式不正确，backend 可能报错

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using AccT = TileAcc<float, 16, 16>;
  using DstT = Tile<TileType::Vec, int8_t, 16, 16>;
  using FpT = Tile<TileType::Scaling, uint64_t, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox>;

  AccT acc;
  DstT dst;
  FpT fp;
  TMOV_FP(dst, acc, fp);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using AccT = TileAcc<float, 16, 16>;
  using DstT = Tile<TileType::Vec, int8_t, 16, 16>;
  using FpT = Tile<TileType::Scaling, uint64_t, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox>;

  AccT acc;
  DstT dst;
  FpT fp;
  TASSIGN(acc, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(fp, 0x3000);
  TMOV_FP(dst, acc, fp);
}
```

### PTO-AS

```text
%dst = tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

## 相关页面

- [TMOV](./tmov_zh.md)
- [TSTORE_FP](../memory-and-data-movement/tstore-fp_zh.md)
- 指令集总览：[布局与重排](../../layout-and-rearrangement_zh.md)
