# pto.tmov.fp

`pto.tmov.fp` 属于[布局与重排](./tile/layout-and-rearrangement_zh.md)指令集。

## 概述

使用缩放 (`fp`) Tile 作为向量量化参数，将累加器 Tile 移动/转换到目标 Tile。概念上使用从 `fp` 派生的实现定义的量化/反量化配置转换每个元素：$ \mathrm{dst}_{i,j} = \mathrm{Convert}\!\left(\mathrm{src}_{i,j};\ \mathrm{fp}\right) $

## 机制

该指令将累加器 Tile 中的数据转换后写入目标 Tile，转换参数由 `fp` Tile 提供。`fp` Tile 包含实现定义的量化参数，用于控制转换行为。

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

声明于 `include/pto/common/pto_instr.hpp` 和 `include/pto/common/constants.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| dst | 输出 | 目标 Tile |
| src | 输入 | 源累加器 Tile |
| fp | 输入 | 浮点量化参数 Tile |
| events | 可选 | 等待事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| dst | DstTileData& | 量化转换后的目标 Tile |
| 事件 | RecordEvent | 同步事件 |

## 副作用

该指令使用 `fp` Tile 中的量化参数进行数据转换。

## 约束

- 实现检查 (A2A3):
    - fp 路径仅支持累加器转换，并通过 TMOV_IMPL(dst, src, fp) 中的内部编译时检查进行验证
    - FpTileData::Loc 必须是 TileType::Scaling（static_assert）
- 实现检查 (A5):
    - 通过 CheckTMovAccValid(...) 和 TMOV_IMPL(dst, src, fp) 中的相关编译时检查进行验证
    - FpTileData::Loc 必须是 TileType::Scaling（static_assert）
    - 目标位置取决于目标（fp 路径支持 Vec 或 Mat）

## 异常与非法情形

- 未指定

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| Acc -> Vec (fp) | - | 支持 | 支持 |
| Acc -> Mat (fp) | - | 支持 | 支持 |

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
  TASSIGN(fp,  0x3000);
  TMOV_FP(dst, acc, fp);
}
```

### PTO-AS

```text
# 自动模式
%dst = pto.tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>

# 手动模式
%dst = pto.tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>

# PTO 汇编形式
%dst = tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tmov.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[布局与重排](./tile/layout-and-rearrangement_zh.md)
- 相关指令：[pto.tmov](./TMOV_zh.md)
