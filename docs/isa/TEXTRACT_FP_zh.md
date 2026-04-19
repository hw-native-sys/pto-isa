# pto.textract_fp

`pto.textract_fp` 属于[布局与重排](./tile/ops/layout-and-rearrangement/textract-fp_zh.md)指令集。

## 概述

带 fp/缩放 Tile 的提取（向量量化参数），语义在有效区域上定义，目标相关的行为标记为实现定义。

## 机制

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.textract_fp %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.textract_fp ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src` | - | 源 Tile |
| `fp` | - | FP 缩放 Tile |
| `indexRow` | - | 行偏移量 |
| `indexCol` | - | 列偏移量 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | - | 带 FP 缩放的提取结果 Tile |

## 副作用

提取操作可能触发目标特定的缩放或量化行为。

## 约束

类型/布局/位置/形状的合法性取决于后端；将实现特定的说明视为该后端的规范。

## 异常与非法情形

- 当输入 tile 类型或布局不被目标支持时行为未定义。
- 当偏移量超出有效范围时行为未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| FP 提取 | ✓ | ✓ | ✓ |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Mat, float, 16, 16>;
  using DstT = TileLeft<float, 16, 16>;
  using FpT = TileScale<float, 16, 2>;
  SrcT src;
  DstT dst;
  FpT fp;
  TEXTRACT_FP(dst, src, fp, /*indexRow=*/0, /*indexCol=*/0);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Mat, float, 16, 16>;
  using DstT = TileLeft<float, 16, 16>;
  using FpT = TileScale<float, 16, 2>;
  SrcT src;
  DstT dst;
  FpT fp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(fp, 0x3000);
  TEXTRACT_FP(dst, src, fp, /*indexRow=*/0, /*indexCol=*/0);
}
```

### PTO-AS

```mlir
%dst = pto.textract_fp %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

![TEXTRACT_FP tile operation](../figures/isa/TEXTRACT_FP.svg)

## 相关页面

- 指令集总览：[布局与重排](./tile/ops/layout-and-rearrangement/textract-fp_zh.md)
