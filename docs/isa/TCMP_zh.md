# pto.tcmp

![TCMP tile operation](../figures/isa/TCMP.svg)

`pto.tcmp` 属于[逐元素 Tile-Tile](./tile/elementwise-tile-tile_zh.md)指令集。

## 概述

比较两个 tile 并将结果写成打包谓词 tile。迭代域由目标 tile 的 valid region 决定。

## 机制

从语义上看，对目标 tile 的 valid region 中每个 `(i, j)`，先定义一个谓词：

$$ p_{i,j} = \left(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{src1}_{i,j}\right) $$

随后把这些谓词位按目标定义的 packed layout 写入 `dst`。也就是说，`dst` 不是"每个 lane 一个布尔数"的朴素 tile，而是某种目标定义的压缩谓词表示。程序不能假设谓词 tile 的具体编码。

## 语法

### PTO-AS

```text
%dst = tcmp %src0, %src1 {cmpMode = #pto.cmp<EQ>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.tcmp %src0, %src1 {cmpMode = #pto.cmp<EQ>} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tcmp ins(%src0, %src1 {cmpMode = #pto.cmp<EQ>}: !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCMP(TileDataDst &dst, TileDataSrc &src0, TileDataSrc &src1,
                          CmpMode cmpMode, WaitEvents &... events);
```

### 比较模式

| 模式 | 含义 |
| --- | --- |
| `EQ` | 等于 |
| `NE` | 不等于 |
| `LT` | 小于 |
| `LE` | 小于等于 |
| `GT` | 大于 |
| `GE` | 大于等于 |

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%src0` | 左 tile | 在 `dst` valid region 上逐坐标参与比较 |
| `%src1` | 右 tile | 在 `dst` valid region 上逐坐标参与比较 |
| `%dst` | 谓词 tile | 保存打包后的比较结果 |
| `cmpMode` | 比较谓词 | 选择 EQ / NE / LT / LE / GT / GE |
| `WaitEvents...` | 可选同步 | 发射前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile<...>` | 打包后的谓词结果 tile，具体编码由目标 profile 定义 |

## 副作用

除产生谓词 tile 外，没有额外架构副作用。

## 约束

- 迭代域是 `dst.GetValidRow() × dst.GetValidCol()`。
- `src0` 的 validRow / validCol 必须与 `dst` 一致。
- `src1` 的 shape / valid 在某些实现里不会做完整运行时校验，因此域外读值属于 implementation-defined。
- 程序不能假设谓词 tile 的具体编码。

## 异常与非法情形

- 假设谓词 tile 是"一位一元素"的普通展开布尔 tile，会导致行为不可预期。
- 对 `dst` 使用不符合目标定义的谓词输出 dtype，会被 verifier 或后端拒绝。

## Target-Profile 限制

| 检查项 | A2/A3 | A5 |
| --- | :---: | :---: |
| 支持输入类型 | `int32_t`、`half`、`float` | `uint32_t`、`int32_t`、`uint16_t`、`int16_t`、`uint8_t`、`int8_t`、`float`、`half` |
| 输出谓词 dtype | `uint8_t` | `uint32_t` |
| tile 位置 | `TileType::Vec` | `TileType::Vec` |
| 布局 | RowMajor | RowMajor |
| `src0` valid == `dst` valid | Required | Required |
| `src1` validity 校验 | Not fully verified | Not fully verified |

对 A2A3，若输入类型是 `int32_t`，实现里可能只走 `EQ` 比较路径；A5 则支持完整比较模式。

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src0, src1;
  MaskT mask(16, 2);
  TCMP(mask, src0, src1, CmpMode::GT);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src0, src1;
  MaskT mask(16, 2);
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(mask, 0x3000);
  TCMP(mask, src0, src1, CmpMode::GT);
}
```

### PTO-AS

```text
# 自动模式
%dst = pto.tcmp %src0, %src1 {cmpMode = #pto.cmp<GT>} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>

# 手动模式
pto.tassign %arg0, @tile(0x1000)
pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcmp %src0, %src1 {cmpMode = #pto.cmp<GT>} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>

# PTO 汇编形式
%dst = tcmp %src0, %src1 {cmpMode = #pto.cmp<EQ>} : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcmp ins(%src0, %src1 {cmpMode = #pto.cmp<EQ>}: !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[逐元素 Tile-Tile](./tile/elementwise-tile-tile_zh.md)
- 规范页：[pto.tcmp](./tile/ops/elementwise-tile-tile/tcmp_zh.md)
