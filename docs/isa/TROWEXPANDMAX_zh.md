# pto.trowexpandmax

`pto.trowexpandmax` 属于[行归约](./tile/ops/reduce-and-expand/trowexpandmax_zh.md)指令集。

## 概述

行广播最大值：将 `src1` 中每行的标量值与 `src0` 对应行的所有元素取最大值，结果写入 `dst`。

## 机制

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。设 `s_i` 为从 `src1` 中获取的每行标量（每行一个值）。

对于 `0 <= i < R` 和 `0 <= j < C`：

$$
\mathrm{dst}_{i,j} = \max(\mathrm{src0}_{i,j}, s_i)
$$

## 语法

### PTO-AS

```text
%dst = trowexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.trowexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.trowexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src0` | 输入 | 源 tile 0，`half` 或 `float` 类型 |
| `src1` | 输入 | 每行一个标量（模式 1）或每行 32 字节数据（模式 2） |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | 输出 | 行广播最大值结果，与 `src0` 类型相同 |

## 副作用

`dst` 的有效区域定义结果的计算范围。

## 约束

- `TileDataDst::DType == TileDataSrc0::DType == TileDataSrc1::DType`
- `TileDataDst::DType`、`TileDataSrc0::DType`、`TileDataSrc1::DType` 必须是以下之一：`half`、`float`
- Tile 形状/布局约束（编译时）：`TileDataDst::isRowMajor`
- 模式 1：`src1` 预期提供每行一个标量（即，其有效形状必须覆盖 `R` 个值）
- 模式 2：`src1` 预期提供每行 32 字节数据
- 确切的布局/分形约束是目标特定的；参见 `include/pto/npu/*/TRowExpand*.hpp` 下的后端头文件

## 异常与非法情形

- 运行时检查失败时，行为由具体实现定义

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 支持 | 是 | 是 | 是 |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TROWEXPANDMAX(dst, src0, src1);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TROWEXPANDMAX(dst, src0, src1);
}
```

### PTO-AS

```text
# 自动模式
%dst = pto.trowexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

## 相关页面

- 指令集总览：[行归约](./tile/ops/reduce-and-expand/trowexpandmax_zh.md)
