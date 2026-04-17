# pto.trowsum

`pto.trowsum` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述

对每一行按列求和。

## 机制

设：

- `R = src.GetValidRow()`
- `C = src.GetValidCol()`

则对 `0 <= i < R`：

$$ \mathrm{dst}_{i,0} = \sum_{j=0}^{C-1} \mathrm{src}_{i,j} $$

也就是说，`trowsum` 把 `(R, C)` 压成 `(R, 1)`，保留行、折叠列。lowering 过程中通常会引入临时 tile，C++ intrinsic 因此要求显式传入 `tmp`。

## 语法

同步形式：

```text
%dst = trowsum %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入

- `src`：源 tile
- `tmp`：归约过程中的临时 tile
- `dst`：目标 tile

## 预期输出

- `dst[i,0]`：第 `i` 行所有列元素的和

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 与 `src` 都必须是 `TileType::Vec`
- `src` 必须是标准 ND 布局：行主且非分形
- `dst` 可以是 ND，或 `Cols == 1` 的 DN 布局
- `dst` 与 `src` 元素类型必须一致
- 运行时要求：
  - `src.GetValidRow() != 0`
  - `src.GetValidCol() != 0`
  - `src.GetValidRow() == dst.GetValidRow()`

### A2A3

- 支持类型：`half`、`float`、`int32_t`、`int16_t`
- `tmp` 会进入后端调用路径，但当前文档不额外扩大 checked implementation 没显式声明的 shape / layout 约束

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能

### A2A3

`TROWSUM` 在 A2/A3 上通常会 lowering 成多阶段向量归约序列。英文页给出的要点是：

- 总周期由多段 `vcgadd / vadd / vcadd` 组合而成
- shape 不同，采用的序列不同

示例：

| valid shape | 典型序列 |
| --- | --- |
| `64×128` | `vcgadd*128 -> vadd*8 -> vcgadd*8` |
| `32×256` | `vcgadd*128 -> vadd*8 -> vadd*4 -> vcgadd*4` |
| `16×512` | `vcgadd*128 -> vcgadd*16 -> vcgadd*2` |
| `8×1024` | `vcgadd*128 -> vcgadd*16 -> vadd*8 -> vcgadd*8` |

### A5

当前手册未单列 `trowsum` 的独立周期表，应视为目标 profile 相关。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWSUM(dst, src, tmp);
}
```

## 相关页面

- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- 下一条指令：[pto.tcolsum](./tcolsum_zh.md)
