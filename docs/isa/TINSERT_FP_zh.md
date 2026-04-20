# pto.tinsert_fp

`pto.tinsert_fp` 属于[不规则与复杂指令](./tile/irregular-and-complex_zh.md)集。

## 概述

带 fp/缩放 Tile 的插入操作，用于向量量化参数。除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

## 机制

对目标有效区域内的每个元素 `(i, j)`，将 `fp` Tile 中对应位置的值插入到 `src` 中，索引由 `idxrow` 和 `idxcol` 指定。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.tinsert_fp %src, %fp, %idxrow, %idxcol : (!pto.tile<...>, !pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tinsert_fp ins(%src, %fp, %idxrow, %idxcol : !pto.tile_buf<...>, !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src` | 源 Tile | 源数据 |
| `fp` | 源 Tile | fp/缩放 Tile |
| `idxrow` | 标量 | 行索引 |
| `idxcol` | 标量 | 列索引 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 插入操作后的目标 Tile |

## 副作用

无。

## 约束

- 类型/布局/位置/形状的合法性取决于后端；将实现特定的说明视为该后端的规范。

## 异常与非法情形

- 未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, fp, dst;
  uint16_t idxrow = 0, idxcol = 0;
  TINSERT_FP(dst, src, fp, idxrow, idxcol);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, fp, dst;
  uint16_t idxrow = 0, idxcol = 0;
  TASSIGN(src, 0x1000);
  TASSIGN(fp, 0x2000);
  TASSIGN(dst, 0x3000);
  TINSERT_FP(dst, src, fp, idxrow, idxcol);
}
```

### PTO-AS

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tinsert_fp %src, %fp, %idxrow, %idxcol : (!pto.tile<...>, !pto.tile<...>, dtype, dtype) -> !pto.tile<...>

# 手动模式：先显式绑定资源，再发射指令。
# pto.tassign %src, @tile(0x1000)
# pto.tassign %fp, @tile(0x2000)
# pto.tassign %dst, @tile(0x3000)
%dst = pto.tinsert_fp %src, %fp, %idxrow, %idxcol : (!pto.tile<...>, !pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

## 相关页面

- 指令集总览：[不规则与复杂指令](./tile/irregular-and-complex_zh.md)
