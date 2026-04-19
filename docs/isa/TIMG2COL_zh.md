# pto.timg2col

`pto.timg2col` 属于[布局与重排](./tile/ops/layout-and-rearrangement/timg2col_zh.md)指令集。

## 概述

用于类卷积工作负载的图像到列变换，将图像 tile 重新排列为适合 GEMM 操作的列格式，语义在有效区域上定义，目标相关的行为标记为实现定义。

## 机制

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
PTO_INST RecordEvent TIMG2COL(TileData &dst, ConvTileData &src, uint16_t posM = 0, uint16_t posK = 0,
                              WaitEvents&... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src` | Conv | 输入卷积图像 Tile |
| `posM` | - | M 维度偏移量 |
| `posK` | - | K 维度偏移量 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | - | img2col 变换后的 Tile |

## 副作用

img2col 变换可能产生填充值或目标特定的填充模式。

## 约束

- 此指令是目标/实现特定的。
- 参见 `include/pto/npu/*/TImg2col.hpp` 了解支持的 tile 类型/布局和配置字段。

## 异常与非法情形

- 当输入 tile 类型或布局不被目标支持时行为未定义。
- 当偏移量超出有效范围时行为未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| img2col 变换 | ✓ | ✓ | ✓ |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  ConvTileData src;
  TileData dst;
  TIMG2COL(dst, src);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  ConvTileData src;
  TileData dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TIMG2COL(dst, src);
}
```

### PTO-AS

```mlir
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

![TIMG2COL tile operation](../figures/isa/TIMG2COL.svg)

## 相关页面

- 指令集总览：[布局与重排](./tile/ops/layout-and-rearrangement/timg2col_zh.md)
