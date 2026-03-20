# TRESHAPE

## 指令示意图

![TRESHAPE tile operation](../figures/isa/TRESHAPE.svg)

## 简介

将 Tile 重新解释为另一种 Tile 类型/形状，同时保留底层字节。

## 数学语义

除非另有说明，语义定义在有效区域上，目标相关行为标记为实现定义。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

```text
%dst = treshape %src : !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.treshape %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.treshape ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TRESHAPE(TileDataOut &dst, TileDataIn &src, WaitEvents &... events);
```

## 约束

由 `TRESHAPE_IMPL` 强制执行：

- **Tile 类型必须匹配**：`TileDataIn::Loc == TileDataOut::Loc`。
- **总字节大小必须匹配**：`sizeof(InElem) * InNumel == sizeof(OutElem) * OutNumel`。
- **不允许 boxed/non-boxed 转换**：
    - 不能在 `SLayout::NoneBox` 与 boxed 布局之间进行 reshape。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using Src = Tile<TileType::Vec, float, 16, 16>;
  using Dst = Tile<TileType::Vec, float, 8, 32>;
  static_assert(Src::Numel == Dst::Numel);

  Src src;
  Dst dst;
  TRESHAPE(dst, src);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.treshape %src : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.treshape %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = pto.treshape %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.treshape ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

