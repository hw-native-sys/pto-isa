# pto.treshape

`pto.treshape` 属于[布局与重排](../../layout-and-rearrangement_zh.md)指令集。

## 概述

`TRESHAPE` 重新解释一个 Tile 的字节视图，而不改变底层字节内容。它不是数值转换，也不是数据搬运；它做的是"同一块数据，用另一种 Tile 形状/类型规则来看"。因此它的核心前提不是"形状能不能算"，而是"总字节数和布局类别能不能兼容"。

## 机制

从结果上看，可以把 `TRESHAPE` 理解成 `src` 的字节序列保持不变，而 `dst` 只是用另一套 Tile 元数据去解释同一批字节。

## 语法

### PTO-AS

```text
%dst = treshape %src : !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.treshape %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.treshape ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TRESHAPE(TileDataOut &dst, TileDataIn &src, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src` | 输入 Tile | 源 Tile，保持字节内容不变 |
| `dst` | 输出 Tile | 用新的 Tile 元数据解释同一批字节 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 与 `src` 字节内容相同但 Tile 元数据不同的 Tile |

## 副作用

`TRESHAPE` 在 NPU 上实际更接近"受约束的别名/重解释"，而不是一次真实复制。

## 约束

- 所有 backend 都必须满足：`TileDataIn::Loc == TileDataOut::Loc`
- 所有 backend 都必须满足：`sizeof(InElem) * InNumel == sizeof(OutElem) * OutNumel`
- 不能在 boxed layout 和 non-boxed layout 之间重解释
- CPU 模拟器还会额外检查元素类型兼容性：同类型，或都是浮点，或都是整数
- A2/A3 在非自动路径下会把 `dst` 直接别名到 `src` 的地址；自动路径用 `__cce_alias`
- A5 和 Kirin9030 复用 A2/A3 的 `TRESHAPE` 实现

## 异常与非法情形

- 如果违反上述约束，行为由具体 backend 决定

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using Src = Tile<TileType::Vec, float, 16, 16>;
  using Dst = Tile<TileType::Vec, float, 8, 32>;
  static_assert(sizeof(typename Src::DType) * Src::Numel == sizeof(typename Dst::DType) * Dst::Numel);

  Src src;
  Dst dst;
  TRESHAPE(dst, src);
}
```

## 相关页面

- [TALIAS](../../../TALIAS_zh.md)
- [TMOV](./tmov_zh.md)
- 指令集总览：[布局与重排](../../layout-and-rearrangement_zh.md)
