# pto.talias

`pto.talias` 属于[布局与重排指令](./tile/layout-and-rearrangement_zh.md)集。

## 概述

`TALIAS` 创建一个与源 Tile 共享底层存储的别名视图。它不复制数据，只改变后续代码观察这块存储的逻辑方式。`TALIAS` 的结果与源 Tile 指向同一块底层存储，通过任一别名写入的数据都会对共享该存储的其他别名立即可见。

## 机制

### 数学语义

```math
\mathrm{dst} \equiv \mathrm{src} \quad \text{其中 } \mathrm{storage}(\mathrm{dst}) = \mathrm{storage}(\mathrm{src})
```

别名视图共享同一底层 buffer，但可具有不同的 shape、stride 或 layout 解释方式。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.talias ...
```

### AS Level 2（DPS）

```mlir
pto.talias ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src` | 输入 | 源 Tile，提供底层存储 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 与源共享底层存储的别名视图 |

## 副作用

通过任一别名写入的数据，对共享该存储的其他别名立即可见。

## 约束

- `TALIAS` 不会修复原始 Tile 的非法 shape、layout 或 location intent
- alias 后得到的 Tile 仍需满足后续消费者指令的合法性要求
- 若多个别名在没有额外同步的情况下并发读写，共享存储的可见顺序由后续使用这些别名的指令负责建立

## 异常与非法情形

- 若源 Tile 本身未绑定有效存储，行为未定义
- 若别名视图被后续指令以不兼容的 shape 或 layout 使用，结果未定义

## 示例

### C++

`TALIAS` 常与子视图、局部重排和"同存储多视图"模式配合使用。

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_alias() {
  using TileT = Tile<TileType::Vec, float, 32, 32>;
  TileT src;
  // 创建别名视图
  auto alias_view = TALIAS(src);
  // alias_view 与 src 共享底层存储
}
```

### PTO-AS

```mlir
# 创建别名视图
%dst = pto.talias %src : (!pto.tile<...>) -> !pto.tile<...>
```

## 相关页面

- 指令集总览：[布局与重排指令](./tile/layout-and-rearrangement_zh.md)
- [Tile 与有效区域](./programming-model/tiles-and-valid-regions_zh.md)
