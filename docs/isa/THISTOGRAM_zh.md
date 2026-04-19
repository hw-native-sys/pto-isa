# pto.thistogram

`pto.thistogram` 属于[不规则与复杂](./tile/irregular-and-complex_zh.md)指令集。

## 概述

使用索引 Tile 从源值中累计直方图 bin 计数。

## 机制

语义随具体指令变体而变化。除非另有说明，行为都按目标 valid region 定义。

## 语法

### PTO-AS

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.thistogram ...
```

### AS Level 2（DPS）

```mlir
pto.thistogram ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| 源值 Tile | 输入 | 待累计的源值 Tile |
| 索引 Tile | 输入 | 指定每个值应落入的直方图 bin |
| 直方图 Tile | 输入/输出 | 累计计数的直方图 Tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile_buf<u32>` | 更新后的直方图 Tile |

## 副作用

直方图 Tile 中的 bin 计数会被原地更新。

## 约束

- 数据类型、layout、location 和 shape 的进一步限制以对应 backend 的合法性检查为准。
- 索引值必须在直方图 bin 范围内。

## 异常与非法情形

- 索引越界会被 verifier 或运行时检测并拒绝。
- 不支持的元素类型或布局会被后端拒绝。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 直方图操作 | Simulated | Supported | Supported |

## 示例

### C++ 自动模式

具体的 Auto / Manual 使用方式见 `docs/isa/` 下的相关指令页。

### C++ 手动模式

具体的 Auto / Manual 使用方式见 `docs/isa/` 下的相关指令页。

### PTO-AS

```text
%dst = pto.thistogram %values, %indices, %histogram : ...
```

### AS Level 2（DPS）

```mlir
pto.thistogram ins(%values, %indices : ...) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[不规则与复杂](./tile/irregular-and-complex_zh.md)
