# pto.tpack

`pto.tpack` 属于[不规则与复杂](./tile/irregular-and-complex_zh.md)指令集。

## 概述

将 Tile 元素打包或转换为更窄的目标表示。

## 机制

语义随具体指令变体而变化。除非另有说明，行为都按目标 valid region 定义。

## 语法

### PTO-AS

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.tpack ...
```

### AS Level 2（DPS）

```mlir
pto.tpack ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| 源 Tile | 输入 | 待打包的源 Tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile_buf<...>` | 打包后的目标 Tile |

## 副作用

除产生目标 Tile 外，没有额外架构副作用。

## 约束

- 数据类型、layout、location 和 shape 的进一步限制以对应 backend 的合法性检查为准。

## 异常与非法情形

- 源 Tile 与目标 Tile 类型不兼容时会被 verifier 或后端拒绝。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 打包操作 | Simulated | Supported | Supported |

## 示例

### C++ 自动模式

具体的 Auto / Manual 使用方式见 `docs/isa/` 下的相关指令页。

### C++ 手动模式

具体的 Auto / Manual 使用方式见 `docs/isa/` 下的相关指令页。

### PTO-AS

```text
%dst = pto.tpack %src : ...
```

### AS Level 2（DPS）

```mlir
pto.tpack ins(%src : ...) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[不规则与复杂](./tile/irregular-and-complex_zh.md)
