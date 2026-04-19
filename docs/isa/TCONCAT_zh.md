# pto.tconcat

`pto.tconcat` 属于[布局与重排](./tile/layout-and-rearrangement_zh.md)指令集。

## 概述

沿列维度将两个源 Tile 拼接到目标 Tile 中，形成更宽的 Tile。

## 机制

语义随具体指令变体而变化。除非另有说明，行为都按目标 valid region 定义。

## 语法

### PTO-AS

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.tconcat ...
```

### AS Level 2（DPS）

```mlir
pto.tconcat ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| 源 Tile 1 | 输入 | 待拼接的第一个 Tile |
| 源 Tile 2 | 输入 | 待拼接的第二个 Tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile_buf<...>` | 拼接后的目标 Tile |

## 副作用

拼接操作会产生目标 Tile，无额外架构副作用。

## 约束

- 数据类型、layout、location 和 shape 的进一步限制以对应 backend 的合法性检查为准。

## 异常与非法情形

- 不支持的 Tile 形状/布局组合会被 verifier 或后端拒绝。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 拼接操作 | Simulated | Supported | Supported |

## 示例

### C++ 自动模式

具体的 Auto / Manual 使用方式见 `docs/isa/` 下的相关指令页。

### C++ 手动模式

具体的 Auto / Manual 使用方式见 `docs/isa/` 下的相关指令页。

### PTO-AS

```text
%dst = pto.tconcat %src1, %src2 : !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tconcat ins(%src1, %src2 : ...) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[布局与重排](./tile/layout-and-rearrangement_zh.md)
