# pto.tfree

`pto.tfree` 属于[同步与配置](./tile/sync-and-config_zh.md)指令集。

## 概述

将当前占用的 pipe 或 FIFO 槽位释放回生产者。

## 机制

语义随具体指令变体而变化。除非另有说明，行为都按目标 valid region 定义。

## 语法

### PTO-AS

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.tfree ...
```

### AS Level 2（DPS）

```mlir
pto.tfree ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| 源 Tile | 输入 | 待释放槽位中的 Tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| 无 | - | 释放操作无返回值，仅将槽位归还生产者 |

## 副作用

释放 pipe 或 FIFO 槽位，将其归还给生产者供后续使用。

## 约束

- 数据类型、layout、location 和 shape 的进一步限制以对应 backend 的合法性检查为准。
- 只能释放当前核/线程持有的槽位。

## 异常与非法情形

- 释放未持有的槽位属于未定义行为。
- 在 pipe 或 FIFO 未正确初始化的情形下执行会被后端拒绝。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 释放操作 | Simulated | Supported | Supported |

## 示例

### C++ 自动模式

具体的 Auto / Manual 使用方式见 `docs/isa/` 下的相关指令页。

### C++ 手动模式

具体的 Auto / Manual 使用方式见 `docs/isa/` 下的相关指令页。

### PTO-AS

```text
%result = pto.tfree %tile : !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tfree ins(%tile : ...) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[同步与配置](./tile/sync-and-config_zh.md)
