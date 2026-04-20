# pto.tpop

`pto.tpop` 属于[同步与配置](./tile/sync-and-config_zh.md)指令集。

## 概述

从 pipe 或 FIFO 的消费者端弹出一个 Tile。

## 机制

语义随具体指令变体而变化。除非另有说明，行为都按目标 valid region 定义。

## 语法

### PTO-AS

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.tpop ...
```

### AS Level 2（DPS）

```mlir
pto.tpop ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| Pipe/FIFO 引用 | 输入 | 待弹出 Tile 的 pipe 或 FIFO |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `%dst` | `!pto.tile_buf<...>` | 从消费者端弹出的 Tile |

## 副作用

从 pipe 或 FIFO 中移除一个 Tile，该 Tile 所有权转移给消费者。

## 约束

- 数据类型、layout、location 和 shape 的进一步限制以对应 backend 的合法性检查为准。
- 只能在 pipe 或 FIFO 有可用 Tile 时执行。

## 异常与非法情形

- 在 pipe 或 FIFO 为空时执行属于未定义行为。
- 消费者无权访问该 pipe 或 FIFO 时会被拒绝。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 弹出操作 | Simulated | Supported | Supported |

## 示例

### C++ 自动模式

具体的 Auto / Manual 使用方式见 `docs/isa/` 下的相关指令页。

### C++ 手动模式

具体的 Auto / Manual 使用方式见 `docs/isa/` 下的相关指令页。

### PTO-AS

```text
%tile = pto.tpop %pipe : !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tpop ins(%pipe : ...) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[同步与配置](./tile/sync-and-config_zh.md)
