# pto.tpush

`pto.tpush` 属于[同步与配置](./tile/sync-and-config_zh.md)指令集。

## 概述

将 Tile 推入 pipe 或 FIFO 的生产者端。

## 机制

语义随具体指令变体而变化。除非另有说明，行为都按目标 valid region 定义。

## 语法

### PTO-AS

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```mlir
%dst = pto.tpush ...
```

### AS Level 2（DPS）

```mlir
pto.tpush ins(...) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| 源 Tile | 输入 | 待推入 pipe 或 FIFO 的 Tile |
| Pipe/FIFO 引用 | 输入 | 目标 pipe 或 FIFO |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| 无 | - | 推送操作无返回值，Tile 所有权转移给生产者 |

## 副作用

将 Tile 移入 pipe 或 FIFO 的生产者端，Tile 所有权从消费者转移给生产者。

## 约束

- 数据类型、layout、location 和 shape 的进一步限制以对应 backend 的合法性检查为准。
- 只能在 pipe 或 FIFO 有可用槽位时执行。

## 异常与非法情形

- 在 pipe 或 FIFO 满时执行属于未定义行为。
- 生产者无权访问该 pipe 或 FIFO 时会被拒绝。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 推送操作 | Simulated | Supported | Supported |

## 示例

### C++ 自动模式

具体的 Auto / Manual 使用方式见 `docs/isa/` 下的相关指令页。

### C++ 手动模式

具体的 Auto / Manual 使用方式见 `docs/isa/` 下的相关指令页。

### PTO-AS

```text
pto.tpush %tile, %pipe : !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.tpush ins(%tile, %pipe : ...) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- 指令集总览：[同步与配置](./tile/sync-and-config_zh.md)
