# pto.tsettf32mode

`pto.tsettf32mode` 属于[同步与配置指令](../../sync-and-config_zh.md)集。

## 概述

`TSETTF32MODE` 设置 TF32 变换模式，该模式供后续指令使用。这是一个状态配置指令，本身不产生直接的张量算术结果，而是更新控制状态。

## 机制

该指令将 TF32 启用状态和舍入模式写入目标配置寄存器，精确模式取值和硬件行为由目标实现定义。后续使用 TF32 的计算指令将根据此设置决定变换行为。

## 语法

### PTO-AS

```text
tsettf32mode {enable = true, mode = ...}
```

### AS Level 1（SSA）

```mlir
pto.tsettf32mode {enable = true, mode = ...}
```

### AS Level 2（DPS）

```mlir
pto.tsettf32mode ins({enable = true, mode = ...}) outs()
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <bool isEnable, RoundMode tf32TransMode = RoundMode::CAST_ROUND, typename... WaitEvents>
PTO_INST RecordEvent TSETTF32MODE(WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `events...` | 可选 | 等待事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| 无 | - | 状态配置指令不产生数据结果 |

## 副作用

设置 TF32 变换模式状态，后续 TF32 计算指令将使用此配置。

## 约束

- 仅在对应 backend capability macro 启用时可用。
- 精确模式取值和硬件行为由目标实现定义。
- 该指令具有控制状态副作用，应与依赖它的计算指令建立正确顺序。

## 异常与非法情形

- 当目标架构不支持 TF32 时，行为未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| TF32 支持 | - | - | 是 |

## 示例

### C++

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_enable_tf32() {
  TSETTF32MODE<true, RoundMode::CAST_ROUND>();
}
```

### PTO-AS

```text
tsettf32mode {enable = true, mode = ...}
```

## 相关页面

- 指令集总览：[同步与配置](../../sync-and-config_zh.md)
- [TSETTF32MODE](./tsettf32mode_zh.md)
