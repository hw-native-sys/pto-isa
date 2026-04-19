# pto.tsettf32mode

`pto.tsettf32mode` 属于[同步与配置](./tile/sync-and-config_zh.md)指令集。

## 概述

`TSETTF32MODE` 设置 TF32 相关的变换模式。它本身不做张量算术，而是更新后续相关计算会读取的模式状态。该指令属于同步与配置路径，更接近"模式寄存器写入"，而不是普通 tile 运算。它的效果取决于目标实现如何解释 TF32 模式配置。

## 机制

该指令写入 TF32 模式寄存器，设置是否启用 TF32 以及具体的变换模式。后续的矩阵运算指令会读取此状态来决定是否使用 TF32 计算路径。

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
| 事件 | 可选 | 等待事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| 事件 | RecordEvent | 同步事件 |

## 副作用

该指令设置 TF32 模式寄存器状态，影响后续所有使用 TF32 格式的矩阵运算行为。

## 约束

- 仅在对应 backend capability macro 启用时可用
- 精确模式取值和硬件行为由目标实现定义
- 该指令具有控制状态副作用，应与依赖它的计算指令建立正确顺序

## 异常与非法情形

- 未指定

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| TF32 模式设置 | - | 可选 | 可选 |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_enable_tf32() {
  TSETTF32MODE<true, RoundMode::CAST_ROUND>();
}
```

### PTO-AS

```text
pto.tsettf32mode {enable = true, mode = ...}
```

## 相关页面

- 指令集总览：[同步与配置](./tile/sync-and-config_zh.md)
- 相关指令：[TSETHF32MODE](./tsethf32mode_zh.md)
