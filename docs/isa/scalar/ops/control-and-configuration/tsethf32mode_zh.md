# pto.tsethf32mode

`pto.tsethf32mode` 属于[控制与配置指令](../../control-and-configuration_zh.md)集。

## 概述

配置后续计算路径使用的 HF32（半精度浮点 32 位）模式。该指令更新后续 HF32 相关计算路径会读取的模式状态，因此它的架构角色属于控制/配置，而不是 tile 算术。

## 机制

`pto.tsethf32mode` 不会修改 tile payload。本指令更新后续 HF32 相关计算路径会读取的模式状态。

## 语法

### PTO-AS

```text
tsethf32mode {enable = true, mode = ...}
```

### AS Level 1（SSA）

```mlir
pto.tsethf32mode {enable = true, mode = ...}
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `enable` | 配置 | 布尔值，启用或关闭 HF32 模式 |
| `mode` | 配置 | HF32 的 rounding mode 选择 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| 无 | - | 该指令不产生新的 SSA payload 值，只更新模式状态 |

## 副作用

更新后续 HF32 相关计算路径读取的全局模式状态。

## 约束

- 具体 mode 取值和硬件行为由目标实现定义
- 该配置必须出现在依赖它的计算指令之前

## 异常与非法情形

- 若该配置在依赖它的计算指令之后发出，行为未定义
- 若指定的 mode 不被目标硬件支持，结果未定义

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 支持 | 是 | 是 | 是 |

## 示例

### C++

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  // 启用 HF32 模式，使用默认 rounding mode
  pto.tsethf32mode {enable = true, mode = Hf32RoundingMode::ToEven};

  // 后续计算使用 HF32 模式
  // ...
}
```

### PTO-AS

```text
# 启用 HF32 模式
tsethf32mode {enable = true, mode = to_even}

# 关闭 HF32 模式
tsethf32mode {enable = false}
```

## 相关页面

- 指令集总览：[控制与配置指令](../../control-and-configuration_zh.md)
- [旧 tile 路径兼容入口](../../../tile/ops/sync-and-config/tsethf32mode_zh.md)
