# pto.tsetfmatrix

`pto.tsetfmatrix` 属于[控制与配置指令](../../control-and-configuration_zh.md)集。

## 概述

配置后续 IMG2COL 一类路径会读取的 FMATRIX 寄存器状态。`pto.tsetfmatrix` 从 `Img2colTileConfig` 一类配置对象中提取输入特征图几何信息与 padding 信息，并把它们写入 FMATRIX 寄存器。

## 机制

`pto.tsetfmatrix` 从配置对象中提取 feature-map 几何信息与 padding 信息，并写入 FMATRIX 寄存器。该操作本身不直接变换 tile 数据，因此其架构角色属于控制/配置。

## 语法

### PTO-AS

```text
tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

### AS Level 1（SSA）

```mlir
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `%cfg` | 配置 | 包含 feature-map 几何与 padding 信息的配置对象 |
| `FmatrixMode` | 配置 | 选择写入 A 侧还是 B 侧 FMATRIX 寄存器 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| 无 | - | 该指令不产生新的 SSA payload 值，只更新 FMATRIX 配置状态 |

## 副作用

更新 FMATRIX 寄存器状态，后续 IMG2COL 指令会读取该配置。

## 约束

- `%cfg` 必须满足所选 target profile 的 IMG2COL 配置要求
- 该配置必须出现在依赖它的消费指令之前

## 异常与非法情形

- 若 `%cfg` 不满足目标 profile 的 IMG2COL 要求，行为未定义
- 若该配置在依赖它的消费指令之后发出，行为未定义

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
  // 创建 IMG2COL 配置
  Img2colTileConfig cfg;
  cfg.input_h = 224;
  cfg.input_w = 224;
  cfg.padding_h = 1;
  cfg.padding_w = 1;

  // 配置 FMATRIX（写入 A 侧）
  pto.tsetfmatrix(cfg, FmatrixMode::A);
}
```

### PTO-AS

```text
# 创建并配置 FMATRIX（写入 A 侧）
%cfg = pto.create_fmatrix_config {input_h = 224, input_w = 224, pad_h = 1, pad_w = 1} : !pto.fmatrix_config
pto.tsetfmatrix %cfg {mode = a} : !pto.fmatrix_config -> ()

# 写入 B 侧
pto.tsetfmatrix %cfg {mode = b} : !pto.fmatrix_config -> ()
```

## 相关页面

- 指令集总览：[控制与配置指令](../../control-and-configuration_zh.md)
- [旧 tile 路径兼容入口](../../../tile/ops/sync-and-config/tsetfmatrix_zh.md)
