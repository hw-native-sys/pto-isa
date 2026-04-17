# pto.tsethf32mode

`pto.tsethf32mode` 虽然保留历史 `t` 前缀，但在手册中归入[控制与配置](../../control-and-configuration_zh.md)路径，因为它配置的是标量可见模式状态，而不是 tile payload。

## 简介

配置后续计算路径使用的 HF32 模式。

## 机制

`pto.tsethf32mode` 不会修改 tile payload。本指令更新后续 HF32 相关计算路径会读取的模式状态，因此它的架构角色属于控制 / 配置，而不是 tile 算术。

## 汇编语法

```text
tsethf32mode {enable = true, mode = ...}
```

## 输入

- `enable`：启用或关闭 HF32 模式
- `mode`：选择 HF32 的 rounding mode

## 输出

该指令不产生新的 SSA payload 值，只更新模式状态。

## 约束

- 具体 mode 取值和硬件行为由目标实现定义。
- 该配置必须出现在依赖它的计算指令之前。

## 相关页面

- [控制与配置](../../control-and-configuration_zh.md)
- [旧 tile 路径兼容入口](../../../tile/ops/sync-and-config/tsethf32mode_zh.md)
