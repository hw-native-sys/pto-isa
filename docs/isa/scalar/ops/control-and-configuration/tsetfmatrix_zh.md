# pto.tsetfmatrix

`pto.tsetfmatrix` 虽然保留历史 `t` 前缀，但在手册中归入[控制与配置](../../control-and-configuration_zh.md)路径，因为它配置的是标量可见寄存器状态，而不是 tile payload。

## 简介

配置后续 IMG2COL 一类路径会读取的 FMATRIX 寄存器状态。

## 机制

`pto.tsetfmatrix` 从 `Img2colTileConfig` 一类配置对象中提取输入特征图几何信息与 padding 信息，并把它们写入 FMATRIX 寄存器。该操作本身不直接变换 tile 数据，因此其架构角色属于控制 / 配置。

## 汇编语法

```text
tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

## 输入

- `%cfg`：包含 feature-map 几何与 padding 信息的配置对象
- `FmatrixMode`：选择写入 A 侧还是 B 侧 FMATRIX 寄存器

## 输出

该指令不产生新的 SSA payload 值，只更新 FMATRIX 配置状态。

## 约束

- `%cfg` 必须满足所选 target profile 的 IMG2COL 配置要求。
- 该配置必须出现在依赖它的消费指令之前。

## 相关页面

- [控制与配置](../../control-and-configuration_zh.md)
- [旧 tile 路径兼容入口](../../../tile/ops/sync-and-config/tsetfmatrix_zh.md)
