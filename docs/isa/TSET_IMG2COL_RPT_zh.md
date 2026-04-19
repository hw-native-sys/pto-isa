# pto.tset_img2col_rpt

`pto.tset_img2col_rpt` 属于[同步与配置](./tile/sync-and-config_zh.md)指令集。

## 概述

从 IMG2COL 配置 Tile 设置 IMG2COL 重复次数元数据。该指令本身不产生直接的张量算术结果，而是更新供后续数据搬运操作使用的 IMG2COL 控制状态。

## 机制

该指令将配置 Tile 中的重复次数参数写入硬件状态寄存器，后续的 IMG2COL 操作会读取这些参数来确定数据重复搬运的次数。

## 语法

### PTO-AS

```text
tset_img2col_rpt %cfg
```

### AS Level 1（SSA）

```mlir
pto.tset_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

### AS Level 2（DPS）

```mlir
pto.tset_img2col_rpt ins(%cfg : !pto.fmatrix_config) outs()
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename ConvTileData, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_RPT(ConvTileData &src, WaitEvents &... events);

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_RPT(ConvTileData &src, WaitEvents &... events);
```

For `MEMORY_BASE` targets, an overload without `SetFmatrixMode` is also provided.

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| src | 输入 | IMG2COL 配置 Tile |
| events | 可选 | 等待事件 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| 事件 | RecordEvent | 同步事件 |

## 副作用

该指令更新 IMG2COL 重复次数控制状态，影响后续 TIMG2COL 操作的数据搬运行为。

## 约束

- 该指令是后端特定的，仅适用于暴露 IMG2COL 配置状态的硬件平台
- src 必须是后端实现可接受的 IMG2COL 配置 tile 类型
- 该指令更新的确切寄存器/元数据字段由实现定义
- 在同一执行流中，应在依赖的 TIMG2COL 操作之前使用此指令

## 异常与非法情形

- 未指定

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| IMG2COL 重复次数配置 | - | 支持 | 支持 |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_set_img2col_rpt(Img2colTileConfig<uint64_t>& cfg) {
  TSET_IMG2COL_RPT(cfg);
}
```

### PTO-AS

```text
pto.tset_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

## 相关页面

- 指令集总览：[同步与配置](./tile/sync-and-config_zh.md)
- 相关指令：[TIMG2COL](./timg2col_zh.md)、[TSET_IMG2COL_PADDING](./TSET_IMG2COL_PADDING_zh.md)
