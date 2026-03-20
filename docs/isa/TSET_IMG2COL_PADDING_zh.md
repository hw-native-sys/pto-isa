# TSET_IMG2COL_PADDING

## 指令示意图

![TSET_IMG2COL_PADDING tile operation](../figures/isa/TSET_IMG2COL_PADDING.svg)

## 简介

从 IMG2COL 配置 Tile 设置 IMG2COL 填充元数据（实现定义）。

## 数学语义

该指令不直接产生张量算术结果。它会更新后续数据搬运类操作使用的 IMG2COL 填充控制状态。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

示意形式：

```text
tset_img2col_padding %cfg
```

### AS Level 1（SSA）

```text
pto.tset_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

### AS Level 2（DPS）

```text
pto.tset_img2col_padding ins(%cfg : !pto.fmatrix_config) outs()
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename ConvTileData, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_PADDING(ConvTileData &src, WaitEvents &... events);

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_PADDING(ConvTileData &src, WaitEvents &... events);
```

在 `MEMORY_BASE` 目标上，还提供不带 `SetFmatrixMode` 模板参数的重载。

## 约束

- 该指令属于后端相关能力，仅在支持 IMG2COL 配置状态的后端可用。
- `src` 必须是目标后端接受的 IMG2COL 配置 Tile 类型。
- 该指令更新的填充相关字段属于实现定义行为。
- 在同一执行流中，应先设置该状态，再执行依赖的 `TIMG2COL` 指令。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_set_img2col_padding(Img2colTileConfig<uint64_t>& cfg) {
  TSET_IMG2COL_PADDING(cfg);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
pto.tset_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tset_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

### PTO 汇编形式

```text
tset_img2col_padding %cfg
# AS Level 2 (DPS)
pto.tset_img2col_padding ins(%cfg : !pto.fmatrix_config) outs()
```

