# TSETFMATRIX

## 指令示意图

![TSETFMATRIX tile operation](../figures/isa/TSETFMATRIX.svg)

## 简介

为类 IMG2COL 操作设置 FMATRIX 寄存器。

## 数学语义

除非另有说明，语义定义在有效区域上，目标相关行为标记为实现定义。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

### AS Level 1（SSA）

```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

### AS Level 2（DPS）

```text
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSETFMATRIX(ConvTileData &src, WaitEvents &... events);
```

## 约束

类型/布局/位置/形状的合法性由后端决定；对于特定后端，请将实现相关说明视为规范性约束。

## 示例

参见 `docs/isa/` 和 `docs/coding/tutorials/` 中的相关示例。

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

### PTO 汇编形式

```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
# AS Level 2 (DPS)
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

