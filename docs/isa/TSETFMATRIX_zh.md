# TSETFMATRIX

## 指令示意图

![TSETFMATRIX tile operation](../figures/isa/TSETFMATRIX.svg)

## 简介

为类 IMG2COL 操作设置 FMATRIX 寄存器。

## 数学语义

除非另有说明, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

### IR Level 1（SSA）

```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

### IR Level 2（DPS）

```text
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename T = uint64_t, typename... WaitEvents>
PTO_INST RecordEvent TSETFMATRIX(const Img2colTileConfig<T> &cfg = Img2colTileConfig<T>{}, WaitEvents&... events);
```

## 约束

Type/layout/location/shape legality is backend-dependent; treat implementation-specific notes as normative for that backend.

## 示例

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

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
# IR Level 2 (DPS)
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

