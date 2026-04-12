<!-- Generated from `docs/isa/tile/ops/sync-and-config/tsetfmatrix_zh.md` -->

# TSETFMATRIX

## 指令示意图

![TSETFMATRIX tile operation](../figures/isa/TSETFMATRIX.svg)

## 简介

为类 IMG2COL 操作设置 FMATRIX 寄存器。

## 数学语义

除非另有说明, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## 汇编语法

PTO-AS 形式：参见 [PTO-AS Specification](../assembly/PTO-AS.md).

### AS Level 1 (SSA)

```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

### AS Level 2 (DPS)

```text
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

### AS Level 1（SSA）

```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

### AS Level 2（DPS）

```text
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`:

```cpp
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSETFMATRIX(ConvTileData &src, WaitEvents&... events);
```

## 约束

Type/layout/location/shape legality is backend-dependent; treat implementation-specific notes as normative for that backend.

## 示例

See related examples in `docs/isa/` and `docs/coding/tutorials/`.
