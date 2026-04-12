<!-- Generated from `docs/isa/TSETHF32MODE_zh.md` -->

# TSETHF32MODE

## 指令示意图

![TSETHF32MODE tile operation](../figures/isa/TSETHF32MODE.svg)

## 简介

设置 HF32 变换模式（实现定义）。

## 数学语义

No direct tensor arithmetic is produced by this instruction. It updates target mode state used by subsequent instructions.

## 汇编语法

PTO-AS 形式：参见 `docs/assembly/PTO-AS.md`.

Schematic form:

```text
tsethf32mode {enable = true, mode = ...}
```

### AS Level 1（SSA）

```text
pto.tsethf32mode {enable = true, mode = ...}
```

### AS Level 2（DPS）

```text
pto.tsethf32mode ins({enable = true, mode = ...}) outs()
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`:

```cpp
template <bool isEnable, RoundMode hf32TransMode = RoundMode::CAST_ROUND, typename... WaitEvents>
PTO_INST RecordEvent TSETHF32MODE(WaitEvents &... events);
```

## 约束

- Available only when the corresponding backend capability macro is enabled.
- Exact mode values and hardware behavior are target-defined.
- This instruction has control-state side effects and should be ordered appropriately relative to dependent compute instructions.

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_enable_hf32() {
  TSETHF32MODE<true, RoundMode::CAST_ROUND>();
}
```
