<!-- Generated from `docs/isa/TSET_IMG2COL_RPT_zh.md` -->

# TSET_IMG2COL_RPT

## 指令示意图

![TSET_IMG2COL_RPT tile operation](../figures/isa/TSET_IMG2COL_RPT.svg)

## 简介

从 IMG2COL 配置 Tile 设置 IMG2COL 重复次数元数据。

## 数学语义

No direct tensor arithmetic is produced by this instruction. It updates IMG2COL control state used by subsequent data-movement operations.

## 汇编语法

PTO-AS 形式：参见 [PTO-AS Specification](../assembly/PTO-AS.md).

Schematic form:

```text
tset_img2col_rpt %cfg
```

### AS Level 1 (SSA)

```text
pto.tset_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

### AS Level 2 (DPS)

```text
pto.tset_img2col_rpt ins(%cfg : !pto.fmatrix_config) outs()
```

### AS Level 1（SSA）

```text
pto.tset_img2col_rpt %cfg
```

### AS Level 2（DPS）

```text
pto.tset_img2col_rpt ins(%cfg) outs()
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`:

```cpp
template <typename ConvTileData, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_RPT(ConvTileData &src, WaitEvents &... events);

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_RPT(ConvTileData &src, WaitEvents &... events);
```

For `MEMORY_BASE` targets, an overload without `SetFmatrixMode` is also provided.

## 约束

- This instruction is backend-specific and available only for backends that expose IMG2COL configuration state.
- `src` must be a valid IMG2COL configuration tile type accepted by the backend implementation.
- The exact register/metadata fields updated by this instruction are implementation-defined.
- Use this instruction before dependent `TIMG2COL` operations in the same execution stream.

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_set_img2col_rpt(Img2colTileConfig<uint64_t>& cfg) {
  TSET_IMG2COL_RPT(cfg);
}
```
