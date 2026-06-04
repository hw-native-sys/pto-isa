# TCVT

## 指令示意图

![TCVT tile operation](../figures/isa/TCVT.svg)

## 简介

带指定舍入模式的逐元素类型转换。

## 数学语义

对每个元素 `(i, j)` 在有效区域内：

$$ \mathrm{dst}_{i,j} = \mathrm{cast}_{\mathrm{rmode}}\!\left(\mathrm{src}_{i,j}\right) $$

其中 `rmode` 是舍入策略（参见 `pto::RoundMode`）。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tcvt %src{rmode = #pto<round_mode xx>}: !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tcvt ins(%src{rmode = #pto<round_mode xx>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp` 和 `include/pto/common/constants.hpp`：

```cpp
template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode, WaitEvents &... events);

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, WaitEvents &... events);
```

## 约束

- `dst` 和 `src` 必须在形状/有效区域方面兼容，如实现所要求的。
- 对于给定的 `RoundMode`，转换 `(src 元素类型) -> (dst 元素类型)` 必须被目标支持。
- **实现说明 (A2A3/A5)**:
    - 一种形式接受显式的 `SaturationMode`，指定的饱和行为会直接传递给实现。
    - 另一种形式不显式给出 `SaturationMode`；此时实现会针对具体类型对选择目标定义的默认饱和行为。
    - 在 CPU 实现中，目前仅实现了不显式传入 `SaturationMode` 的形式。

## 支持的转换（A2A3 与 A5 并排对比）

| 源类型 | A2A3 目标类型 | A5 目标类型 | 差异 |
|---|---|---|---|
| FP32 | FP16, FP32（仅舍入）, BF16, I16, I32, I64 | FP32, FP16, BF16, I16, I32, I64, FP8_E4M3, FP8_E5M2, H8 | A5 新增 FP8/H8 目标 |
| FP16 | FP32, I32, I16, I8, U8, S4（int4b_t） | FP32, I32, I16, I8, U8, H8 | A2A3 有 S4；A5 有 H8 |
| BF16 | FP32, I32 | FP32, I32, FP16, FP4_E1M2X2, FP4_E2M1X2 | A5 新增 FP16/FP4 目标 |
| I16 | FP16, FP32 | U8, FP16, FP32, U32, I32 | A5 新增 U8/U32/I32 |
| I32 | FP32, I16, I64, FP16（deq 路径） | FP32, I16, U16, I64, U8 | A2A3 支持 I32 -> FP16（half，deq）；A5 不支持 |
| I64 | FP32, I32 | FP32, I32 | 相同 |
| U8 | FP16 | FP16, U16 | A5 新增 U16 |
| I8 | FP16 | FP16, I16, I32 | A5 新增 I16/I32 |
| S4（int4b_t） | FP16 | N/A | A2A3 独有 |
| U32 | N/A | U8, U16, I16 | A5 独有源类型 |
| FP8_E4M3 | N/A | FP32 | A5 独有源类型 |
| FP8_E5M2 | N/A | FP32 | A5 独有源类型 |
| H8 | N/A | FP32 | A5 独有源类型 |
| FP4_E1M2X2 | N/A | BF16 | A5 独有源类型 |
| FP4_E2M1X2 | N/A | BF16 | A5 独有源类型 |

说明：
- 关键差异：A2A3 支持 I32 -> FP16（half，deq 路径），A5 不支持 I32 -> FP16。
- A5 上不支持 FP16 -> FP8_E4M3 和 FP16 -> FP8_E5M2。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  SrcT src;
  DstT dst;
  TCVT(dst, src, RoundMode::CAST_RINT);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, half, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TCVT(dst, src, RoundMode::CAST_RINT);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tcvt %src{rmode = #pto<round_mode xx>}: !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcvt %src{rmode = #pto<round_mode xx>}: !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tcvt %src {rmode = #pto.round_mode<CAST_RINT>} : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcvt ins(%src{rmode = #pto<round_mode xx>}: !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

