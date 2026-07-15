# 标量参数与枚举

许多 PTO 内建接口除了 Tile 外还会带标量参数（例如比较模式、舍入模式、原子模式或字面常量）。

本文档汇总 `include/pto/common/pto_instr.hpp` 公共内建接口中出现的标量/枚举类型。

## 标量值

部分指令直接使用 C++ 基本类型作为标量参数：

- `TADDS/TMULS/TDIVS/TEXPANDS`：标量类型为 `TileData::DType`。
- `TMINS`：标量为模板类型 `T`，必须可转换为 Tile 元素类型。
- `TCI`：标量 `S` 为模板类型 `T`，并且必须与 `TileData::DType` 匹配（由实现中的 `static_assert` 强制）。

## PTO ISA 类型助记符（参考）

ISA 文档会用简短助记符（例如 `fp16`、`s8`）描述指令语义。不同后端在任意时刻可能只支持其中一部分；实现状态可参考 `include/README_zh.md`。

### 整数类型

| 类型 | 助记符 |
|---|---|
| 有符号 | `s4`, `s8`, `s16`, `s32`, `s64` |
| 无符号 | `u4`, `u8`, `u16`, `u32`, `u64` |

### 浮点类型

| 类型 | 助记符 |
|---|---|
| 4-bit float 家族 | `fp4`, `hif4`, `mxfp4` |
| 8-bit float 家族 | `fp8`, `hif8`, `mxfp8` |
| 16-bit float 家族 | `bf16`, `fp16` |
| 32-bit float 家族 | `tf32`, `hf32`, `fp32` |
| 64-bit float | `fp64` |

### 无类型比特宽度

| 类型 | 助记符 |
|---|---|
| typeless bits | `b4`, `b8`, `b16`, `b32`, `b64` |

#### 兼容性规则（ISA 约定）

当两个助记符满足以下条件时，认为二者兼容：

- 位宽相同，并且满足其一：
  - 类型相同；或
  - 同位宽的有符号/无符号整数；或
  - 其中一侧为同位宽的 typeless bits（`b*`）。

这属于文档层面的规则，用于描述指令合法性；具体指令可能进一步限制类型。

## 核心枚举

以下枚举均可通过 `#include <pto/pto-inst.hpp>` 使用。

### `pto::RoundMode`

定义于 `include/pto/common/constants.hpp`。`TCVT` 使用它来指定舍入行为（例如 `RoundMode::CAST_RINT`）。

### `pto::CmpMode`

定义于 `include/pto/common/type.hpp`。`TCMPS`（以及 `TCMP`）使用它进行逐元素比较（`EQ/NE/LT/GT/GE/LE`）。

### `pto::MaskPattern`

定义于 `include/pto/common/type.hpp`。mask-pattern 形式的 `TGATHER` 使用它选择预定义的 0/1 mask 模式。

### `pto::AtomicType`

定义于 `include/pto/common/constants.hpp`。作为 `TSTORE<..., AtomicType::AtomicAdd>`（或 `AtomicNone`）的模板参数。

### `pto::AccToVecMode` 与 `pto::ReluPreMode`

定义于 `include/pto/common/constants.hpp`。用于 `TMOV` 的部分重载：从累加器 Tile 移动到向量/矩阵 Tile 时，选择量化与/或 ReLU 行为。

### `pto::PadValue`

定义于 `include/pto/common/constants.hpp`。属于 `Tile<...>` 模板参数，并被一些实现用于定义有效区域外元素的处理策略（例如 select/copy/pad）。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example(Tile<TileType::Vec, float, 16, 16>& dst,
             Tile<TileType::Vec, float, 16, 16>& src) {
  TCVT(dst, src, RoundMode::CAST_RINT);
  TMINS(dst, src, 0.0f);
}
```
