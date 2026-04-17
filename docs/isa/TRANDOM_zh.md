# TRANDOM


## Tile Operation Diagram

![TRANDOM tile operation](../figures/isa/TRANDOM.svg)

## 简介

使用基于计数器的密码算法在目标 Tile 中生成随机数。

## 数学解释

该指令实现了一个基于计数器的随机数生成器。对于有效区域中的每个元素，它基于密钥和计数器状态，使用可配置轮数的密码类变换生成伪随机值。

该算法使用：
- 128 位状态（4 × 32 位计数器）
- 64 位密钥（2 × 32 位字）
- 类似 ChaCha 的四分之一轮操作，使用向量指令

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
trandom %dst, %key, %counter : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.trandom %key, %counter : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.trandom ins(%key, %counter : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内置函数

声明于 `include/pto/npu/a5/TRandom.hpp`：

```cpp
template <uint16_t Rounds = 10, typename DstTile>
PTO_INST void TRANDOM_IMPL(DstTile &dst, TRandomKey &key, TRandomCounter &counter);
```

## 约束条件

- **实现检查（A5）**：
    - `DstTile::DType` 必须为以下类型之一：`int32_t`、`uint32_t`。
    - Tile 布局必须为行主序（`DstTile::isRowMajor`）。
    - `Rounds` 必须为 7 或 10（默认为 10）。
    - `key` 和 `counter` 不能为空。
- **有效区域**：
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域。

## 示例

### Auto 模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileT dst;
  TRandomKey key = {0x01234, 0x56789};
  TRandomCounter counter = {0, 0, 0, 0};
  TRANDOM_IMPL(dst, key, counter);
}
```

### Manual 模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileT dst;
  TRandomKey key = {0x01234, 0x56789};
  TRandomCounter counter = {0, 0, 0, 0};
  TASSIGN(dst, 0x0);
  TRANDOM_IMPL<10>(dst, key, counter);
}
```

## 汇编形式示例

### Auto 模式

```text
# Auto 模式：编译器/运行时管理的布局和调度。
%dst = pto.trandom %key, %counter : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual 模式

```text
# Manual 模式：在发出指令之前显式绑定资源。
# Tile 操作数可选：
# pto.tassign %arg0, @tile(0x3000)
%dst = pto.trandom %key, %counter : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
trandom %dst, %key, %counter : !pto.tile<...>
# AS Level 2 (DPS)
pto.trandom ins(%key, %counter : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
