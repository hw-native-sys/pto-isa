# pto.trandom

`pto.trandom` 属于[不规则与复杂指令](./tile/irregular-and-complex_zh.md)集。

## 概述

使用基于计数器的密码算法在目标 Tile 中生成伪随机数。该指令实现了一个基于计数器的随机数生成器，对于有效区域中的每个元素，它基于密钥和计数器状态，使用可配置轮数的密码类变换生成伪随机值。算法使用 128 位状态（4 × 32 位计数器）、64 位密钥（2 × 32 位字），以及类似 ChaCha 的四分之一轮操作。

## 机制

### 数学语义

对有效区域中的每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \mathrm{CipherRound}^R\left(\mathrm{counter}_{i,j},\ \mathrm{key}\right) $$

其中 $R$ 为轮数（默认 10 轮，可选 7 轮），使用类似 ChaCha 的四分之一轮操作进行密码学变换。

## 语法

### PTO-AS

```text
trandom %dst, %key, %counter : !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.trandom %key, %counter : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.trandom ins(%key, %counter : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/npu/a5/TRandom.hpp`：

```cpp
template <uint16_t Rounds = 10, typename DstTile>
PTO_INST void TRANDOM_IMPL(DstTile &dst, TRandomKey &key, TRandomCounter &counter);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `key` | 输入 | 64 位密钥（2 × 32 位字），包含 `key0` 和 `key1` |
| `counter` | 输入 | 128 位计数器状态（4 × 32 位），每次调用后递增 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 生成的伪随机数，有效区域内有效 |

## 副作用

该操作通过密码学变换更新内部计数器状态。

## 约束

- A5 实现检查：
    - `DstTile::DType` 必须为 `int32_t` 或 `uint32_t`
    - Tile 布局必须为行主序（`DstTile::isRowMajor`）
    - `Rounds` 必须为 7 或 10（默认为 10）
    - `key` 和 `counter` 不能为空
- 有效区域：
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域

## 异常与非法情形

- 若 `key` 或 `counter` 为空，行为未定义
- 若 `DstTile::DType` 不是 `int32_t` 或 `uint32_t`，编译失败
- 若 `Rounds` 不是 7 或 10，编译失败

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 支持 | 是 | 否 | 是 |

## 示例

### C++ 自动模式

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

### C++ 手动模式

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

### PTO-AS

```text
# 自动模式：编译器/运行时管理的布局和调度
%dst = pto.trandom %key, %counter : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>

# 手动模式：在发出指令之前显式绑定资源
# pto.tassign %arg0, @tile(0x3000)
%dst = pto.trandom %key, %counter : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

## 相关页面

- 指令集总览：[不规则与复杂指令](./tile/irregular-and-complex_zh.md)
