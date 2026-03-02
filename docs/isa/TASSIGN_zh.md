# TASSIGN

## 指令示意图

![TASSIGN tile operation](../figures/isa/TASSIGN.svg)

## 简介

将 Tile 对象绑定到实现定义的片上地址（手动放置）。

## 数学语义

Not applicable.

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

`TASSIGN` is typically introduced by bufferization/lowering when mapping SSA tiles to physical storage.

同步形式：

```text
tassign %tile, %addr : !pto.tile<...>, index
```

### IR Level 1（SSA）

```text
pto.tassign %tile, %addr : !pto.tile<...>, dtype
```

### IR Level 2（DPS）

```text
pto.tassign ins(%tile, %addr : !pto.tile_buf<...>, dtype)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename T, typename AddrType>
PTO_INST void TASSIGN(T& obj, AddrType addr);
```

## 约束

- **实现检查**:
  - If `obj` is a Tile:
    - In manual mode (when `__PTO_AUTO__` is not defined), `addr` must be an integral type and is reinterpreted as the tile's storage address.
    - In auto mode (when `__PTO_AUTO__` is defined), `TASSIGN(tile, addr)` is a no-op.
  - If `obj` is a `GlobalTensor`:
    - `addr` must be a pointer type.
    - The pointed-to element type must match `GlobalTensor::DType`.

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT t;
  TASSIGN(t, 0x1000);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(c, 0x3000);
  TADD(c, a, b);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
pto.tassign %tile, %addr : !pto.tile<...>, dtype
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tassign %tile, %addr : !pto.tile<...>, dtype
```

### PTO 汇编形式

```text
tassign %tile, %addr : !pto.tile<...>, index
# IR Level 2 (DPS)
pto.tassign ins(%tile, %addr : !pto.tile_buf<...>, dtype)
```

