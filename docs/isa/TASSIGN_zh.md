# TASSIGN

## 指令示意图

![TASSIGN tile operation](../figures/isa/TASSIGN.svg)

## 简介

将 Tile 对象绑定到实现定义的片上地址（手动放置）。

## 数学语义

Not applicable.

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

`TASSIGN` is typically introduced by bufferization/lowering when mapping SSA tiles to physical storage.

同步形式：

```text
tassign %tile, %addr : !pto.tile<...>, index
```

### AS Level 1（SSA）

```text
pto.tassign %tile, %addr : !pto.tile<...>, dtype
```

### AS Level 2（DPS）

```text
pto.tassign ins(%tile, %addr : !pto.tile_buf<...>, dtype)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

### 形式 1：运行时地址

```cpp
template <typename T, typename AddrType>
PTO_INST void TASSIGN(T& obj, AddrType addr);
```

将 `obj` 绑定到片上地址 `addr`。不执行编译时边界检查（地址值在编译时不可知）。

### 形式 2：编译时地址（含静态边界检查）

```cpp
template <std::size_t Addr, typename T>
PTO_INST void TASSIGN(T& obj);
```

将 `obj` 绑定到片上地址 `Addr`。由于 `Addr` 是非类型模板参数，编译器通过 `static_assert`
执行以下**编译时**检查：

| 检查项 | 条件 | 断言 ID | 错误信息 |
|--------|------|---------|----------|
| 内存空间存在 | `capacity > 0` | SA-0351 | 当前架构不支持该内存空间。 |
| Tile 可放入内存 | `tile_size <= capacity` | SA-0352 | Tile 存储大小超出内存空间容量。 |
| 地址未越界 | `Addr + tile_size <= capacity` | SA-0353 | addr + tile_size 超出内存空间容量（越界）。 |
| 地址对齐 | `Addr % alignment == 0` | SA-0354 | addr 未按目标内存空间要求对齐。 |

修复建议请参阅 `docs/coding/debug.md`（修复方案 `FIX-A12`）。

内存空间、容量和对齐由 Tile 的 `TileType`（即 `Loc` 模板参数）自动确定：

| TileType | 内存空间 | 容量 (A2A3) | 容量 (A5) | 容量 (Kirin9030) | 容量 (KirinX90) | 对齐 |
|----------|----------|-------------|-----------|------------------|-----------------|------|
| Vec | UB | 192 KB | 256 KB | 128 KB | 128 KB | 32 B |
| Mat | L1 | 512 KB | 512 KB | 512 KB | 1024 KB | 32 B |
| Left | L0A | 64 KB | 64 KB | 32 KB | 64 KB | 32 B |
| Right | L0B | 64 KB | 64 KB | 32 KB | 64 KB | 32 B |
| Acc | L0C | 128 KB | 256 KB | 64 KB | 128 KB | 32 B |
| Bias | Bias | 1 KB | 4 KB | 1 KB | 1 KB | 32 B |
| Scaling | FBuffer | 2 KB | 4 KB | 7 KB | 6 KB | 32 B |
| ScaleLeft | L0A | N/A | 4 KB | N/A | N/A | 32 B |
| ScaleRight | L0B | N/A | 4 KB | N/A | N/A | 32 B |

容量可通过编译标志 `-D` 覆盖（如 `-DPTO_UBUF_SIZE_BYTES=262144`）。详见 `include/pto/common/buffer_limits.hpp`。

**注意：** 该重载仅适用于 `Tile` 和 `ConvTile` 类型。对于 `GlobalTensor`，请使用 `TASSIGN(obj, pointer)`（形式 1）。

## 约束

- **实现检查**:
  - 若 `obj` 为 Tile：
    - 在手动模式下（未定义 `__PTO_AUTO__`），`addr` 必须为整型，将被解释为 Tile 的存储地址。
    - 在自动模式下（已定义 `__PTO_AUTO__`），`TASSIGN(tile, addr)` 为空操作。
  - 若 `obj` 为 `GlobalTensor`：
    - `addr` 必须为指针类型。
    - 被指向的元素类型必须与 `GlobalTensor::DType` 一致。

## 示例

### 运行时地址（无编译时检查）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_runtime() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c;
  TASSIGN(a, 0x1000);
  TASSIGN(b, 0x2000);
  TASSIGN(c, 0x3000);
  TADD(c, a, b);
}
```

### 编译时地址（含静态边界检查）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_checked() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c;

  TASSIGN<0x0000>(a);   // OK: 0x0000 + 1024 <= 192KB
  TASSIGN<0x0400>(b);   // OK: 0x0400 + 1024 <= 192KB
  TASSIGN<0x0800>(c);   // OK: 0x0800 + 1024 <= 192KB
  TADD(c, a, b);
}
```

以下示例触发编译错误：

```cpp
void example_oob() {
  // Tile<Vec, float, 256, 256> 占用 256*256*4 = 256KB
  using BigTile = Tile<TileType::Vec, float, 256, 256>;
  BigTile t;

  // static_assert 触发 [SA-0352]: tile_size (256KB) > UB 容量 (A2A3 为 192KB)
  TASSIGN<0x0>(t);
}
```

```cpp
void example_oob_addr() {
  using TileT = Tile<TileType::Vec, float, 128, 128>;  // 64KB
  TileT t;

  // static_assert 触发 [SA-0353]: 0x20001 + 64KB > 192KB
  TASSIGN<0x20001>(t);
}
```

### Ping-pong L0 缓冲区分配

```cpp
void example_pingpong() {
  using L0ATile = TileLeft<half, 64, 128>;   // L0A tile
  using L0BTile = TileRight<half, 128, 64>;  // L0B tile

  L0ATile a0, a1;
  L0BTile b0, b1;

  TASSIGN<0x0000>(a0);   // L0A ping
  TASSIGN<0x8000>(a1);   // L0A pong
  TASSIGN<0x0000>(b0);   // L0B ping（与 L0A 为不同物理内存）
  TASSIGN<0x8000>(b1);   // L0B pong
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

