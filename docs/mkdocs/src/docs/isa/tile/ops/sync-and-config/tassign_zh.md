<!-- Generated from `docs/isa/tile/ops/sync-and-config/tassign_zh.md` -->

# TASSIGN

## 指令示意图

![TASSIGN tile operation](../figures/isa/TASSIGN.svg)

## 简介

将 Tile 对象绑定到实现定义的片上地址（手动放置）。

## 数学语义

Not applicable.

## 汇编语法

PTO-AS 形式：参见 [PTO-AS Specification](../assembly/PTO-AS.md).

`TASSIGN` is typically introduced by bufferization/lowering when mapping SSA tiles to physical storage.

同步形式：

```text
tassign %tile, %addr : !pto.tile<...>, index
```

### AS Level 1 (SSA)

```text
pto.tassign %tile, %addr : !pto.tile<...>, dtype
```

### AS Level 2 (DPS)

```text
pto.tassign ins(%tile, %addr : !pto.tile_buf<...>, dtype)
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

声明于 `include/pto/common/pto_instr.hpp`.

### Form 1: Runtime address

```cpp
template <typename T, typename AddrType>
PTO_INST void TASSIGN(T& obj, AddrType addr);
```

Binds `obj` to the on-chip address `addr`. No compile-time bounds checking is
performed (the address value is not available at compile time).

### Form 2: Compile-time address (with static bounds check)

```cpp
template <std::size_t Addr, typename T>
PTO_INST void TASSIGN(T& obj);
```

Binds `obj` to the on-chip address `Addr`. Because `Addr` is a non-type
template parameter, the compiler performs the following **compile-time** checks
via `static_assert`:

| Check | Condition | Assertion ID | Error message |
|-------|-----------|--------------|---------------|
| Memory space exists | `capacity > 0` | SA-0351 | Memory space is not available on this architecture. |
| Tile fits in memory | `tile_size <= capacity` | SA-0352 | Tile storage size exceeds memory space capacity. |
| Address in bounds | `Addr + tile_size <= capacity` | SA-0353 | addr + tile_size exceeds memory space capacity (out of bounds). |
| Address aligned | `Addr % alignment == 0` | SA-0354 | addr is not properly aligned for the target memory space. |

See `docs/coding/debug.md` (fix recipe `FIX-A12`) for suggested remedies.

The memory space, capacity, and alignment are determined automatically from the
Tile's `TileType` (i.e. `Loc` template parameter):

| TileType | Memory | Capacity (A2A3) | Capacity (A5) | Capacity (Kirin9030) | Capacity (KirinX90) | Alignment |
|----------|--------|-----------------|---------------|----------------------|---------------------|-----------|
| Vec | UB | 192 KB | 256 KB | 128 KB | 128 KB | 32 B |
| Mat | L1 | 512 KB | 512 KB | 512 KB | 1024 KB | 32 B |
| Left | L0A | 64 KB | 64 KB | 32 KB | 64 KB | 32 B |
| Right | L0B | 64 KB | 64 KB | 32 KB | 64 KB | 32 B |
| Acc | L0C | 128 KB | 256 KB | 64 KB | 128 KB | 32 B |
| Bias | Bias | 1 KB | 4 KB | 1 KB | 1 KB | 32 B |
| Scaling | FBuffer | 2 KB | 4 KB | 7 KB | 6 KB | 32 B |
| ScaleLeft | L0A | N/A | 4 KB | N/A | N/A | 32 B |
| ScaleRight | L0B | N/A | 4 KB | N/A | N/A | 32 B |

Capacities can be overridden at build time via `-D` flags (e.g.
`-DPTO_UBUF_SIZE_BYTES=262144`). See `include/pto/common/buffer_limits.hpp`.

**Note:** This overload is only available for `Tile` and `ConvTile` types. For
`GlobalTensor`, use `TASSIGN(obj, pointer)` (Form 1).

## 约束

- **实现检查**:
    - If `obj` is a Tile:
    - In manual mode (when `__PTO_AUTO__` is not defined), `addr` must be an integral type and is reinterpreted as the tile's storage address.
    - In auto mode (when `__PTO_AUTO__` is defined), `TASSIGN(tile, addr)` is a no-op.
    - If `obj` is a `GlobalTensor`:
    - `addr` must be a pointer type.
    - The pointed-to element type must match `GlobalTensor::DType`.

## 示例

### Runtime address (no compile-time check)

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

### Compile-time address (with static bounds check)

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

The following triggers a compile error:

```cpp
void example_oob() {
  // Tile<Vec, float, 256, 256> occupies 256*256*4 = 256KB
  using BigTile = Tile<TileType::Vec, float, 256, 256>;
  BigTile t;

  // static_assert fires: tile_size (256KB) > UB capacity (192KB on A2A3)
  TASSIGN<0x0>(t);
}
```

```cpp
void example_oob_addr() {
  using TileT = Tile<TileType::Vec, float, 128, 128>;  // 64KB
  TileT t;

  // static_assert fires: 0x20000 (128KB) + 64KB = 192KB,
  //                       but 0x20001 + 64KB > 192KB
  TASSIGN<0x20001>(t);
}
```

### Ping-pong L0 buffer allocation

```cpp
void example_pingpong() {
  using L0ATile = TileLeft<half, 64, 128>;   // L0A tile
  using L0BTile = TileRight<half, 128, 64>;  // L0B tile

  L0ATile a0, a1;
  L0BTile b0, b1;

  TASSIGN<0x0000>(a0);   // L0A ping
  TASSIGN<0x8000>(a1);   // L0A pong
  TASSIGN<0x0000>(b0);   // L0B ping  (separate physical memory from L0A)
  TASSIGN<0x8000>(b1);   // L0B pong
}
```
