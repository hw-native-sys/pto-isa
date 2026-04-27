# pto.tassign

`pto.tassign` is part of the [Sync And Config](../../sync-and-config.md) instruction set.

## Summary

Bind a Tile object to an on-chip physical address. In Manual mode, this is how the author controls which tile buffer holds which logical tile variable.

## Mechanism

`TASSIGN` maps a tile variable to a physical storage address within the tile buffer space (UB, L0A, L0B, L0C, etc.). The mapping is immediate and persists until the tile is rebound or the program ends.

Without `TASSIGN`, the compiler/runtime assigns tile buffers automatically (Auto mode). With `TASSIGN`, the author overrides this assignment for Manual mode code.

The physical address is interpreted by the tile buffer controller on the AI Core. On A2/A3, physical addresses are byte offsets within the declared memory space (UB, L0A, L0B, L0C, Bias, or FBuffer) and the controller routes each access to the appropriate physical buffer based on the address range. On A5, physical addresses follow the same byte-offset model but the address space may be larger (e.g., 256 KB UB vs. 192 KB on A2/A3); the controller similarly routes based on declared capacity and alignment boundaries. On the CPU simulator, addresses are validated against the same logical capacity/alignment model but resolve to heap allocations with no hardware routing.

## Syntax

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

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`.

### Form 1: Runtime address

```cpp
template <typename T, typename AddrType>
PTO_INST void TASSIGN(T& obj, AddrType addr);
```

Binds `obj` to the on-chip address `addr`. The address value is interpreted at runtime; no compile-time bounds checking is performed.

### Form 2: Compile-time address (with static bounds check)

```cpp
template <std::size_t Addr, typename T>
PTO_INST void TASSIGN(T& obj);
```

Binds `obj` to the on-chip address `Addr`. Because `Addr` is a non-type template parameter, the compiler performs static bounds checks at compile time:

| Check | Condition | Error ID | Error Message |
|-------|-----------|----------|---------------|
| Memory space exists | `capacity > 0` | SA-0351 | Memory space is not available on this architecture. |
| Tile fits in memory | `tile_size <= capacity` | SA-0352 | Tile storage size exceeds memory space capacity. |
| Address in bounds | `Addr + tile_size <= capacity` | SA-0353 | addr + tile_size exceeds memory space capacity (out of bounds). |
| Address aligned | `Addr % alignment == 0` | SA-0354 | addr is not properly aligned for the target memory space. |

The memory space, capacity, and alignment are determined from the tile's `TileType`:

| TileType | Buffer | Capacity (A2A3) | Capacity (A5) | Capacity (Kirin9030) | Capacity (KirinX90) | Alignment |
|----------|--------|:---------:|:---------:|:----------:|:----------:|:---------:|
| Vec | UB | 192 KB | 256 KB | 128 KB | 128 KB | 32 B |
| Mat | L1 | 512 KB | 512 KB | 512 KB | 1024 KB | 32 B |
| Left | L0A | 64 KB | 64 KB | 32 KB | 64 KB | 32 B |
| Right | L0B | 64 KB | 64 KB | 32 KB | 64 KB | 32 B |
| Acc | L0C | 128 KB | 256 KB | 64 KB | 128 KB | 32 B |
| Bias | Bias | 1 KB | 4 KB | 1 KB | 1 KB | 32 B |
| Scaling | FBuffer | 2 KB | 4 KB | 7 KB | 6 KB | 32 B |
| ScaleLeft | L0A | N/A | 4 KB | N/A | N/A | 32 B |
| ScaleRight | L0B | N/A | 4 KB | N/A | N/A | 32 B |

Capacities can be overridden at build time via `-D` flags (e.g., `-DPTO_UBUF_SIZE_BYTES=262144`). See `include/pto/common/buffer_limits.hpp`.

Form 2 is only available for `Tile` and `ConvTile` types. For `GlobalTensor`, use Form 1.

## Inputs

| Operand | Description |
|---------|-------------|
| `tile` | The tile object to bind |
| `addr` | The on-chip address to bind (runtime value or compile-time constant) |

## Expected Outputs

None. The binding takes effect immediately; subsequent tile operations on `tile` use the bound physical address.

## Side Effects

Binds the tile variable to a physical address. Using the same physical address for two non-alias tiles simultaneously produces undefined results.

## Constraints

- Two non-alias tiles must not use the same physical address without an intervening `TSYNC`.
- Configuration state must only be used where later instructions document the dependency.
- Programs must not treat manual placement as a portable substitute for legal PTO behavior.

## Exceptions

- In Auto mode, `TASSIGN(tile, addr)` is a no-op because the compiler/runtime manages placement.
- If `obj` is a `GlobalTensor`, `addr` must be a pointer type matching `GlobalTensor::DType`.
- Using `TASSIGN` with an out-of-bounds address triggers a compile-time error (Form 2) or undefined behavior (Form 1).

## Examples

### Runtime address

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
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

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c;
  TASSIGN<0x0000>(a);    // OK: 0 + 1024 <= 192KB
  TASSIGN<0x0400>(b);    // OK: 0x0400 + 1024 <= 192KB
  TASSIGN<0x0800>(c);    // OK: 0x0800 + 1024 <= 192KB
  TADD(c, a, b);
}
```

Compile-time errors for out-of-bounds placement:

```cpp
// Compile error: tile size exceeds buffer capacity
void bad1() {
  using BigTile = Tile<TileType::Vec, float, 256, 256>;  // 256KB
  BigTile t;
  TASSIGN<0x0>(t);  // SA-0352: exceeds 192KB on A2A3
}

// Compile error: address + tile size exceeds capacity
void bad2() {
  using TileT = Tile<TileType::Vec, float, 128, 128>;  // 64KB
  TileT t;
  TASSIGN<0x20001>(t);  // SA-0353: 0x20001 + 64KB > 192KB
}
```

### Ping-pong L0 buffer allocation

```cpp
void pingpong() {
  using L0ATile = TileLeft<half, 64, 128>;
  using L0BTile = TileRight<half, 128, 64>;
  L0ATile a0, a1;
  L0BTile b0, b1;
  TASSIGN<0x0000>(a0);   // L0A ping
  TASSIGN<0x8000>(a1);   // L0A pong
  TASSIGN<0x0000>(b0);   // L0B ping (separate physical memory)
  TASSIGN<0x8000>(b1);   // L0B pong
}
```

## See Also

- Instruction set overview: [Sync And Config](../../sync-and-config.md)
- Previous op in instruction set: [pto.tsync](./tsync.md)
- Next op in instruction set: [pto.settf32mode](./settf32mode.md)
