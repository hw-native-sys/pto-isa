# TASSIGN

## 指令示意图

![TASSIGN tile operation](../figures/isa/TASSIGN.svg)

## 简介

将 Tile 对象绑定到实现定义的片上地址（手动放置）。

## 数学语义

不适用。

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
| Vec | UB | 192KB | 256KB | 128KB | 128KB | 32B |
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
    - 如果 `obj` 是 Tile（含 ConvTile）：
        - 在手动模式下（未定义 `__PTO_AUTO__` 时），`addr` 必须是整数类型，并被重新解释为 tile 的存储地址。
        - 在自动模式下（定义了 `__PTO_AUTO__` 时），`TASSIGN(tile, addr)` 是空操作。
    - 如果 `obj` 是 `GlobalTensor`：
        - `addr` 必须是指针类型。
        - 指向的元素类型必须匹配 `GlobalTensor::DType`。

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

  // 主要触发 [SA-0352]: tile_size (256KB) > UB 容量 (A2A3 为 192KB)
  // （tile 已超过整块缓冲区，SA-0353 越界断言同时成立）
  TASSIGN<0x0>(t);
}
```

```cpp
void example_oob_addr() {
  using TileT = Tile<TileType::Vec, float, 128, 128>;  // 64KB
  TileT t;

  // static_assert 触发 [SA-0353]: 0x20020 + 64KB > 192KB（地址已对齐，仅越界）
  TASSIGN<0x20020>(t);
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
