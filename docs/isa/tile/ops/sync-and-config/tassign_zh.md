# pto.tassign

`pto.tassign` 属于[同步与配置指令](../../sync-and-config_zh.md)集。

## 概述

`TASSIGN` 把一个 Tile 或 `GlobalTensor` 对象绑定到具体存储地址。它不做算术，也不搬运数据；它做的是"把抽象对象落到某个物理或模拟地址上"。这条指令主要服务于手动放置和手动调度场景。

## 机制

对 Tile 来说，`TASSIGN` 的作用是把内部数据指针指向某个片上地址；对 `GlobalTensor` 来说，则是把对象绑定到一段外部指针地址。它本身没有独立的数学语义，真正重要的是：绑定的是哪类对象、地址是在运行时给出还是在编译期给出、以及当前目标是否允许这种地址落点。

运行时地址形式在 NPU manual 模式下，地址会被直接解释成 tile 存储地址；在 `__PTO_AUTO__` 打开的自动模式下，NPU backend 中的 `TASSIGN(tile, addr)` 当前是空操作。CPU 模拟器不会直接把整型当裸地址使用，而是通过 `NPUMemoryModel` 把它解析到对应架构的模拟缓冲区。

## 语法

### PTO-AS

```text
tassign %tile, %addr : !pto.tile<...>, index
```

### AS Level 1（SSA）

```mlir
pto.tassign %tile, %addr : !pto.tile<...>, dtype
```

### AS Level 2（DPS）

```mlir
pto.tassign ins(%tile, %addr : !pto.tile_buf<...>, dtype)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

### 运行时地址形式

```cpp
template <typename T, typename AddrType>
PTO_INST void TASSIGN(T& obj, AddrType addr);
```

### 编译期地址形式

```cpp
template <std::size_t Addr, typename T>
PTO_INST std::enable_if_t<is_tile_data_v<T> || is_conv_tile_v<T>> TASSIGN(T& obj);
```

编译期地址写法会在编译期执行静态边界与对齐检查，因此更适合固定地址的手动布局。

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `obj` | 输入/输出 | Tile 或 GlobalTensor 对象 |
| `addr` | 输入 | 目标地址（运行时为整型，编译期为模板参数） |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `obj` | Tile/GlobalTensor | 被绑定到指定地址的对象 |

## 副作用

将 Tile 或 GlobalTensor 绑定到指定地址，可能影响后续操作的数据位置。

## 约束

- 运行时地址形式要求 `addr` 是整型地址。
- 在 NPU manual 模式下，地址会被直接解释成 tile 存储地址。
- 在自动模式下，`TASSIGN(tile, addr)` 当前是空操作。
- CPU 模拟器通过 `NPUMemoryModel` 解析地址到模拟缓冲区。
- 对于 GlobalTensor，`addr` 必须是指针类型，且指针指向的元素类型必须和 `GlobalTensor::DType` 一致。
- 编译期地址检查会根据 Tile 的 `Loc` 自动推导对应内存空间，并检查该内存空间是否存在、Tile 是否能放得下、`Addr + tile_size` 是否越界、以及地址是否满足对齐要求。

## 异常与非法情形

- 当指定地址越界时，行为未定义。
- 当指针类型与 `GlobalTensor::DType` 不匹配时，编译错误。
- 当目标架构不支持指定的内存空间时，编译错误。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 运行时地址形式 | 是 | 是 | 是 |
| 编译期地址形式 | 是 | 是 | 是 |
| GlobalTensor 支持 | 是 | 是 | 是 |

## 示例

### C++ 运行时地址

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

### C++ 编译期地址

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_checked() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c;

  TASSIGN<0x0000>(a);
  TASSIGN<0x0400>(b);
  TASSIGN<0x0800>(c);
  TADD(c, a, b);
}
```

### PTO-AS

```text
tassign %tile, %addr : !pto.tile<...>, index
```

## 相关页面

- 指令集总览：[同步与配置](../../sync-and-config_zh.md)
- [TSUBVIEW](./tsubview_zh.md)
- [TALIAS](../../../TALIAS_zh.md)
- [TASSIGN](./tassign_zh.md)
