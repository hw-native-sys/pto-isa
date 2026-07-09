# TSYNC

## 指令示意图

![TSYNC tile operation](../figures/isa/TSYNC.svg)

## 简介

同步 PTO 执行（等待事件或插入每操作流水线屏障）。

- `TSYNC(events...)` 等待一组显式事件令牌。
- `TSYNC<Op>()` 为单个向量操作类插入流水线屏障。

`include/pto/common/pto_instr.hpp` 中的许多内建函数在发射指令前会在内部调用 `TSYNC(events...)`。

## 数学语义

不适用。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <Op OpCode>
PTO_INST void TSYNC();

template <typename... WaitEvents>
PTO_INST void TSYNC(WaitEvents &... events);
```

## 约束

- **实现检查（`TSYNC<Op>()`）**:
  - **A2A3**：`TSYNC_IMPL<Op>()` 支持 S / V / M / MTE1 / MTE2 / MTE3 / FIX / ALL 流水线（`static_assert` 位于 `include/pto/npu/a2a3/TSync.hpp`）。
  - **A5**：`TSYNC_IMPL<Op>()` 仅支持 MTE2 / MTE3 / ALL 流水线（`static_assert` 位于 `include/pto/npu/a5/TSync.hpp`）。
- **`TSYNC(events...)` 语义**:
  - `TSYNC(events...)` 调用 `WaitAllEvents(events...)`，后者对每个事件令牌调用 `events.Wait()`。在auto模式下是no-op。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto(__gm__ float* in) {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<float, 16, 16, Layout::ND>;
  using GT = GlobalTensor<float, GShape, GStride, Layout::ND>;

  GT gin(in);
  TileT t;
  Event<Op::TLOAD, Op::TADD> e;
  e = TLOAD(t, gin);
  TSYNC(e);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT a, b, c;
  Event<Op::TADD, Op::TSTORE_VEC> e;
  e = TADD(c, a, b);
  TSYNC<Op::TADD>();
  TSYNC(e);
}
```
