# pto.tsync

`pto.tsync` 属于[同步与配置指令](../../sync-and-config_zh.md)集。

## 概述

`TSYNC` 用于同步 PTO 执行，它有两种形式：`TSYNC(events...)` 等待一组显式事件令牌，`TSYNC<Op>()` 为单个向量操作类插入流水线屏障。许多内建函数在发射指令前会在内部调用 `TSYNC(events...)`。

## 机制

`TSYNC(events...)` 调用 `WaitAllEvents(events...)`，后者对每个事件令牌调用 `events.Wait()`。在自动模式下这是空操作。

`TSYNC<Op>()` 仅为向量流水线操作（`PIPE_V`）插入同步屏障，其他流水线类型不支持。

## 语法

### PTO-AS

```text
tsync %e0, %e1 : !pto.event<...>, !pto.event<...>
```

### AS Level 1（SSA）

SSA 形式不支持显式同步原语。

### AS Level 2（DPS）

```mlir
pto.record_event[src_op, dst_op, eventID]
pto.wait_event[src_op, dst_op, eventID]
pto.barrier(op)
```

支持的 op：TLOAD，TSTORE_ACC，TSTORE_VEC，TMOV_M2L，TMOV_M2S，TMOV_M2B，TMOV_M2V，TMOV_V2M，TMATMUL，TVEC。

`record_event` 和 `wait_event` 应视为 TSYNC 的低层形式。前端 kernel 通常不应手工编写事件连线，而应依赖 `ptoas --enable-insert-sync` 自动插入同步。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <Op OpCode>
PTO_INST void TSYNC();

template <typename... WaitEvents>
PTO_INST void TSYNC(WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `events...` | 输入 | 要等待的事件令牌 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| 无 | - | 同步操作不产生数据结果 |

## 副作用

`TSYNC` 可能阻塞执行流水线直到指定事件完成，或插入硬件屏障。

## 约束

- `TSYNC<Op>()` 仅支持向量流水线操作（通过 `static_assert(pipe == PIPE_V)` 强制执行）。
- `TSYNC(events...)` 在自动模式下是空操作。
- 手动模式下会等待所有指定事件完成。

## 异常与非法情形

- 在非向量流水线上调用 `TSYNC<Op>()` 将导致编译错误。
- 当指定的事件未正确发出时，`TSYNC(events...)` 可能永久阻塞。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `TSYNC(events...)` | 是 | 是 | 是 |
| `TSYNC<Op>()` | 是 | 是 | 是 |

## 示例

### C++ 自动模式

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

### C++ 手动模式

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

### PTO-AS

```text
tsync %e0, %e1 : !pto.event<...>, !pto.event<...>
```

### AS Level 2 (DPS)

```mlir
pto.record_event[src_op, dst_op, eventID]
```

## 相关页面

- 指令集总览：[同步与配置](../../sync-and-config_zh.md)
- [TSYNC](./tsync_zh.md)
