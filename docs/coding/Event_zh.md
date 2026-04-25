# 事件与同步

PTO Tile Lib 支持显式事件（event）模型，用于表达操作之间的依赖关系，而不必为每条指令都引入全局屏障。

本文档描述 `include/pto/common/pto_instr.hpp` 与 `include/pto/common/event.hpp` 中使用的 C++ 事件类型。

> 注意：具体的 `pto::Event<SrcOp, DstOp>` 类型仅在设备构建（`__CCE_AICORE__`）中定义。CPU 仿真后端通常将 `TSYNC` 视为 no-op，并依赖单线程的普通程序顺序来验证语义。

## 关键类型

### `pto::Op`

`pto::Op` 是类似 opcode 的枚举，用于对操作分类。每个 `Op` 映射到一个硬件流水线（`PIPE_V`、`PIPE_MTE2` 等）。

### `pto::RecordEvent`

许多内建接口（例如 `TADD`、`TLOAD`、`TSTORE`）会返回 `pto::RecordEvent`。它是一个标记值，可赋给 `Event<SrcOp, DstOp>`，用于在该 op 结束后记录一个 token。

### `pto::Event<SrcOp, DstOp>`（仅设备）

在设备构建（`__CCE_AICORE__`）中，`include/pto/common/event.hpp` 定义：

```cpp
template <Op SrcOp, Op DstOp>
struct Event {
  void Wait();
  void Record();
  Event& operator=(RecordEvent);
};
```

- `Wait()`：阻塞直到 producer 侧 token 满足。
- `Record()`：在 producer 流水线上设置 token。
- `evt = OP(...)`：从 `RecordEvent` 赋值会自动记录 token。

模板参数编码 producer/consumer 的 opcode，用于选择正确的流水线对。

### `TSYNC<OpCode>()`（单流水线屏障）

`TSYNC<OpCode>()` 是单 op 的屏障形式，由 `TSYNC_IMPL<OpCode>()` 实现：

- 在设备上，当前实现将单 op 形式限制在向量流水线 op（`PIPE_V`）上。
- 在 CPU 仿真后端（`__CPU_SIM`）中，`TSYNC_IMPL` 为 no-op。

## 内建接口中的 `WaitEvents&...` 机制

`include/pto/common/pto_instr.hpp` 中多数内建接口在参数末尾带有 `WaitEvents&... events` 可变参包，模式为：

- 内建接口调用 `TSYNC(events...)`。
- `TSYNC(events...)` 调用 `WaitAllEvents(events...)`，对每个 event 调用 `events.Wait()`。
- 指令执行后，内建接口返回 `RecordEvent`。

这支持一种“SSA 风格”的 C++ 写法：

1. 将 event token 作为 C++ 变量保存。
2. 将它们传给下一条 op 以表达顺序约束。
3. 将 op 返回的 `RecordEvent` 赋给 event 变量以记录新 token。

## 抽象层面的顺序建议

事件主要用于表达**流水线类之间**的顺序（例如“内存加载完成后向量计算才能消费该 Tile”）。

- 没有显式数据或事件依赖的操作，在设备上可能乱序执行。
- 通过事件建立依赖链的操作，必须满足 `Wait()`/`Record()` 所隐含的顺序。

当顺序对正确性有影响时，`docs/isa/` 指令页会说明相关约束（中文参考页参见 `docs/isa/README_zh.md`）。

## 最小示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void pipeline(__gm__ float* in0, __gm__ float* in1, __gm__ float* out) {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<float, 16, 16, Layout::ND>;
  using GT = GlobalTensor<float, GShape, GStride, Layout::ND>;

  GT gin0(in0), gin1(in1), gout(out);
  TileT a, b, c;

  Event<Op::TLOAD, Op::TADD> e0;
  Event<Op::TLOAD, Op::TADD> e1;
  Event<Op::TADD, Op::TSTORE_VEC> e2;

  e0 = TLOAD(a, gin0);
  e1 = TLOAD(b, gin1);
  e2 = TADD(c, a, b, e0, e1);
  TSTORE(gout, c, e2);
}
```

