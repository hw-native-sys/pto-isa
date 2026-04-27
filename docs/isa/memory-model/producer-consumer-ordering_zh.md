# 生产者-消费者排序

生产者-消费者排序是解释 PTO 可见性规则最直接的方式。只有当消费者通过当前指令集允许的同步与搬运机制，看见生产者必须公开的写入或状态变化时，程序才是合法的。

## 生产者-消费者状态机

每个数据移动或计算操作都参与一条 producer-consumer 链：

```text
IDLE -> IN_PROGRESS -> COMPLETE -> consumed by next op
```

一个操作被后续操作消费，通常通过以下两种方式之一：

1. 把生产者返回的 `RecordEvent` 作为 `WaitEvents...`
2. 对同一事件发出 `wait_flag`

## Tile 指令排序

对 `pto.t*` 程序，常见顺序链如下：

```text
TLOAD -> Tile Compute -> TSTORE
```

### RecordEvent 链接

```cpp
RecordEvent e0 = TLOAD(a, ga);
RecordEvent e1 = TLOAD(b, gb);
TMATMUL(c, a, b, e0, e1);
RecordEvent e2 = TMATMUL(...);
TSTORE(gc, c, e2);
```

当一个操作带有多个 `WaitEvents...` 时，它必须等待所有这些事件完成后才能开始。

### TSYNC

当不需要细粒度事件链时，可以使用 `TSYNC`：

```cpp
TLOAD(a, ga);
TLOAD(b, gb);
TSYNC();
TADD(c, a, b);
TSYNC();
TSTORE(gc, c);
```

## 向量指令排序

对 `pto.v*` 程序，排序链一般包含显式 DMA：

```text
copy_gm_to_ubuf
  -> set_flag
  -> wait_flag
  -> vlds
  -> vector compute
  -> vsts
  -> set_flag
  -> wait_flag
  -> copy_ubuf_to_gm
```

### Tile 指令与向量指令的差异

| 方面 | Tile 指令 | 向量指令 |
| --- | --- | --- |
| 同步机制 | `RecordEvent`, `TSYNC` | `set_flag` / `wait_flag` |
| 数据路径 | GM ↔ Tile Buffer | GM ↔ UB ↔ Vector Register |
| 隐式顺序 | 同一 tile buffer 内有程序顺序 | DMA 与计算之间没有隐式顺序 |

## 跨指令集交接

当 tile 指令的结果要给向量指令消费，或反过来时，交接必须经过 UB 或 GM：

```text
Tile Buffer -> TSTORE / GM -> copy_gm_to_ubuf -> UB -> vlds
```

因此跨指令集 handoff 不能省略搬运与同步。

## 约束

- 消费者只能在建立了 producer-consumer 边后依赖可见性
- 同一操作的 `RecordEvent` 只能被后续操作消费，不能逆向使用
- 指令集页面和 per-op 页面必须显式说明各自需要的顺序机制

## 不允许的情形

- 只写“消费者可见”，却不说明生产者如何建立可见性
- 把某个目标偶然更强的顺序当作架构契约
- 省略跨指令集 handoff 的搬运和同步
- 在 `copy_gm_to_ubuf` 完成前直接 `vlds`
- 在 `vsts` 完成前直接 `copy_ubuf_to_gm`

## 相关页面

- [一致性基线](./consistency-baseline_zh.md)
- [顺序与同步](../machine-model/ordering-and-synchronization_zh.md)
- [Tile 指令集](../instruction-families/tile-families_zh.md)
- [向量指令集](../vector/README_zh.md)
