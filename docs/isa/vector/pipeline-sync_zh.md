# 向量指令集：流水线同步

PTO指令集架构中的 `pto.v*` 同步原语定义在这里。这些操作描述了向量流水线与 DMA 流水线之间的可见契约，也记录了当前 A5 风格目标配置中后端必须保留的同步语义。

> **类别：** 协调流水线执行的同步原语
> **流水线：** MTE2（GM→向量 tile buffer）、`PIPE_V`（向量）、MTE3（向量 tile buffer→GM）

向量路径运行在访问与执行解耦的机器模型上。MTE 与 `PIPE_V` 可以并发推进，因此任何跨流水线的数据依赖都必须显式建模，否则就会出现“数据尚未到达就开始计算”或“结果尚未写回就开始下一阶段搬运”的风险。

---

## 核内流水线同步

这组操作用来协调同一向量核心内部不同流水线之间的数据流。

### `pto.set_flag`

- **语法：** `pto.set_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- **语义：** 源流水线向目标流水线发送一个事件信号。

```c
set_flag(src_pipe, dst_pipe, event_id);
```

典型用法是 MTE2 完成 GM→向量 tile buffer 搬运后，通知 `PIPE_V` 可以开始消费数据：

```mlir
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

### `pto.wait_flag`

- **语法：** `pto.wait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- **语义：** 目标流水线阻塞，直到接收到匹配的事件。

```c
wait_flag(src_pipe, dst_pipe, event_id);
```

```mlir
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

### `pto.pipe_barrier`

- **语法：** `pto.pipe_barrier "PIPE_*"`
- **语义：** 排空指定流水线中所有已发射但尚未完成的操作。屏障前的操作必须全部完成，屏障后的操作才能开始。

```c
pipe_barrier(pipe);
```

支持的流水线标识包括 `PIPE_MTE2`、`PIPE_V`、`PIPE_MTE3`。

下面这个例子里，两次 `copy_ubuf_to_gm` 都写向同一个 GM 地址。如果不插入 `pipe_barrier`，MTE3 有可能重排它们，导致最终结果不确定：

```mlir
pto.copy_ubuf_to_gm %ub_partial_0, %gm_result, ...
pto.pipe_barrier "PIPE_MTE3"
pto.copy_ubuf_to_gm %ub_partial_1, %gm_result, ...
```

### `pto.get_buf`

- **语法：** `pto.get_buf "PIPE_*", %buf_id, %mode : i64, i64`
- **语义：** 获取一个 buffer 槽位，用于跨流水线双缓冲或多缓冲协调。

```c
get_buf(pipe, buf_id, mode);
```

### `pto.rls_buf`

- **语法：** `pto.rls_buf "PIPE_*", %buf_id, %mode : i64, i64`
- **语义：** 释放先前获取的 buffer 槽位，让其他流水线可以继续使用这个槽位。

```c
rls_buf(pipe, buf_id, mode);
```

### `pto.mem_bar`

- **语法：** `pto.mem_bar "BARRIER_TYPE"`
- **语义：** 在 `__VEC_SCOPE__` 范围内建立向量内存栅栏。这个操作只约束向量侧内存可见性，不替代跨流水线的生产者/消费者顺序。

```c
mem_bar(barrier_type);
```

| 类型 | 语义 |
|------|------|
| `VV_ALL` | 所有先前向量指令完成后，后续向量指令才能开始 |
| `VST_VLD` | 所有先前向量存储对后续向量加载可见 |
| `VLD_VST` | 所有先前向量加载完成后，后续向量存储才能开始 |

如果同一块向量 tile buffer 地址既被 `vsts` 写入、又被后续 `vlds` 读取，就需要合适的 `mem_bar`：

```mlir
pto.vsts %v0, %ub[%c0] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>
pto.mem_bar "VST_VLD"
%v1 = pto.vlds %ub[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

---

## 核内同步模式与示例

### 示例 1：`set_flag` / `wait_flag`

最直接的做法是：每一条跨流水线依赖边都用一组显式事件来表示，生产者发信号，消费者等待。

```mlir
pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr, ...
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
scf.for %dummy = %c0 to %c1 step %c1 {
  %v = pto.vlds %ub_ptr[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
} {llvm.loop.aivector_scope}

pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.copy_ubuf_to_gm %ub_out, %gm_out, ...
```

这种写法的优点是因果关系非常清楚；缺点是只要跨流水线边一多，事件数量也会线性膨胀。

### 示例 2：`get_buf` / `rls_buf`

另一条路是不用显式事件，而是让流水线围绕同一个 buffer ID 做获取与释放。只要多个阶段按程序顺序围绕同一 buffer 进行 acquire / release，RAW 和 WAR 依赖就会自动体现出来。

```mlir
pto.get_buf "PIPE_MTE2", %bufid_ub_ptr, %mode : i64, i64
pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr, ...
pto.rls_buf "PIPE_MTE2", %bufid_ub_ptr, %mode : i64, i64

pto.get_buf "PIPE_V", %bufid_ub_ptr, %mode : i64, i64
pto.get_buf "PIPE_V", %bufid_ub_out, %mode : i64, i64
scf.for %dummy = %c0 to %c1 step %c1 {
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
  %v = pto.vlds %ub_ptr[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
} {llvm.loop.aivector_scope}
pto.rls_buf "PIPE_V", %bufid_ub_ptr, %mode : i64, i64
pto.rls_buf "PIPE_V", %bufid_ub_out, %mode : i64, i64

pto.get_buf "PIPE_MTE3", %bufid_ub_out, %mode : i64, i64
pto.copy_ubuf_to_gm %ub_out, %gm_out, ...
pto.rls_buf "PIPE_MTE3", %bufid_ub_out, %mode : i64, i64
```

这套协议不需要给每条依赖边单独分配事件 ID，更适合循环中的多缓冲流水。

### 示例 3：ping/pong 双缓冲循环

双缓冲的核心思想是让连续迭代交替使用两个 buffer。这样 MTE2、`PIPE_V`、MTE3 虽然都出现在同一个循环体里，但不同迭代实际上在不同 buffer 上并行推进。

#### 3a. 用 `set_flag` / `wait_flag`

假设有两个 ping/pong buffer，同时存在两类流水线配对关系：

- MTE2 ↔ `PIPE_V`，负责输入 buffer
- `PIPE_V` ↔ MTE3，负责输出 buffer

这意味着总共要管理 8 个事件 ID：

| 事件 ID | 方向 | 用途 |
|---------|------|------|
| `EVT_IN_FWD_0` | MTE2 → V | `buf_in[0]` 数据准备好 |
| `EVT_IN_FWD_1` | MTE2 → V | `buf_in[1]` 数据准备好 |
| `EVT_IN_REV_0` | V → MTE2 | `buf_in[0]` 已被向量侧读完 |
| `EVT_IN_REV_1` | V → MTE2 | `buf_in[1]` 已被向量侧读完 |
| `EVT_OUT_FWD_0` | V → MTE3 | `buf_out[0]` 结果准备好 |
| `EVT_OUT_FWD_1` | V → MTE3 | `buf_out[1]` 结果准备好 |
| `EVT_OUT_REV_0` | MTE3 → V | `buf_out[0]` 已被 MTE3 读完 |
| `EVT_OUT_REV_1` | MTE3 → V | `buf_out[1]` 已被 MTE3 读完 |

这种方案还有两个额外的收尾动作：

- **入环前 priming：** 先给所有反向依赖发一次初始 `set_flag`，否则第一轮迭代的 `wait_flag` 会因为没人发过信号而死锁。
- **出环后 draining：** 最后几轮循环发出的信号在循环内部不会再被消费，必须在循环外补上对应的 `wait_flag`。

这就是显式事件模型在复杂循环里容易变重的原因。

#### 3b. 用 `get_buf` / `rls_buf`

同一个 ping/pong 双缓冲，如果改用 acquire / release，就不再需要入环 priming 和出环 draining：

```mlir
scf.for %i = %c0 to %N step %c1 {
  pto.get_buf %bufid_buf[%pp], "PIPE_MTE2"
  pto.copy_gm_to_ubuf %gm_ptr[%i], %ub_buf[%pp], ...
  pto.rls_buf %bufid_buf[%pp], "PIPE_MTE2"

  pto.get_buf %bufid_buf[%pp], "PIPE_V"
  pto.get_buf %bufid_out[%pp], "PIPE_V"
  scf.for %dummy = %c0 to %c1 step %c1 {
    %v = pto.vlds %ub_buf[%pp][%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
    pto.vsts %abs, %ub_out[%pp][%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
  } {llvm.loop.aivector_scope}
  pto.rls_buf %bufid_buf[%pp], "PIPE_V"
  pto.rls_buf %bufid_out[%pp], "PIPE_V"

  pto.get_buf %bufid_out[%pp], "PIPE_MTE3"
  pto.copy_ubuf_to_gm %ub_out[%pp], %gm_out[%i], ...
  pto.rls_buf %bufid_out[%pp], "PIPE_MTE3"
}
```

自动建立的依赖关系是：

- **RAW（MTE2→V）：** 向量侧的 `get_buf` 会阻塞到 MTE2 完成 `rls_buf`
- **WAR（V→MTE2）：** 两轮之后 MTE2 再次获取同一个 buffer 时，会阻塞到向量侧释放它
- **第一轮迭代：** buffer 初始为空闲状态，因此第一次 `get_buf` 不需要额外 priming

---

## 两种同步模型的对比

| 方面 | `set_flag` / `wait_flag` | `get_buf` / `rls_buf` |
|------|---------------------------|-----------------------|
| 依赖表达方式 | 显式事件 | 通过 buffer acquire / release 隐式表达 |
| ID 成本 | buffer 越多，事件 ID 越多 | 依赖由 buffer ID 复用，扩展性更好 |
| 反向依赖（WAR） | 需要额外事件对 | 自动体现 |
| 入环准备 | 需要 priming | 不需要 |
| 出环收尾 | 需要 drain | 不需要 |
| 直线代码 | 简洁直观 | 稍显啰嗦 |
| 双缓冲或多缓冲循环 | 管理成本高 | 更自然 |

如果只是少量阶段串接，`set_flag` / `wait_flag` 很直接；如果在写双缓冲、多缓冲、流水重叠循环，`get_buf` / `rls_buf` 通常更稳。

---

## 核间同步

> **说明：** 核间同步只在混合 Cube + Vector 的任务里才重要。如果任务纯粹是 `pto.v*` 向量路径，可以跳过这一节。

一个 core cluster 通常由 1 个 Cube block 和 2 个 Vector subblock 组成，三者都有各自的 SU（Sequencer Unit），也都有相对独立的流水线。因此 Cube 与 Vector 之间的数据交接，不能靠核内事件直接覆盖。

### 平台差异：A2A3 与 A5

| 方面 | A2A3（Ascend 910B / 910C） | A5（Ascend 950 PR / DT） |
|------|----------------------------|--------------------------|
| 发信号操作 | `set_cross_core` | `set_intra_block` |
| 等待操作 | `wait_flag_dev` | `wait_intra_core` |
| 等待粒度 | SU 级阻塞，整个核都停 | 按指定 pipeline 阻塞 |
| 信号资源 | 每 cluster 16 个 ID | 16 个物理 ID，对外暴露 32 个地址 |
| Cube→Vector | 一次广播到两个向量 subblock | 必须分别给 AIV0 / AIV1 发信号 |
| Vector→Cube | 一次 wait 可归并两个 subblock | Cube 侧要分别 wait 两个 subblock |

### A2A3：`set_cross_core` / `wait_flag_dev`

```c
set_cross_core(pipe, semaphore_id);
wait_flag_dev(semaphore_id);
```

A2A3 的 `mode2` 语义允许在 1:2 的 cluster 里做广播 / 归并：

- Cube→Vector：一次 `set_cross_core` 可以同时把信号送到 AIV0 与 AIV1
- Vector→Cube：Cube 侧一次 `wait_flag_dev` 可以等待两个向量 subblock 都到位

#### `pto.set_cross_core`

- **语法：** `pto.set_cross_core %core_id, %event_id : i64, i64`
- **语义：** 向另一个核心发出事件。在 A2A3 的 1:2 cluster 上使用 mode2 语义。

#### `pto.wait_flag_dev`

- **语法：** `pto.wait_flag_dev %core_id, %event_id : i64, i64`
- **语义：** 等待来自其他核心的事件。阻塞粒度是 SU 级，也就是整个核心上的执行序列都会停住。

### A5：`set_intra_block` / `wait_intra_core`

```c
set_intra_block(trigger_pipe, semaphore_id);
wait_intra_core(wait_pipe, semaphore_id);
```

A5 没有 A2A3 那种对双 subblock 的硬件广播 / 归并语义。它提供的是 1:1 信号：

- ID `0–15` 对应 AIV0
- ID `16–31`（常写作原 ID 加 15 偏移）对应 AIV1

因此：

- Cube→Vector：需要分别对 AIV0、AIV1 做两次 `set_intra_block`
- Vector→Cube：Cube 侧要分别对两个 ID 做两次 `wait_intra_core`

#### `pto.set_intra_block`

- **语法：** `pto.set_intra_block %block_id, %event_id : i64, i64`
- **语义：** 在 block 内部发送事件。A5 下这个事件是按 subblock 单播的，不会自动广播到另一个向量 subblock。

#### `pto.wait_intra_core`

- **语法：** `pto.wait_intra_core %block_id, %event_id : i64, i64`
- **语义：** 等待 block 内部事件。它只阻塞指定的 pipeline，不会让整个 SU 停住。

### 等待粒度的差别

```text
A2A3 的 wait_flag_dev：
    SU 级阻塞，PIPE_MTE2 / PIPE_V / PIPE_MTE3 都会一起停住

A5 的 wait_intra_core "PIPE_MTE2"：
    只阻塞 PIPE_MTE2
    SU、PIPE_V、PIPE_MTE3 仍然可以继续运行
```

这也是 A5 在混合流水重叠时更灵活的原因之一：等待某个外部生产者时，不必把整条执行序列一并冻结。

---

## 相关页面

- [向量 DMA 拷贝](./dma-copy_zh.md)
- [向量加载与存储](./vector-load-store_zh.md)
- [排序与同步](../machine-model/ordering-and-synchronization_zh.md)
