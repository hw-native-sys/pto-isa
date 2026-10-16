# CV Pipe

## 概述

CV Pipe 用于描述 Cube（AIC）与 Vector（AIV）之间的核内数据通道。

这里只需要关注可直接编写的接口，也就是：

- 如何在两侧声明同一个 pipe
- 如何由生产者 `push`
- 如何由消费者 `pop`
- 如何在消费完成后 `free`

本章不讨论更底层的实现接口。

## CV Pipe 关注什么

CV Pipe 主要描述以下问题：

- 哪一侧是生产者，哪一侧是消费者
- 数据是按 C2V 还是 V2C 方向流动
- 同一个 pipe 的两侧如何通过 `id` 对齐
- 一个条目如何经历 `push -> pop -> free`
- 条目传递的是本地 tile，还是张量视图描述

从用户角度看，CV Pipe 的核心价值是：

- 让 Cube 核和 Vector 核之间建立结构化的数据交换路径
- 用一组固定的接口操作表达生产、消费和回收
- 避免用户自己处理过多底层细节

## 接口

CV Pipe 在两种接口形态下都提供同一组语义操作，命名方式不同：

| 语义 | C++ 内建接口 | IR 操作 |
| --- | --- | --- |
| 初始化 pipe | （由 `TPipe` 模板参数声明） | `pto.aic_initialize_pipe` / `pto.aiv_initialize_pipe` |
| 生产者推送 | `TPUSH(Pipe&, ...)` | `pto.tpush_to_aiv` / `pto.tpush_to_aic` |
| 消费者弹出 | `TPOP(Pipe&, ...)` | `pto.tpop_from_aic` / `pto.tpop_from_aiv` |
| 消费后释放 | `TFREE(Pipe&, ...)` | `pto.tfree_from_aic` / `pto.tfree_from_aiv` |
| 分配全局槽位 | `TALLOC(Pipe&, GlobalData&)` | — |

以下 IR 侧的操作名采用"谁向谁发送"的方式，方向更直观：

- `tpush_to_aiv`：Cube 侧把条目推送给 Vector 侧
- `tpush_to_aic`：Vector 侧把条目推送给 Cube 侧
- `tpop_from_aic`：Vector 侧弹出来自 Cube 侧的条目
- `tpop_from_aiv`：Cube 侧弹出来自 Vector 侧的条目

## 基本编程模型

一个最小的 CV Pipe 使用流程通常包括四步：

1. 在 Cube 侧和 Vector 侧分别初始化同一个 `id` 的 pipe
2. 生产者侧执行 `push`
3. 消费者侧执行 `pop`
4. 消费完成后执行对应的 `free`

其中：

- 初始化决定 pipe 的方向、条目大小和槽位配置
- `push` 表示生产一个条目
- `pop` 表示借出一个可消费条目
- `free` 表示消费结束后归还该条目

## 用户需要遵守的约束

### 两侧 `id` 必须一致

同一个逻辑 pipe 的 Cube 侧和 Vector 侧必须使用同一个 `id`，否则无法正确配对。

### 两侧方向必须一致

若 pipe 声明为 C2V，则：

- Cube 侧负责生产
- Vector 侧负责消费

若声明为 V2C，则方向相反。

### 条目类型必须一致

两侧对同一个 pipe 中条目的理解必须一致，例如：

- 都是同一类本地 tile 对象
- 或都对应兼容的全局张量视图

否则后续验证会失败。

### `pop` 后必须有匹配的 `free`

`pop` 取得的是一个正在被消费的 pipe 条目。消费完成后必须显式 `free`，否则条目不会正确回收。

### `split` / `nosplit` 语义必须一致

如果初始化阶段限制了不分裂使用，那么后续 `push` / `pop` / `free` 也必须满足相同约束。

## 适用场景

CV Pipe 适合以下场景：

- Cube 侧产出中间结果，由 Vector 侧继续处理
- Vector 侧产出数据，再交给 Cube 侧继续计算
- 需要在核内建立稳定的 C2V / V2C 数据通路

如果只是普通的全局内存读写、本地 tile 计算或常规 `tload` / `tstore` 流程，通常不需要使用 CV Pipe。
