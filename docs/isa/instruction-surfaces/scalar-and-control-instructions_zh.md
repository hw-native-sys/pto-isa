# 标量与控制指令集

`pto.*` 中的标量与控制指令集负责建立执行外壳：配置、同步、DMA、谓词和控制流。它们围绕 tile 与 vector 有效载荷工作，而不是直接产出 tile / vector payload。

## 指令集概览

标量与控制指令不直接生成 tile 或向量结果，它们主要生成：

- 控制效果（barrier、控制流推进、状态更新）
- 明确的 producer-consumer 顺序边
- 谓词 mask
- DMA 参数与 DMA 状态
- 标量结果值

标量操作数是单值的，它们的职责是把 tile / vector 这两条 payload 路径串起来。

## 指令分类

| 类别 | 说明 | 示例 |
|------|------|------|
| 控制与配置 | NOP、barrier、yield 以及模式配置 | `nop`、`barrier`、`yield`、`tsethf32mode`、`tsetfmatrix` |
| 流水线同步 | 跨流水线的事件同步与 barrier | `set_flag`、`wait_flag`、`pipe_barrier` |
| DMA 拷贝 | GM↔向量 tile buffer 的搬运配置与发起 | `copy_gm_to_ubuf`、`copy_ubuf_to_gm`、`set_loop_size_outtoub` |
| 谓词加载存储 | 谓词在内存与谓词寄存器之间搬运 | `pld`、`plds`、`pst`、`pstu` |
| 谓词生成与代数 | 构造 tail mask、布尔组合、谓词重排 | `pset_b8`、`pge_b8`、`plt_b8`、`pand`、`por`、`pxor` |

## 输入

标量与控制指令常见输入包括：

- 标量寄存器或立即数
- pipe 标识：`PIPE_MTE1`、`PIPE_MTE2`、`PIPE_MTE3`、`PIPE_V`、`PIPE_M`
- event 标识：`EVENT_ID0`～`EVENT_ID15`（具体范围受 profile 约束）
- buffer id
- 指针：`!pto.ptr<T, ub>`、`!pto.ptr<T, gm>`
- DMA loop size / stride 参数

## 预期输出

标量与控制指令会产生：

- 控制状态变化
- 显式同步边
- 谓词 mask
- 已配置好的 DMA 状态
- 标量结果

## 副作用

| 类别 | 架构副作用 |
|------|------------|
| 流水线同步 | 建立 producer-consumer 顺序，或阻塞某个流水线 / 执行单元 |
| DMA 拷贝 | 发起 GM 与向量 tile buffer 之间的数据搬运 |
| 谓词加载存储 | 读写谓词对应的内存表示 |

## 事件模型

标量 / 控制同步使用显式事件模型。一个事件由 `(src_pipe, dst_pipe, event_id)` 三元组标识：

| 字段 | 典型取值 | 含义 |
|------|----------|------|
| `src_pipe` | `PIPE_MTE1`、`PIPE_MTE2`、`PIPE_MTE3`、`PIPE_V`、`PIPE_M` | 产生事件的流水线 |
| `dst_pipe` | `PIPE_MTE1`、`PIPE_MTE2`、`PIPE_MTE3`、`PIPE_V`、`PIPE_M` | 消费事件的流水线 |
| `event_id` | 0–15（取决于目标 profile） | 事件槽编号 |

```text
Producer pipe                               Consumer pipe
   │                                            │
   │  发起 DMA 或计算                            │
   ▼                                            │
set_flag(src_pipe, dst_pipe, EVENT_ID)          │
   │                                            │
   │                               wait_flag(src_pipe, dst_pipe, EVENT_ID)
   │                                            │
   ▼                                            ▼
结果或数据可见                                 后续操作继续
```

## 不同 profile 的 pipe 空间

| Pipe | CPU Sim | A2A3 | A5 |
|------|:-------:|:----:|:--:|
| `PIPE_MTE1` | 模拟 | 支持 | 支持 |
| `PIPE_MTE2` | 模拟 | 支持 | 支持 |
| `PIPE_MTE3` | 模拟 | 支持 | 支持 |
| `PIPE_V` | 模拟 | 桥接/模拟 | 原生 |
| `PIPE_M` | 模拟 | 支持 | 支持 |

## 约束

- pipe / event 空间必须符合所选 profile。
- `set_flag` / `wait_flag` 必须成对使用。
- DMA 参数必须在发起传输前配置完成。
- 谓词宽度必须和目标操作匹配。
- 依赖外壳不能跳过：例如，在 `copy_gm_to_ubuf` 完成前就直接执行依赖其结果的 `vlds` 是非法的。

## 不允许的情形

- 等待一个从未建立的事件
- 使用目标 profile 不支持的 pipe 或 event 标识
- 配置不自洽的 DMA 参数
- 混用和目标向量宽度不匹配的谓词宽度
- 没有正确等待就跨越 DMA / vector / tile 顺序边

## 语法

### PTO-AS 形式

```asm
set_flag PIPE_MTE2, PIPE_V, EVENT_ID0
wait_flag PIPE_MTE2, PIPE_V, EVENT_ID0
pipe_barrier PIPE_V
```

### SSA 形式（AS Level 1）

```mlir
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.pipe_barrier["PIPE_V"]
```

完整语法见 [汇编拼写与操作数](../syntax-and-operands/assembly-model_zh.md)。

## C++ 内建接口

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

PTO_INST void set_flag(pipe_t src_pipe, pipe_t dst_pipe, event_t event_id);
PTO_INST void wait_flag(pipe_t src_pipe, pipe_t dst_pipe, event_t event_id);
PTO_INST void pipe_barrier(pipe_t pipe);

PTO_INST void copy_gm_to_ubuf(ub_ptr dst, gm_ptr src, uint64_t sid,
                              uint64_t n_burst, uint64_t len_burst,
                              uint64_t dst_stride, uint64_t src_stride);

PTO_INST void copy_ubuf_to_gm(gm_ptr dst, ub_ptr src, uint64_t sid,
                              uint64_t n_burst, uint64_t len_burst,
                              uint64_t reserved, uint64_t dst_stride, uint64_t src_stride);
```

## 相关页面

- [标量与控制指令族](../instruction-families/scalar-and-control-families_zh.md)
- [标量参考入口](../scalar/README_zh.md)
- [顺序与同步](../machine-model/ordering-and-synchronization_zh.md)
