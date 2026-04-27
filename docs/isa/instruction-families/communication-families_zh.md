# 通信指令族

通信指令族覆盖跨 NPU collective、点到点交换和通知协议。这些操作需要并行组或远端 rank 语义，不能和 tile payload 计算或系统调度协议混在同一个分类中。

## 指令族概览

| 指令族 | 说明 | Profile |
| --- | --- | --- |
| 通信与运行时 | 并行组上的点对点与集合通信 | A2/A3, A5 |

### 通信与运行时

这组操作跨越并行组，需要 `ParallelGroup` 句柄，并引入网络 / 互连可见的排序与副作用。

| 类别 | 操作 |
|------|------|
| 集合广播 | `tbroadcast`、`tscatter`、`tgather` |
| 点对点 | `tget`、`tget_async`、`tput`、`tput_async` |
| 集合归约 | `treduce` |
| 通知协议 | `tnotify`、`ttest`、`twait` |

## 共享操作数模型

- 通信族使用并行组句柄、GM 视图、暂存 tile 和异步事件对象。

## 共享副作用

- 通信族会引入跨 NPU 的排序与可见性副作用。

## 共享约束

- 并行组一致性
- 缓冲区角色与尺寸匹配
- profile 支持子集明确可见

## 不允许的情形

- 集合操作中各 rank 的协议不一致
- 依赖未声明的 backend 便利实现
- 在 CPU simulator 上使用通信指令

## 相关页面

- [通信指令集](./communication-families_zh.md)
- [通信指令参考](../comm/README_zh.md)
- [通信运行时契约](../comm/communication-runtime_zh.md)
