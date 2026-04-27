# 标量与控制指令族

标量与控制指令族定义 `pto.*` 外壳操作共享的契约：它们设置状态、建立顺序、配置 DMA，并为 tile / vector 有效载荷提供谓词和控制结构。

## 指令族概览

| 指令族 | 说明 |
| --- | --- |
| 控制与配置 | barrier、yield 和控制外壳 |
| 流水线同步 | `set_flag`、`wait_flag`、`pipe_barrier` 等 |
| DMA 拷贝 | GM↔UB 传输配置与启动 |
| 谓词加载存储 | 谓词相关内存访问 |
| 谓词生成与代数 | 谓词比较和布尔代数 |
| 共享算术 / 共享 SCF | PTO 区域周边的标量运算与结构化控制 |

## 共享操作数模型

- 标量寄存器
- pipe / event 标识
- DMA 循环参数与指针
- 谓词 mask 或控制参数

## 共享副作用

- 建立或消费同步边
- 配置 DMA 状态
- 生成谓词结果
- 改变控制流和标量状态

## 共享约束

- pipe / event 空间必须符合 profile。
- DMA 参数必须满足实现要求。
- 谓词宽度必须与目标操作匹配。
- 顺序边必须与后续 payload 操作对齐。

## 不允许的情形

- 等待未建立事件
- 使用目标不支持的 pipe / event
- 在缺少同步的情况下跨越 producer-consumer 边

## 相关页面

- [标量与控制指令集](scalar-and-control-families_zh.md)
- [标量参考入口](../scalar/README_zh.md)
