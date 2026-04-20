# PTO Abstract Machine

本目录定义 PTO ISA 与 PTO Tile Lib 使用的**抽象执行模型**。该模型刻意站在"写代码的人"的视角：描述正确程序可以依赖的行为假设，而不要求读者理解每一代设备的微架构细节。

## 按任务选择

| 你的需求 | 从这里开始 |
|----------|----------|
| 理解 core / device / host 分层 | [PTO 机器模型](abstract-machine_zh.md) |

## 文档

- [PTO 机器模型（core/device/host）](abstract-machine_zh.md)

## 与其他文档的关系

| 相关文档 | 内容 |
|----------|------|
| [docs/coding/Tile_zh.md](../coding/Tile_zh.md) | Tile 抽象与编程模型 |
| [docs/coding/GlobalTensor_zh.md](../coding/GlobalTensor_zh.md) | 全局内存张量 |
| [docs/coding/Event_zh.md](../coding/Event_zh.md) | 事件与同步 |
| [docs/coding/Scalar_zh.md](../coding/Scalar_zh.md) | 标量值与枚举 |
| [docs/isa/README_zh.md](../isa/README_zh.md) | 指令参考 |

## 相关文档

- [docs/README_zh.md](../../README_zh.md) — 文档总入口
- [docs/isa/README_zh.md](../isa/README_zh.md) — ISA 参考入口
