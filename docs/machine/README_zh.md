# PTO 抽象机器（Abstract Machine）

本目录定义 PTO ISA 与 PTO Tile Lib 使用的抽象执行模型。该模型刻意站在“写代码的人”的视角：描述正确程序可以依赖的行为假设，而不要求读者理解每一代设备的所有微架构细节。

## 文档

- [PTO 机器模型（core/device/host）](abstract-machine_zh.md)

## 与其他文档的关系

- 数据模型与编程模型：
  - [Tiles](../coding/Tile_zh.md)
  - [全局内存张量](../coding/GlobalTensor_zh.md)
  - [事件与同步](../coding/Event_zh.md)
  - [标量值与枚举](../coding/Scalar_zh.md)
- [指令参考](../isa/scalar/ops/micro-instruction/README_zh.md)
