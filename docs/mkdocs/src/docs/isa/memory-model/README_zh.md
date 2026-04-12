<!-- Generated from `docs/isa/memory-model/README_zh.md` -->

# 内存模型

本章描述 PTO 的内存一致性模型：可见性与排序规则，涵盖生产者-消费者 Ordering 以及与其他 ISA 操作的关系。

## 本章内容

- [一致性基线](memory-model/consistency-baseline.md) — GM / UB / Tile Buffer 三层内存空间、Program Order / Event Order / Barrier Order 三级 Ordering 分类表、未定义/未指明/实现定义行为的精确区分
- [生产者-消费者排序](memory-model/producer-consumer-ordering.md) — 完整状态机图（IDLE → IN_PROGRESS → COMPLETE）、Tile Surface 和 Vector Surface 的 Ordering 链、跨表面传递规则

## 阅读建议

建议按以下顺序阅读：

1. 先读 [一致性基线](memory-model/consistency-baseline.md)，理解 PTO 的三层内存空间和三 level Ordering
2. 再读 [生产者-消费者排序](memory-model/producer-consumer-ordering.md)，理解 Tile Surface（RecordEvent / TSYNC）和 Vector Surface（set_flag / wait_flag）的具体排序机制

## 章节定位

本章属于手册的第 6 章。建议在阅读指令集章节（第 7 章）之前，先理解内存模型，因为许多指令的行为与内存 Ordering 直接相关，特别是 `TLOAD` / `TSTORE` 和 `copy_gm_to_ubuf` / `copy_ubuf_to_gm` 的同步语义。
