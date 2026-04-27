# Vector Predicate And Materialization

Predicate and materialization 指令使用 Scalar/Control ISA 定义的 predicate 设施：

- [Predicate Load/Store（标量）](../scalar/predicate-load-store.md) — `pto.plds`、`pto.pld`、`pto.pldi`、`pto.psts`、`pto.pst`、`pto.psti`、`pto.pstu`

## 为什么存在本页面

Vector 指令可以消费标量 load/store 操作产生的 predicate mask。Vector ISA 重用了 Scalar/Control ISA 定义的 predicate-register 基础设施。本页面说明了 Vector ISA 与标量 predicate 模型的关系。

## Vector ISA Predicate 消费

Vector 指令消费 `!pto.mask` 操作数以实现条件 lane 执行：

```mlir
%vdst = pto.vadd %vsrc0, %vsrc1, %mask
    : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask) -> !pto.vreg<64xf32>
```

Mask 由标量 predicate load/store 操作产生，详见 [Predicate Load/Store（标量）](../scalar/predicate-load-store_zh.md)。

## Vector 特有 Predicate 操作

Vector ISA 增加了两个 materialization 操作，用于从 vector 数据生成 mask：

- [pto.vbr](./ops/predicate-and-materialization/vbr.md) — Vector broadcast：将标量值复制到所有 lane
- [pto.vdup](./ops/predicate-and-materialization/vdup.md) — Vector duplicate：将标量值复制到 vector register

## 参见

- [Predicate Load/Store（规范页面）](../scalar/predicate-load-store.md) — `plds`、`pld`、`pldi`、`psts`、`pst`、`psti`、`pstu` 的完整规范
- [Vector ISA 概述](../instruction-families/vector-families.md) — 指令集契约
