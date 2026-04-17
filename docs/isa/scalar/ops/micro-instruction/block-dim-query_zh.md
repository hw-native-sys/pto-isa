# PTO 微指令：BlockDim 与运行时查询

本页记录 PTO 微指令层的运行时查询操作。这些操作向标量代码暴露 block 级执行坐标，当前主要对应 A5（Ascend 950）profile。

## 概览

这些操作把当前 kernel 实例所处的 block / subblock 坐标暴露给标量代码。可以把它们看成 PTO 层面对应的 `GetBlockIdx()`、`GetBlockNum()` 一类查询。

当同一份 kernel body 会在多个 block 或 subblock 上重复执行，而每个执行实例又必须自己算出全局工作区间时，就需要这类查询。

## 机制

这组查询操作都是纯标量生产者。它们不做数据搬运、不分配内存，也不会直接创建 tiling 或 double buffering。它们只是把 launch 时确定的执行坐标提供给后续的标量算术、指针形成和控制流。

## BlockDim 查询操作

### 常见使用模式

典型写法是：

- 把整个输入 / 输出张量切分成 `block_num` 个互不重叠的 block 区间
- 每个 block 用 `block_idx` 算出自己的起始偏移
- 在 block 内部，再继续用普通标量 `arith` / `scf` 做更细的 tiling

例如，一个张量被均匀分成 8 个 block，每个 block 处理 `block_length = 2048` 个元素，那么第 `b` 个 block 对应的全局区间就是 `[b * block_length, (b + 1) * block_length)`。对应的 GM 基址可以通过 `block_idx * block_length` 这个元素偏移量算出来。

### 执行模型

在 PTO 微指令层，这些查询操作只是标量结果值。它们不做 DMA，不申请 tile buffer，也不会自动建立同步关系。它们的用途是为外层地址计算与结构化控制流提供输入。

## `pto.get_block_idx`

**语法**：`%block = pto.get_block_idx`

**结果类型**：`i64`

**语义**：返回当前 block 的编号，取值范围为 `[0, pto.get_block_num())`。

```c
block = block_idx();
```

### 输入

无。

### 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%block` | `i64` | 当前 block 编号，范围为 `[0, block_num)` |

### 约束

- 返回值必须位于 `[0, get_block_num())`。
- 这类查询只在定义了 block 维度的 kernel 启动上下文中有意义。

### 示例

```mlir
%block = pto.get_block_idx
```

## `pto.get_subblock_idx`

**语法**：`%subblock = pto.get_subblock_idx`

**结果类型**：`i64`

**语义**：返回当前 subblock 的编号，取值范围为 `[0, pto.get_subblock_num())`。

```c
subblock = subblock_idx();
```

### 输入

无。

### 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%subblock` | `i64` | 当前 subblock 编号，范围为 `[0, subblock_num)` |

### 约束

- 返回值必须位于 `[0, get_subblock_num())`。

### 示例

```mlir
%subblock = pto.get_subblock_idx
```

## `pto.get_block_num`

**语法**：`%block_num = pto.get_block_num`

**结果类型**：`i64`

**语义**：返回当前 kernel 启动时可见的 block 总数。

```c
block_num = block_num();
```

### 输入

无。

### 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%block_num` | `i64` | 当前启动可见的 block 总数 |

### 约束

- 返回值必须是正整数。

### 示例

```mlir
%block_num = pto.get_block_num
```

## `pto.get_subblock_num`

**语法**：`%subblock_num = pto.get_subblock_num`

**结果类型**：`i64`

**语义**：返回当前执行实例可见的 subblock 总数。

```c
subblock_num = subblock_num();
```

### 输入

无。

### 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%subblock_num` | `i64` | 当前 block 内可见的 subblock 总数 |

### 约束

- 返回值必须是正整数。

## 典型用法：按 block 切分数据

```mlir
%block = pto.get_block_idx
%block_num = pto.get_block_num

%block_len = arith.constant 2048 : index
%block_len_i64 = arith.index_cast %block_len : index to i64

%base = arith.index_cast %block : i64 to index
%offset = arith.muli %base, %block_len : index

%block_in = pto.addptr %gm_in, %offset : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
%block_out = pto.addptr %gm_out, %offset : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
```

在这种写法里，所有 block 执行的是同一份 kernel body，但每个 block 会看到不同的 `%block`，因此算出不同的 GM 区间。

### Grid 设计考虑

| Grid 维度 | 典型用途 |
|-----------|----------|
| `block_num` | 在多个互不重叠的数据区间之间做并行划分 |
| `subblock_num` | 在单个 block 内部做更细的层次化切分 |

## 相关页面

- [指针操作](./pointer-operations_zh.md)
- [共享标量算术](../../shared-arith_zh.md)
