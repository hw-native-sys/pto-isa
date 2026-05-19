# PTO 微指令：VMS4 状态查询（`pto.get_vms4_sr`）

本页说明 PTO 微指令中的 `VMS4_SR` 状态寄存器查询操作。它属于 PTO 微指令表面，对应 A5（Ascend 950）profile。

## 概览

`pto.get_vms4_sr` 把 `VMS4_SR` 硬件寄存器的内容暴露给标量代码。在一次会耗尽的 [`pto.vmrgsort4`](../../../vector/ops/sfu-and-dsa-ops/vmrgsort_zh.md) 合并排序结束后，`VMS4_SR` 记录了各源列表已执行的元素计数；读出这个寄存器可以让 kernel 知道每条输入列表消费到了哪里。

## 机制

`pto.get_vms4_sr` 是一条纯标量生产者操作。它不搬数据、不做流水线同步、不会改变任何架构状态。它只是读 `VMS4_SR` 的四个 16 位字段，并以四个 SSA `i16` 值返回。

典型用法是：发起一次可能在中途耗尽的 `pto.vmrgsort4`，然后读 `VMS4_SR` 查看各源列表的推进位置，根据这些计数驱动下一轮排序/合并。

## `pto.get_vms4_sr`

**语法**：`%list0, %list1, %list2, %list3 = pto.get_vms4_sr : i16, i16, i16, i16`

**语义**：读取 `VMS4_SR`，返回源列表 0、1、2、3 的已完成元素计数。

### 输入

无。

### 预期输出

| 结果 | 类型 | 描述 |
|--------|------|------|
| `%list0` | `i16` | 源列表 0 的已完成计数 |
| `%list1` | `i16` | 源列表 1 的已完成计数 |
| `%list2` | `i16` | 源列表 2 的已完成计数 |
| `%list3` | `i16` | 源列表 3 的已完成计数 |

### 寄存器位段

| 位 | 含义 |
|------|------|
| `[15:0]` | 源列表 0 的已完成计数 |
| `[31:16]` | 源列表 1 的已完成计数 |
| `[47:32]` | 源列表 2 的已完成计数 |
| `[63:48]` | 源列表 3 的已完成计数 |

```c
status = VMS4_SR;
list0 = (uint16_t)(status & 0xffff);
list1 = (uint16_t)((status >> 16) & 0xffff);
list2 = (uint16_t)((status >> 32) & 0xffff);
list3 = (uint16_t)((status >> 48) & 0xffff);
```

### 约束

- 返回值是各源列表已消费元素数量的无符号 16 位计数。
- 典型用法是在一次会耗尽的 `pto.vmrgsort4` 之后读 `VMS4_SR`，以了解部分进度。
- 本操作是纯标量生产者，没有任何架构副作用。

### 示例

```mlir
// 在一次部分完成的 pto.vmrgsort4 之后读出各列表已执行计数
%list0, %list1, %list2, %list3 = pto.get_vms4_sr : i16, i16, i16, i16

// 用这些计数驱动下一轮排序
%c0_i64 = arith.extui %list0 : i16 to i64
// ... 喂给下一个 vmrgsort4 的初始化
```

## 相关操作

- 4 路合并排序：[`pto.vmrgsort`](../../../vector/ops/sfu-and-dsa-ops/vmrgsort_zh.md)
- Block 运行时查询：[BlockDim 查询操作](./block-dim-query_zh.md)
