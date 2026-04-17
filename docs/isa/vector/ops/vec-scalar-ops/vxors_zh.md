# pto.vxors

`pto.vxors` 属于[向量-标量操作](../../vec-scalar-ops_zh.md)指令集。

## 概述

`pto.vxors` 对一个向量寄存器和一个广播标量做逐 lane 按位异或。

## 机制

对每个活跃 lane `i`，执行 `dst[i] = src[i] ^ scalar`。标量位掩码会广播到每个活跃 lane。非活跃 lane 不参与运算。

## 语法

### PTO 汇编形式

```text
vxors %dst, %src, %scalar, %mask : !pto.vreg<NxT>, T
```

### AS Level 1（SSA）

```mlir
%result = pto.vxors %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>
```

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%input` | `!pto.vreg<NxT>` | 源向量寄存器 |
| `%scalar` | `T` | 广播到每个活跃 lane 的标量位掩码 |
| `%mask` | `!pto.mask` | 谓词掩码；只有掩码位为 1 的 lane 参与运算 |

## 预期输出

| 结果 | 类型 | 说明 |
|------|------|------|
| `%result` | `!pto.vreg<NxT>` | 活跃 lane 上得到逐 lane 按位异或结果 |

## 副作用

这条指令除了产生目标值，没有其他架构副作用。

## 约束

- 位运算形式只对整数元素类型定义。
- `%input` 与 `%result` 必须具有相同的向量宽度 `N` 和相同的元素类型 `T`。
- 掩码宽度必须与 `N` 一致。

## 异常与非法情形

- verifier 会拒绝非法的操作数形状、不支持的元素类型以及不合法的属性组合。
- 约束部分列出的额外非法情形，同样属于 `pto.vxors` 的契约。

## 目标 Profile 限制

- 只支持整数元素类型。
- A5 是当前手册里最细的具体 profile；CPU 模拟器和 A2/A3 类目标可以在保留可见 PTO 契约的前提下做等效模拟。

## 示例

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = src[i] ^ scalar;
```

```mlir
%result = pto.vxors %values, %bitmask, %mask : !pto.vreg<64xi32>, i32, !pto.mask -> !pto.vreg<64xi32>
```

## 相关页面

- 指令集总览：[向量-标量操作](../../vec-scalar-ops_zh.md)
- 上一条指令：[pto.vors](./vors_zh.md)
- 下一条指令：[pto.vshls](./vshls_zh.md)
