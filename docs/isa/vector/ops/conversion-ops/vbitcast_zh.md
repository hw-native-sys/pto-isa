# pto.vbitcast

`pto.vbitcast` 属于 [Conversion Ops](../../conversion-ops_zh.md) 指令集。

## 摘要

`pto.vbitcast` 对 `!pto.vreg<...>` 值执行按位重新解释，不改变底层位模式。VPTO 的 `vreg` 总位宽恒为 2048 bits，本操作只改变元素类型与车道数解释。

与 [pto.vcvt](./vcvt_zh.md) 不同，`pto.vbitcast` 不做舍入、饱和或数值重定标——源寄存器的每一位都原封不动地拷贝到目标寄存器。

## 机制

本操作在向量寄存器层面是一次纯类型转换。Payload 字节不变，只改变周围 VPTO IR 对该寄存器的解释。这样可以把整型与浮点家族之间的 type punning 在 SSA 中显式化，而不必依赖任何隐含的硬件状态。

## 语法

```mlir
%result = pto.vbitcast %input : !pto.vreg<NxT0> -> !pto.vreg<MxT1>
```

## 输入

| 操作数 | 类型 | 描述 |
|---------|------|------|
| `%input` | `!pto.vreg<NxT0>` | 源向量寄存器。 |

## 预期输出

| 结果 | 类型 | 描述 |
|--------|------|------|
| `%result` | `!pto.vreg<MxT1>` | 与源位模式完全相同、被重新解释为 `MxT1` 的目标寄存器。 |

## 副作用

`pto.vbitcast` 除了产生 SSA 结果以外没有任何架构层面的副作用。它不会预留缓冲区、发出事件，也不会建立内存屏障。

## 约束

!!! warning "约束"
    - 源和目标都必须是 `!pto.vreg<...>` 类型。
    - 源与目标的总位宽必须相等（当前为 2048 bits）：`N * bitwidth(T0) = M * bitwidth(T1) = 2048`。
    - 仅支持整型和浮点元素类型。

**位宽相等的形状示例：**

- `f32<64>` → `i32<64>`（两侧都是 32 位元素，共 2048 bits）
- `f16<128>` → `i16<128>`（两侧都是 16 位元素，共 2048 bits）
- `bf16<128>` → `ui16<128>`（两侧都是 16 位元素，共 2048 bits）
- `si32<64>` → `ui32<64>`（两侧都是 32 位元素，共 2048 bits）
- `f32<64>` → `i16<128>`（32 位/16 位元素，共 2048 bits）

verifier 会拒绝总位宽不一致的形状。

## 与 `pto.vcvt` 的比较

| 维度 | `pto.vcvt` | `pto.vbitcast` |
|--------|------------|----------------|
| 位模式 | 可能改变（舍入、饱和、符号扩展） | 完全保持 |
| 车道数 | 在已记录的类型对规则下可改变 | 在总位宽 2048 不变的前提下可改变 |
| 舍入 / 饱和属性 | 支持 (`rnd`, `sat`, `part`) | 无 |
| 谓词操作数 | 必须提供 `%mask` | 不需要——bitcast 是无条件的 |

## 示例

### 将浮点重新解释为整型以做位操作

```mlir
// 准备一个浮点向量
%fvec = pto.vlds %ub[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// 重新解释为整型，准备做位操作
%ivec = pto.vbitcast %fvec : !pto.vreg<64xf32> -> !pto.vreg<64xi32>

// 取符号位（bit 31）
%sign_bits = pto.vand %ivec, %sign_mask, %mask
    : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>

// 再解释回浮点
%fvec_without_sign = pto.vbitcast %sign_bits : !pto.vreg<64xi32> -> !pto.vreg<64xf32>
```

### 有符号与无符号整型之间的 type punning

```mlir
%signed = pto.vlds %ub[%lane] : !pto.ptr<si32, ub> -> !pto.vreg<64xsi32>
%unsigned = pto.vbitcast %signed : !pto.vreg<64xsi32> -> !pto.vreg<64xui32>
// 位完全相同，解释从有符号变成无符号
```

## 相关页面

- 指令集总览：[Conversion Ops](../../conversion-ops_zh.md)
- 数值变换：[pto.vcvt](./vcvt_zh.md)
- 谓词侧 bitcast：[pto.pbitcast](./pbitcast_zh.md)
