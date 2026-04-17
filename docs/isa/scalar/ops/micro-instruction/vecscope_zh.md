# PTO 微指令：向量执行作用域（`pto.vecscope` / `pto.strict_vecscope`）

本页记录 PTO 微指令层的向量执行作用域操作。它们定义了标量单元与向量线程之间的硬件接口，当前主要对应 A5（Ascend 950）profile。

## 概览

`__VEC_SCOPE__` 是 Vector Function（VF）启动的 IR 层表示。在 PTO 架构里，它描述的是标量单元与向量线程之间的执行边界。

在 PTO 微指令源 IR 中，向量执行作用域由专门的 region op 表达：

- 默认形式：`pto.vecscope`
- 严格形式：`pto.strict_vecscope`

当作用域内部不能接受隐式捕获、必须显式声明所有外部输入时，使用 `pto.strict_vecscope`。

## 机制

`pto.vecscope` 与 `pto.strict_vecscope` 本身并不直接执行 payload 运算；它们定义的是一个向量区间（vector interval）的生命周期边界。所有会生成或消费 `!pto.vreg`、`!pto.mask<...>`、`!pto.align` 的操作，都必须恰好位于一个这样的区间内部。严格形式进一步把 region 接口显式化：所有外部值都必须通过 operand 传入，并在 block argument 中接收。

## 输入

这类作用域操作的主要输入是 region body。`pto.strict_vecscope` 还会额外接收一组显式 operand，它们会成为 body block argument。

## 预期输出

这些作用域操作定义向量执行边界，并约束向量可见状态在何处合法。当前文档中的示例不会让它们直接返回 payload 值；真正的 payload 结果由作用域内部的向量操作产生。

## 执行模型

PTO 微指令对应的是 Ascend 950 的 **Decoupled Access-Execute（DAE）** 架构。其执行模型遵循 **非阻塞 fork** 语义：

- **标量侧启动**：标量处理器发起一个 VF 调用。启动命令发出后，标量单元不会因为这次调用而停住，而是继续执行后续标量逻辑。
- **向量侧执行**：被启动的向量线程独立抓取并执行作用域中的指令。
- **并行性**：标量侧可以继续准备地址和控制流，而向量侧同时进行 SIMD 计算。

### 启动机制与约束

- **参数缓冲**：VF 所需的参数必须先被放入硬件约定的参数缓冲区。
- **启动开销**：启动 VF 本身会消耗若干周期，因此很小的 VF 也要考虑 launch overhead。

## `pto.vecscope`：默认向量作用域

### 语法

```mlir
pto.vecscope {
  // region body
}
```

### 语义

`pto.vecscope` 允许 body 直接使用外层 SSA 值，也就是允许隐式捕获。所有生成或消费 `!pto.vreg`、`!pto.mask<...>`、`!pto.align` 的操作，都必须恰好被一个向量区间包围。

### 约束

- 向量区间不能嵌套。普通的 `scf.for` 可以嵌套，但一个 vector interval 不能再包一个 vector interval。
- 不管源形式是 `pto.vecscope`、`pto.strict_vecscope`，还是 lowered 之后带 `llvm.loop.aivector_scope` 的 loop carrier，只要操作生成或消费 `!pto.vreg`、`!pto.mask<...>`、`!pto.align`，就必须恰好属于一个向量区间。

### 示例

```mlir
pto.set_loop2_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop1_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.copy_gm_to_ubuf %7, %2, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c0_i64,
    %false, %c0_i64, %c128_i64, %c128_i64
    : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64, i64, i1, i64, i64, i64

pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

pto.vecscope {
  scf.for %lane = %c0 to %9 step %c64 {
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
    %v = pto.vlds %2[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %abs, %8[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
  }
}

pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop2_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.copy_ubuf_to_gm %8, %14, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c128_i64, %c128_i64
    : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64, i64, i64
```

## `pto.strict_vecscope`：严格向量作用域

### 语法

```mlir
pto.strict_vecscope(%arg1, %arg2, ...) {
^bb0(%in1: <type>, %in2: <type>, ...):
  // region body
}
: (<type1>, <type2>, ...) -> ()
```

### 语义

`pto.strict_vecscope` 要求 body 使用到的所有外部值都必须通过 op operand 显式传入，并在 block argument 中接收。它拒绝隐式捕获。

### 约束

- `pto.strict_vecscope` 不允许从外层隐式捕获 SSA 值。
- `pto.vecscope` 与 `pto.strict_vecscope` 都表示一个显式的 VPTO 向量区间。
- 这个作用域 op 本身只定义区间边界与 region 形参契约。

### 示例

```mlir
pto.strict_vecscope(%ub_in, %ub_out, %lane, %remaining) {
^bb0(%in: !pto.ptr<f32, ub>, %out: !pto.ptr<f32, ub>, %iv: index, %rem: i32):
  %mask, %next_remaining = pto.plt_b32 %rem : i32 -> !pto.mask<b32>, i32
  %v = pto.vlds %in[%iv] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %out[%iv], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
} : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, index, i32) -> ()
```

当你希望向量作用域的所有输入都显式出现在 region 签名里，而不是依赖外层 SSA 可见性时，应使用 `pto.strict_vecscope`。

## `pto.vecscope` 与 `pto.strict_vecscope` 的区别

| 维度 | `pto.vecscope` | `pto.strict_vecscope` |
|------|----------------|----------------------|
| 隐式捕获 | 允许 | 拒绝 |
| Region 参数 | 依赖外层 SSA | 必须显式声明在 operand / block argument 中 |
| 适用场景 | 简单 kernel、快速编写 | 形式化验证、IR 重写 |
| SSA 可见性 | body 可直接使用外层值 | 所有输入都必须显式传入 |

## 与硬件流水线的关系

在向量作用域内部，DAE 架构要求显式协调以下流水线：

- **MTE2**（`PIPE_MTE2`）：GM → 向量 tile buffer 的 DMA copy-in
- **PIPE_V**：向量 ALU 运算
- **MTE3**（`PIPE_MTE3`）：向量 tile buffer → GM 的 DMA copy-out

同步可以通过两类机制完成：

- `pto.set_flag` / `pto.wait_flag`
- `pto.get_buf` / `pto.rls_buf`

## 相关页面

- [流水线同步](../../pipeline-sync_zh.md)
- [共享标量算术](../../shared-arith_zh.md)
- [共享 SCF](../../shared-scf_zh.md)
- [BlockDim 与运行时查询](./block-dim-query_zh.md)
