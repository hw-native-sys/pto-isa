# 向量指令集：加载与存储

PTO指令集架构中的向量加载 / 存储定义在这里。它们负责把数据在向量 tile buffer 与向量寄存器之间来回搬运，同时保持这条路径完全停留在 `pto.v*` 语义之内，不退回 tile 指令面。

> **类别：** 向量 tile buffer ↔ 向量寄存器 数据搬运
> **流水线：** `PIPE_V`

向量加载从向量 tile buffer（当前硬件由 UB 实现）读取数据进入 `vreg`；向量存储则把 `vreg` 写回这个向量 tile buffer。所有向量计算都发生在 `vreg` 上，因此这组指令正好位于 DMA 与向量计算之间。

---

## 执行模型：DMA 与向量流水线的衔接

向量 load/store 连接的是两套不同的执行域：

- DMA 域：MTE2 / MTE3
- 计算域：`PIPE_V`

一段完整的向量 kernel 通常长这样：

```text
GM → MTE2 → 向量 tile buffer → VLDS → vreg → PTO.v* 计算 → VSTS → 向量 tile buffer → MTE3 → GM
```

下面这个 skeleton 把一条完整数据链条串起来了：

```mlir
module attributes {pto.target_arch = "a5"} {
  func.func @kernel_2d(%arg0: !pto.ptr, %arg1: !pto.ptr) {
    %false = arith.constant false

    // 阶段 1：MTE2 从 GM 装载到向量 tile buffer
    pto.get_buf "PIPE_MTE2", 0, 0
    pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
    pto.copy_gm_to_ubuf %arg0, %ub_in, %c0_i64, %c32_i64, %c128_i64,
      %c0_i64, %c0_i64, %false, %c0_i64, %c128_i64, %c128_i64
      : !pto.ptr, !pto.ptr, i64, i64, i64, i64, i64, i1, i64, i64, i64
    pto.rls_buf "PIPE_MTE2", 0, 0

    // 阶段 2：MTE2 → V，同步数据可见性
    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

    // 阶段 3：向量计算
    pto.get_buf "PIPE_V", 0, 0
    pto.vecscope {
      %_:1 = scf.for %offset = %c0 to %c1024 step %c64
          iter_args(%remaining = %c1024_i32) -> (i32) {
        %mask, %next = pto.plt_b32 %remaining : i32 -> !pto.mask, i32
        %vec = pto.vlds %ub_in[%offset] : !pto.ptr -> !pto.vreg<64xf32>
        %out = pto.vabs %vec, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
        pto.vsts %out, %ub_out[%offset], %mask : !pto.vreg<64xf32>, !pto.ptr, !pto.mask
        scf.yield %next : i32
      }
    }
    pto.rls_buf "PIPE_V", 0, 0

    // 阶段 4：V → MTE3，同步结果可见性
    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

    // 阶段 5：MTE3 从向量 tile buffer 回写到 GM
    pto.get_buf "PIPE_MTE3", 0, 0
    pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
    pto.copy_ubuf_to_gm %ub_out, %arg1, %c0_i64, %c32_i64, %c128_i64,
      %c0_i64, %c128_i64, %c128_i64
      : !pto.ptr, !pto.ptr, i64, i64, i64, i64, i64, i64
    pto.rls_buf "PIPE_MTE3", 0, 0

    pto.barrier #pto.pipe
    return
  }
}
```

### 典型同步原语

| 操作 | 作用 |
|------|------|
| `pto.set_flag["PIPE_MTE2", "PIPE_V", id]` | 告诉 `PIPE_V`：MTE2 已经把输入写好 |
| `pto.wait_flag["PIPE_MTE2", "PIPE_V", id]` | `PIPE_V` 在读之前必须等待 |
| `pto.set_flag["PIPE_V", "PIPE_MTE3", id]` | 告诉 MTE3：向量结果已经写回向量 tile buffer |
| `pto.wait_flag["PIPE_V", "PIPE_MTE3", id]` | MTE3 在回写前必须等待 |
| `pto.get_buf "PIPE_*", slot, 0` | 获取某个 buffer 槽位 |
| `pto.rls_buf "PIPE_*", slot, 0` | 释放某个 buffer 槽位 |

`get_buf` / `rls_buf` 的意义是用 buffer 槽位自动承载 RAW/WAR 依赖，避免在复杂双缓冲循环里手写大量事件 ID。

### A2A3 带宽模型

| 传输路径 | 带宽 | 常量 | 周期公式 |
|----------|------|------|----------|
| GM → 向量 tile buffer | 128 B/周期 | `A2A3_BW_GM_VEC` | `ceil(Nbytes / 128)` |
| GM → 矩阵 tile buffer | 256 B/周期 | `A2A3_BW_GM_MAT` | `ceil(Nbytes / 256)` |
| 向量 tile buffer → 向量 tile buffer | 128 B/周期 | `A2A3_BW_VEC_VEC` | `ceil(Nbytes / 128)` |

例如一次 4096B 的向量 tile 装载：

```text
cycles = ceil(4096 / 128) = 32
```

---

## 通用操作数模型

- `%source` / `%dest`：SSA 形式的基址操作数，必须指向向量 tile buffer
- `%offset`：位移操作数，具体编码随指令形式而变化
- `%mask`：谓词操作数，用在 predicated memory 指令中。inactive lane 或 inactive block 不得发起未文档化的内存请求
- `%result`：目标向量寄存器
- `!pto.align`：承载非对齐 load/store 对齐状态的 SSA 值

PTO 把非对齐状态显式放在 SSA 里，而不是让后端偷偷维护隐藏状态。这一点对合法化和调试都很关键。

---

## 连续加载

### `pto.vlds`

- **语法：** `%result = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>`
- **语义：** 用给定分布模式，从向量 tile buffer 把一段数据装成向量寄存器。

约束：

- 有效地址必须满足所选 `DIST` 的对齐要求
- `NORM` 读一个完整向量 footprint
- 广播、上采样、下采样、拆包、拆通道、去交织等模式只改变“如何映射到目标 lane”，不改变“数据来自向量 tile buffer”这个事实

| 模式 | 说明 |
|------|------|
| `NORM` | 连续 256B 装载 |
| `BRC_B8/B16/B32` | 广播单元素到所有 lane |
| `US_B8/B16` | 上采样，重复每个元素 |
| `DS_B8/B16` | 下采样，隔一个取一个 |
| `UNPK_B8/B16/B32` | 拆包并扩展到更宽类型 |
| `SPLT4CHN_B8` | 4 通道拆分 |
| `SPLT2CHN_B8/B16` | 2 通道拆分 |
| `DINTLV_B32` | 32 位去交织 |
| `BLK` | 按块装载 |

```mlir
%v = pto.vlds %ub[%offset] {dist = "NORM"}
    : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

%bcast = pto.vlds %ub[%c0] {dist = "BRC_B32"}
    : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

### `pto.vldas`

- **语法：** `%result = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align`
- **语义：** 为后续非对齐装载预热对齐状态。

它会围绕 `%source` 所在的对齐块初始化 `!pto.align`，供后续 `vldus` 串流使用。

### `pto.vldus`

- **语法：** `%result, %align_out, %base_out = pto.vldus %source, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align, !pto.ptr<T, ub>`
- **语义：** 依赖已预热的 `!pto.align`，执行非对齐装载。

约束：

- 第一条相关 `vldus` 前必须有匹配的 `vldas`
- `!pto.align` 和基址都会随流推进，以 SSA 结果的形式显式返回

```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align2, %ub2 = pto.vldus %ub, %align
    : !pto.ptr<f32, ub>, !pto.align
      -> !pto.vreg<64xf32>, !pto.align, !pto.ptr<f32, ub>
```

---

## 双路加载与去交织

### `pto.vldx2`

- **语法：** `%low, %high = pto.vldx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **语义：** 从交织布局中一次装出两路结果，常用于 AoS→SoA 转换。

合法分布模式包括 `DINTLV_B8`、`DINTLV_B16`、`DINTLV_B32`、`BDINTLV`。

```c
for (int i = 0; i < 64; i++) {
    low[i]  = UB[base + 8*i];
    high[i] = UB[base + 8*i + 4];
}
```

这两路输出是有顺序的，后端不得把它们交换。

---

## 固定步长加载

### `pto.vsld`

- **语法：** `%result = pto.vsld %source[%offset], "STRIDE" : !pto.ptr<T, ub> -> !pto.vreg<NxT>`
- **语义：** 用固定 stride 模式装载。

真正决定“从哪里读”的是 stride token，而不只是 lane 编号。

### `pto.vsldb`

- **语法：** `%result = pto.vsldb %source, %offset, %mask : !pto.ptr<T, ub>, i32, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 按 block stride 从向量 tile buffer 装载，常用于二维 tile 访问。

这里的 `%offset` 不是普通字节偏移，而是一个编码了 stride / repeat 规律的控制字。被 mask 关闭的 block 会在结果中清零，也不应为该 block 触发地址越界异常。

---

## Gather 装载

### `pto.vgather2`

- **语法：** `%result = pto.vgather2 %source, %offsets, %active_lanes : !pto.ptr<T, ub>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- **语义：** 按索引从向量 tile buffer gather。

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i] * sizeof(T)];
```

只允许前 `%active_lanes` 个索引参与。

### `pto.vgatherb`

- **语法：** `%result = pto.vgatherb %source, %offsets, %active_lanes : !pto.ptr<T, ub>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- **语义：** 按 32B block 做 gather，而不是按单字节逐 lane gather。

约束：

- `%source` 必须 32B 对齐
- 每个参与的 offset 都必须描述 32B 对齐块
- inactive block 在结果中清零

### `pto.vgather2_bc`

- **语法：** `%result = pto.vgather2_bc %source, %offsets, %mask : !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask -> !pto.vreg<NxT>`
- **语义：** 带广播语义的 gather，受谓词控制。

被 mask 关闭的 lane 不参与地址合并，也不会触发地址异常，结果位置按零填充。

---

## 连续存储

### `pto.vsts`

- **语法：** `pto.vsts %value, %dest[%offset], %mask {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.mask`
- **语义：** 按给定分布模式，把向量寄存器写回向量 tile buffer。

约束：

- 目标地址必须满足分布模式的对齐要求
- 打包 / 收窄模式可能只保留源向量的一部分比特
- merge-channel 模式会把多个 plane 重新交织成通道数据

| 模式 | 说明 |
|------|------|
| `NORM_B8/B16/B32` | 连续存储 |
| `PK_B16/B32` | 打包 / 收窄存储 |
| `MRG4CHN_B8` | 四通道合并 |
| `MRG2CHN_B8/B16` | 双通道合并 |

### `pto.vstx2`

- **语法：** `pto.vstx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask`
- **语义：** 双路交织存储，常用于 SoA→AoS 转换。

合法分布模式包括 `INTLV_B8`、`INTLV_B16`、`INTLV_B32`。

```c
for (int i = 0; i < 64; i++) {
    UB[base + 8*i]     = low[i];
    UB[base + 8*i + 4] = high[i];
}
```

---

## 固定步长存储

### `pto.vsst`

- **语法：** `pto.vsst %value, %dest[%offset], "STRIDE" : !pto.vreg<NxT>, !pto.ptr<T, ub>`
- **语义：** 用固定 stride 模式存储。

真正定义写入模式的是 stride token。

### `pto.vsstb`

- **语法：** `pto.vsstb %value, %dest, %offset, %mask : !pto.vreg<NxT>, !pto.ptr<T, ub>, i32, !pto.mask`
- **语义：** 按 block stride 存储二维 tile。

这里的 `%offset` 同样是控制字，而不是普通的字节偏移。

---

## Scatter 存储

### `pto.vscatter`

- **语法：** `pto.vscatter %value, %dest, %offsets, %active_lanes : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vreg<NxI>, index`
- **语义：** 按索引把向量值散写到向量 tile buffer。

```c
for (int i = 0; i < active_lanes; i++)
    UB[base + offsets[i] * sizeof(T)] = src[i];
```

约束：

- 当前记录的元素大小只支持 `b8`、`b16`、`b32`
- 索引向量必须使用这套指令支持的整数元素类型与布局
- 如果多个索引别名到同一个地址，只保证会有一次写入生效，获胜 lane 由实现定义

---

## 非对齐存储状态与 flush

### `pto.vsta`

- **语法：** `pto.vsta %value, %dest[%offset] : !pto.align, !pto.ptr<T, ub>, index`
- **语义：** 把待提交的存储对齐状态 flush 到内存。

### `pto.vstu`

`pto.vstu` 有两种语法形态。

**形态 A：**

- **语法：** `%align_out, %base_out = pto.vstu %align_in, %base_in, %value, %dest, %mode : !pto.align, !pto.ptr<T, ub>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index -> !pto.align, !pto.ptr<T, ub>`
- **语义：** 显式串接对齐状态和基址状态的非对齐存储。

**形态 B：**

- **语法：** `%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE" : !pto.align, index, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, index`
- **语义：** 显式更新对齐状态与逻辑偏移。

不论是哪种形态，都只是推进状态，不代表尾部残留字节已经全部落地。之后还需要一个兼容的 flush 指令。

### `pto.vstus`

`pto.vstus` 同样有两种语法形态：

- 标量偏移 + 显式状态串接
- 标量偏移 + 明确 `MODE` 的状态更新

它与 `vstu` 的区别主要在于偏移来源是标量寄存器 / 立即量，而不是 index 形式。

### `pto.vstur`

`pto.vstur` 也有两种形态：

- 只返回更新后的 `!pto.align`
- 带 `MODE` 的显式状态形式

它们都只是“把一段非对齐存储推进一步”，而不是自动完成整个尾部提交。

### `pto.vstas`

- **语法：** `pto.vstas %align, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32`
- **语义：** 用标量偏移把先前累计的尾部状态 flush 到向量 tile buffer。

### `pto.vstar`

- **语法：** `pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>`
- **语义：** 结束一段非对齐存储序列，把剩余缓冲状态全部冲刷到内存。

这三类 flush 指令的共同要求是：前面必须已经有兼容的状态产生序列，否则 `!pto.align` 的含义根本不成立。

---

## 相关页面

- [向量 DMA 拷贝](./dma-copy_zh.md)
- [向量流水线同步](./pipeline-sync_zh.md)
- [谓词与物化](./predicate-and-materialization_zh.md)
