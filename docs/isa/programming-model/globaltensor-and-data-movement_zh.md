# GlobalTensor 与数据搬运

PTO 不隐藏全局内存与本地执行状态之间的数据移动。`GlobalTensor` 是面向 GM 的架构可见对象，数据移动操作定义了数据何时进入或离开局部有效载荷指令集。下文给出 GM 侧类型以及通向 tile buffer 与向量寄存器的完整路径。

## GlobalTensor

### 模板签名

```text
GlobalTensor<DType, Shape, Stride, Layout>
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `DType` | C++ type | 元素类型 |
| `Shape` | `Shape<ND>()` | N 维 shape |
| `Stride` | `Stride<ND>()` | 各维 stride |
| `Layout` | enum | `ND`、`DN`、`NZ` |

`GlobalTensor` 是 `__gm__` 内存的带元数据视图，而不是存储本体。

### GlobalTensor 与 PartitionTensorView

| 类型 | 说明 | 用法 |
| --- | --- | --- |
| `GlobalTensor` | C++ API 里的 GM 视图 | C++ kernel |
| `!pto.partition_tensor_view<...>` | SSA/IR 里的 GM 分区描述 | PTO-AS / MLIR |
| `!pto.memref<...>` | 更低层的标准 memref 形式 | Lowered 形式 |

`partition_tensor_view` 通常以 5D 形式描述 `(B, H, W, R, C)`。

### 支持的 Layout

| Layout | Stride Pattern | 说明 |
| --- | --- | --- |
| `ND` | row-major | 默认形式 |
| `DN` | column-major | 列优先形式 |
| `NZ` | 分形 row-major stride | 用于分形 tile 兼容 |

## Tile 指令的数据路径

Tile 指令通过 MTE2/MTE3 在 GM 与 tile buffer 之间搬运数据：

```text
GM -> UB -> Tile Buffer -> Tile Compute -> Tile Buffer -> UB -> GM
```

### TLOAD

`TLOAD` 把 `GlobalTensor` 中的数据搬进 tile buffer：

```text
dst[i, j] = src[r0 + i, c0 + j]
```

其中 `r0`、`c0` 由 `GlobalTensor` 的 shape/stride 与 tile 的 valid region 推导。

**传输大小**：

```text
dst.GetValidRow() × dst.GetValidCol()
```

**约束**：

- 源 dtype 大小必须等于目标 dtype 大小
- 布局必须兼容

### TSTORE

`TSTORE` 把 tile buffer 中的数据写回 `GlobalTensor`：

```text
dst[r0 + i, c0 + j] = src[i, j]
```

其中 `i ∈ [0, src.GetValidRow())`，`j ∈ [0, src.GetValidCol())`。

### 原子 Store 变体

| AtomicType | 行为 |
| --- | --- |
| `AtomicNone` | 普通覆盖写入 |
| `AtomicAdd` | 原子加 |
| `AtomicMax` | 原子最大值 |
| `AtomicMin` | 原子最小值 |

## 向量指令的数据路径

向量指令先经过显式的 GM↔UB DMA，再做 UB↔向量寄存器的搬运：

```text
GM -> copy_gm_to_ubuf -> UB -> vlds/vgather2 -> vreg -> vector compute
vreg -> vsts/vscatter -> UB -> copy_ubuf_to_gm -> GM
```

### DMA 拷贝操作

| 操作 | 方向 | 说明 |
| --- | --- | --- |
| `copy_gm_to_ubuf` | GM → UB | 从 GM 复制到 UB 暂存区 |
| `copy_ubuf_to_gm` | UB → GM | 从 UB 回写到 GM |
| `copy_ubuf_to_ubuf` | UB → UB | UB 内复制，常见于双缓冲 |

这些操作本身不隐式同步。向量计算真正使用数据前，需要显式 `set_flag` / `wait_flag` 或等价顺序边。

### Vector Load/Store

| 操作 | 路径 | 说明 |
| --- | --- | --- |
| `vlds` | UB → vreg | 标准向量加载 |
| `vsld` | vreg → UB | 标准向量存储 |
| `vgather2` | UB → vreg | gather / stride 加载 |
| `vscatter` | vreg → UB | scatter 存储 |

#### 常见分布模式

| Mode | 含义 |
| --- | --- |
| `NORM` | 连续加载 |
| `BRC_B8/B16/B32` | 广播 |
| `US_B8/B16` | 上采样 |
| `DS_B8/B16` | 下采样 |
| `UNPK_B8/B16/B32` | unpack / 零扩展 |
| `DINTLV_B32` | 去交错 |
| `SPLT2CHN_B8/B16` | 双通道拆分 |
| `SPLT4CHN_B8` | 四通道拆分 |

## MTE 流水

| MTE | 方向 | Tile 指令中的角色 | Vector 指令中的角色 |
| --- | --- | --- | --- |
| `MTE1` | GM → UB | 可选预取 | 向量加载前预取 |
| `MTE2` | GM → UB | `TLOAD` 的加载阶段 | `copy_gm_to_ubuf` |
| `MTE3` | UB → GM | `TSTORE` 的写回阶段 | `copy_ubuf_to_gm` |

## 约束

- 数据移动的合法性依赖源/目标指令集、布局和 target profile
- 数据移动不会抹掉 valid-region 语义
- 向量路径的 buffer/register 规则与 tile 路径不同，不能混用
- DMA 完成前，不得假设向量计算已经看见数据

## 不允许的情形

- 把需要显式移动的数据路径写成“自动可见”
- 把 vector 路径和 tile 路径写成同一套合法性契约
- 把 target-specific 快捷路径写成架构保证
- 在 `copy_gm_to_ubuf` 完成前直接 `vlds`

## 示例

### Tile 指令：逐元素加法

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void vec_add(Tile<float, 16, 16>& c, const GlobalTensor<float>& ga,
             const GlobalTensor<float>& gb) {
    Tile<float, 16, 16> a, b;
    TLOAD(a, ga);
    TLOAD(b, gb);
    TADD(c, a, b);
    TSTORE(gc, c);
}
```

### 向量指令：显式 UB 暂存

```c
copy_gm_to_ubuf(%ub_ptr, %gm_ptr, ...);
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
%vreg = pto.vlds %ub[%offset] {dist = "NORM"};
%result = pto.vadd %vreg, %vreg;
pto.vsts %result, %ub_out[%offset];
copy_ubuf_to_gm(%ub_out, %gm_out, ...);
```

## 相关页面

- [Tile 与有效区域](./tiles-and-valid-regions_zh.md)
- [向量指令集](../vector/README_zh.md)
- [向量加载存储参考](../vector/vector-load-store_zh.md)
- [DMA 拷贝参考](../scalar/dma-copy_zh.md)
