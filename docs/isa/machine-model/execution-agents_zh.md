# 执行代理与目标 Profile

PTO 使用一套架构可见的三级执行层次：host、device 和 core。这不是硬件方框图的原样复刻，而是把工作准备、派发和执行发生的位置显式写出来，同时把 target profile 的能力差异放到明确层次上。

## 执行层次

```text
HOST
  -> 准备参数、提交程序、管理运行时
DEVICE
  -> 调度合法 PTO 工作到各核心，管理 GM
CORE / AI CORE
  -> 标量单元、UB、Tile Register File、Vector Pipeline、Matrix Unit、DMA Engine
```

## Host

Host 负责：

- 准备 kernel 参数和内存描述符
- 向 device scheduler 提交 PTO 程序
- 管理 runtime 编排、stream 和事件
- 管理主机侧参数 staging 内存

Host 本身不执行 PTO 指令。

## Device

Device 是架构可见的调度层，负责：

- 把合法 PTO 工作派发给 AI Core block
- 管理设备级 GM 与主机可见内存关系
- 在需要时维持 block 之间的依赖顺序
- 管理设备侧内存分配

## Core（AI Core）

Core 是 PTO 指令真正执行的位置。它包含：

| 组件 | 说明 | PTO 可见性 |
| --- | --- | --- |
| 标量单元 | 控制流、地址计算、系统查询 | `GetBlockIdx()` 等 |
| UB | 256 KB 片上 SRAM | `!pto.ptr<T, ub>` |
| Tile Register File | Tile buffer 存储 | `!pto.tile_buf<...>` |
| 向量流水线 | 执行 `pto.v*` | `!pto.vreg<NxT>` |
| 矩阵乘单元 | 执行 `TMATMUL` / `TGEMV` | `TileType::Mat/Left/Right/Acc` |
| DMA 引擎（MTE1/2/3） | GM 与 UB 之间搬运 | `copy_*`, `TLOAD`, `TSTORE` |

## 向量寄存器架构（VLane）

在 A5 上，向量寄存器由 **8 个 VLane** 组成，每个 VLane 32 字节：

```text
vreg (256 bytes total):
| VLane0 | VLane1 | ... | VLane7 |
```

各类型在每个 VLane 中的元素数：

| 数据类型 | 每 VLane 元素数 | 每寄存器总元素数 |
| --- | :---: | :---: |
| i8 / u8 | 32 | 256 |
| i16 / u16 / f16 / bf16 | 16 | 128 |
| i32 / u32 / f32 | 8 | 64 |
| i64 / u64 | 4 | 32 |

VLane 是架构可见的：`vcgadd`、`vcgmax`、`vcgmin` 这类 group reduction 按 VLane 独立归约。

## MTE 流水细节

| MTE | 方向 | Tile 指令中的角色 | 向量指令中的角色 |
| --- | --- | --- | --- |
| `MTE1` | GM → UB | 可选预取 | 向量加载前预取 |
| `MTE2` | GM → UB | `TLOAD` 的加载阶段 | `copy_gm_to_ubuf` |
| `MTE3` | UB → GM | `TSTORE` 的写回阶段 | `copy_ubuf_to_gm` |

## 系统查询操作

| 操作 | 返回 | 说明 |
| --- | --- | --- |
| `GetBlockIdx(dim)` | `i32` | 当前 block 在维度 `dim` 上的索引 |
| `GetSubBlockIdx(dim)` | `i32` | 当前 sub-block 在父 block 内的索引 |
| `GetBlockNum(dim)` | `i32` | 维度 `dim` 上 block 总数 |
| `GetSubBlockNum(dim)` | `i32` | 父 block 内 sub-block 总数 |

除这些查询外，其他 tile/vector/scalar 操作都应视为 block-local。

## 目标 Profile

Target profile 只会缩窄 PTO ISA，不会引入新的 ISA 语义。

### CPU Simulator

- `pto.t*` 通过软件模拟
- `pto.v*` 用标量循环模拟
- matmul 使用参考实现
- 分形布局以跨步访问模拟
- UB 由堆内存分配

### A2A3 Profile

- 对应 Ascend 910B 与 Ascend 910C
- `pto.t*` 在硬件上执行
- `pto.v*` 通过 tile-vector bridge 模拟
- CUBE 提供 matmul
- 向量 tile buffer（硬件 UB）每 AI Core 256 KB
- 支持 `textract` compact 模式

### A5 Profile

- 对应 Ascend 950 PR 与 Ascend 950 DT
- `pto.t*` 和 `pto.v*` 都原生执行
- 支持 MX block-scale matmul
- 完整支持分形布局
- 向量 tile buffer（硬件 UB）每 AI Core 256 KB
- 支持 FP8、向量非对齐 store、alignment state 和 block 级通信

### Profile 对比

| 特性 | CPU | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| Tile 指令 | Emulated | Hardware | Hardware |
| Vector 指令 | Emulated | Bridge | Native |
| Matmul | Software | Hardware | Hardware |
| MX format | No | No | Yes |
| FP8 | No | No | Yes |
| `vstu` 类非对齐 store | No | No | Yes |
| Block 级集合通信 | No | Yes | Yes |

## 约束

- 架构可见的依赖顺序必须在目标调度后保留
- target profile 可以缩窄支持集合，但不能重定义合法 PTO 语义
- profile 专属特性不能写成通用 PTO 保证

## 不允许的情形

- 把 A5 专属特性写成 PTO 普遍保证
- 把 CPU 模拟器的行为或性能当成硬件 profile 契约
- 把 profile 限制写成 ISA 自相矛盾

## 相关页面

- [顺序与同步](./ordering-and-synchronization_zh.md)
- [向量指令集](../vector/README_zh.md)
- [Tile 指令集](../instruction-families/tile-families_zh.md)
- [可移植性与目标 Profile](../reference/portability-and-target-profiles_zh.md)
