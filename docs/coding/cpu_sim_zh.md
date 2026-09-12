# CPU_SIM

CPU_SIM 是一个面向纯 CPU 系统执行的后端实现。

与 NPU 后端相比，CPU_SIM 当前存在以下差异和限制：

- 每个 CPU 工作线程内的 PTO 指令同步执行。对于已支持的同步和通信操作，CPU_SIM 会使用 CPU 同步原语进行模拟，其中包括 TileData `TPUSH`/`TPOP`/`TFREE` FIFO 流程。
- 使用特定的内存模型来模拟 NPU 内存层次（见下文）。
- 多线程支持尚不完整。`Tile` 对象的内存访问不具备线程间同步能力，因此除已支持的通信操作外，不建议跨线程共享 Tile。Tile 的惰性内存分配同样不具备线程间同步能力。

## 启用 CPU_SIM

可通过设置编译宏 `__CPU_SIM` 启用 CPU 后端（CPU_SIM）。启用后，可使用标准面向 CPU 的编译器（如 gcc 或 clang）构建程序。

为兼容原本面向 NPU 的程序，仓库在 `include/pto/common/cpu_stub.hpp` 中为 CPU 平台提供了一些 Ascend 相关函数的替代实现。对于已经使用 NPU 后端的已有程序，包含该头文件后通常只需做少量修改即可在 CPU 上编译。

如果不包含该头文件，则需要自行移除或替换诸如 `aclInit`、`aclrtSetDevice` 等函数调用。

CPU stub 中的 `aclInit`、`aclrtSetDevice` 和 `aclrtCreateStream` 分别用于初始化 CPU_SIM 运行时、保存非负 device ID，以及创建轻量级主机 stream handle。运行时环境变量由 `aclInit` 或首次触发惰性运行时初始化的 API 读取，以先发生者为准。

## CPU_SIM 内存模型

通常情况下，CPU_SIM 中所有 Tile 的内存都分配在系统内存中。这与 NPU 后端不同：在 NPU 后端中，内存会划分为 host memory、device memory，以及设备内部不同的片上存储位置。

为了让 CPU_SIM 的行为更接近 NPU，CPU_SIM 会模拟若干与 NPU 架构对应的独立内存位置。

CPU_SIM 会为每个线程分配以下内存区域：

- `UB`
- `L1`
- `L0A`
- `L0B`
- `L0C`

这些区域本质上是按目标 NPU 架构容量预分配的数组。`TASSIGN` 会从这些数组中为 Tile 绑定某一段内存。例如：

- 若对 `Loc == Mat` 的 Tile 调用 `TASSIGN(tile, 10)`，则该 Tile 会绑定到 `L1[10]` 开始的位置。

### 选择模拟目标架构

A2A3 和 A5 表示要模拟的目标架构，不是宿主机安装的 NPU 型号；CPU_SIM 执行不需要 NPU。
该选择影响内存配置及 `TROWSUM` 等具有架构分支的指令，不代表所有 CPU 指令都已分别实现两种架构的逐位模拟。

- 为后续初始化的线程局部内存模型选择默认架构：在启动阶段、任何线程使用模型之前，调用一次
  `pto::NPUMemoryModel::SetDefaultArch(pto::NPUArch::A5)`。未设置时默认为 `pto::NPUArch::A2A3`。
- 仅显式初始化当前线程：在绑定或访问 Tile 前调用
  `pto::NPUMemoryModel::Instance().Initialize(pto::NPUArch::A5)`。
- 同时设置默认架构并初始化当前线程：在启动阶段、其他线程使用模型之前，调用一次
  `pto::NPU_MEMORY_INIT(pto::NPUArch::A5)`。该辅助函数同时调用 `SetDefaultArch` 和 `Initialize`，
  但不会重新配置其他线程已初始化的模型。不带参数的 `NPU_MEMORY_INIT()` 会显式将默认架构
  重设为 A2A3，并以 A2A3 初始化当前线程。
- `SetDefaultArch` 不会重新配置已经初始化的线程局部实例。不要在执行期间并发修改默认架构，也不要在 Tile 仍引用模型存储时重新初始化模型。

外部模拟器集成需要在 kernel 执行前设置目标；集成层的 `a5sim` 名称本身不是 pto-isa 内部的架构选择开关。
更多信息请参考 `include/pto/cpu/NPUMemoryModel.hpp` 和 `include/pto/cpu/TAssign.hpp`。

### 内存容量覆盖

可通过以下环境变量覆盖各模拟内存区域的容量。变量值以字节为单位，且必须为正整数：

- `PTO_CPU_SIM_UB_BYTES`
- `PTO_CPU_SIM_L1_BYTES`
- `PTO_CPU_SIM_L0A_BYTES`
- `PTO_CPU_SIM_L0B_BYTES`
- `PTO_CPU_SIM_L0C_BYTES`

CPU_SIM 默认提供至少 512 KiB 的 UB 临时空间。应在初始化内存模型前设置这些环境变量。

## 自动内存分配

定义 `__PTO_AUTO__` 后，CPU_SIM 中的常规 `Tile` 支持惰性后备存储。如果 Tile 尚未通过 `TASSIGN` 绑定内存，则首次调用 `data()` 时会为其分配私有的主机内存。未定义 `__PTO_AUTO__` 时，常规 Tile 必须在访问前显式绑定内存。

后备存储来自主机内存，不对应 Tile 声明的内存位置，也不会与模拟的 UB、L1、L0A、L0B 或 L0C 缓冲区重叠。如果需要模拟内存位置、偏移、别名或通信行为，应使用 `TASSIGN`。不提供惰性后备存储的其他 Tile 抽象仍需显式绑定内存。

为避免 `__PTO_AUTO__` 模式下并发执行首次访问，CPU_SIM 的 `TMATMUL` 实现会在启动并行工作线程前，由调用线程完成输出、可选累加器和两个矩阵输入 Tile 的后备存储初始化。`TMATMUL_MX` 路径还会初始化两个缩放 Tile。

## 使用建议

对于常规 `Tile`，可采用以下两种策略之一：

- **直接内存绑定**：为每个 Tile 显式调用 `TASSIGN` 绑定内存，并手动计算合适的偏移。
- **惰性后备存储**：定义 `__PTO_AUTO__` 后不显式绑定 Tile，由首次 `data()` 调用分配私有主机内存。

直接绑定用于模拟指定的 NPU 内存区域；惰性后备存储适用于 CPU 侧私有 Tile 的正确性验证，不模拟具体片上地址。

## 已支持行为和后端差异

- CPU_SIM `TROWSUM` 根据调用线程已初始化的架构选择计算及检查路径，两种路径均接受但不访问 `tmp`。类型、布局和数值边界见 [TROWSUM](../isa/TROWSUM_zh.md#cpu_sim实现检查)，计算方式见下文 [TROWSUM 实现说明](#trowsum-实现说明)。
- 当各操作数的元素类型及运行时有效形状一致时，CPU_SIM `TADD` 和 `TABS` 支持操作数使用不同的 Tile 类型，包括混用静态和动态 `ValidRow`/`ValidCol` 模板参数。CPU_SIM 按每个操作数自身的 Tile 布局和物理形状计算索引；运行时有效形状不一致会触发断言。
- CPU_SIM 同时实现 `TCI(dst, start)` 和 `TCI(dst, start, tmp)`。三参数形式接受 `tmp` 但不访问其存储，其升序或降序序列语义与两参数形式相同。编写跨后端 kernel 时，仍须保留目标 NPU 后端要求的临时空间分配。
- CPU_SIM 实现了 `TCVT` 带或不带临时 Tile、显式或省略 `SaturationMode` 的全部四种重载。带临时 Tile 的形式接受但不访问 `tmp`，并与对应的不带临时 Tile 形式保持相同转换结果。CPU_SIM 默认使用 `SaturationMode::OFF`；跨后端 kernel 仍须保留 NPU 所需的临时空间。
- 使用 `SLayout::NoneBox` 的 `TileType::Vec` Tile 做 `TSTORE` 时，只有一种情形按 Tile 自身布局选择 GM 遍历方式，与硬件 DMA 一致：即在 ND/DN/NZ 同布局之外，`TSTORE` 额外允许的单行或单列配对。ColMajor 的 `[N, 1]` Tile 经 ND `GlobalTensor` 落盘，或 RowMajor 的 `[1, N]` Tile 经 DN `GlobalTensor` 落盘，都按连续向量写入，而不是按另一维的跨距散开。其余组合（包括列数大于 1 的 ColMajor Tile）仍按 `GlobalTensor` 的布局取映射。`TLOAD` 不存在该情形：`TileType::Vec` 加载只接受匹配布局。
- TileData `TPUSH`/`TPOP`/`TFREE` 使用主机侧 `TPipe` FIFO 模型。该模型会等待空闲槽位和就绪数据。对于 `Direction::DIR_BOTH`，C2V 和 V2C 使用独立的环形队列，每个方向各有 `SlotNum` 个槽位及独立的 payload、游标和同步状态。该模型通过模拟的 block/subblock 上下文协调 split lane。即使生成代码传入了非空 NPU GM workspace，TileData 数据也始终保存在主机 FIFO 状态拥有的槽位中，CPU_SIM 不会访问该 workspace。`TFREE` 会参与 CPU FIFO 的释放协议并释放对应方向，不是 A2A3 TileData 路径中的空操作。在 `TileSplitAxis::TILE_NO_SPLIT` 下，TileData `TPUSH` 按实际搬运窗口的形状（即所推送 Tile 的有效形状）排布槽位 payload，因此推送宽 Tile 的窄视图时，行与消费者弹出的 Tile 仍然对齐；split 轴仍使用生产者 Tile 的声明形状。CPU_SIM 当前不支持公共 GlobalData `TALLOC`/`TPUSH`/`TPOP`/`TFREE` 流程。
- CPU_SIM 中，Tile-vs-Tile `TCMPS` 重载逐元素比较 `src0[i,j]` 与 `src1[i,j]`。该行为与 A5 一致，与 A2/A3 的标量广播行为不同；标量重载仍按通常的标量比较语义执行。
- CPU arg-reduce 实现（`TCOLARGMIN`、`TCOLARGMAX`、`TROWARGMIN` 和 `TROWARGMAX`）支持 integral、`half`、`bfloat16_t` 和 `float` 源元素。索引输出必须为 `int32_t` 或 `uint32_t`，临时 Tile 参数在 CPU_SIM 中不使用。
- CPU_SIM 支持 `TGATHER` 和 `TSCATTER` 的 AIV 路径，但没有可用的 `CollEngine::CCU` 功能实现。CPU 头文件通过 deferred-fail 在编译期拒绝不支持的 CCU 调用；CCU 路径应使用 A5 NPU 后端。
- `SYNCALL`（包括带 workspace 的 Soft 形式）当前在 CPU_SIM 中只是兼容性空操作，不能作为 CPU 工作线程屏障。工作线程需要交换数据时，必须使用已有 CPU 实现的同步或通信操作。
- 外部 simulator 可通过 `pto::cpu_sim::register_hooks` 提供 subblock ID 和共享 `TPipe` 状态回调。未直接注册回调时，CPU_SIM 还会从主机进程解析 `pto_sim_get_subblock_id` 和 `pto_sim_get_pipe_shared_state` 符号。

### TROWSUM 实现说明

类型、布局和有效区域约束见 [TROWSUM 的 CPU_SIM实现检查](../isa/TROWSUM_zh.md#cpu_sim实现检查)。
以下为 [CPU 实现](../../include/pto/cpu/TRowSum.hpp) 的计算方式，不是新增的 NPU 接口约束。

- A5 浮点路径使用 `rowSumTree`，按连续 256 字节分组：`float` 每组 64 lane，
  `half` 每组 128 lane。组内按相邻元素二叉树归约；尾组未使用的 lane 补零，
  不读取源 Tile 的无效填充区。各组结果从正零开始按列地址递增顺序累加。
  每次树内加法和组间累加均转换回元素类型，包括 `half` 的组间舍入。
- A5 整数路径使用 `rowSumModular`，通过无符号模运算避免有符号溢出的未定义行为。
  `int16_t` 使用 32 位累加器，最终截取低 16 位；其他支持的整数按自身位宽回绕。
  有符号输出保留结果位模式，不执行饱和处理。
- A2A3 保留原 CPU 累加循环，使用 `TypeSum<TileDst>`：
  `half`/`bfloat16_t` 输出先用 `float` 累加，写回时转换；其他输出用自身类型。
  此路径不采用 A5 的归约树或无符号模运算溢出处理。
  行主序循环保留向量化提示：即使未启用 `-ffast-math`，Clang 也可能重排浮点加法。
  因此不保证严格从左到右累加，也不保证不同编译器的结果逐位一致。
- 不支持原生 BF16 的工具链可能将 `bfloat16_t` 定义为 `half` 的别名；
  该占位类型走 half 路径，不代表实现了 BF16 模拟。
- CPU 归约入口及执行函数均以 `uint32_t` 传递有效行、列数，避免 16 位截断；
  这不代表支持任意 32 位大小。Tile 维度及 `Numel` 仍为 `int`，
  布局、元素数可表示范围及可用存储约束仍然适用。
- 调用线程确保自身内存模型已初始化后选择架构，内部行并行工作线程按值接收该选择，
  不读取自身线程局部模型的默认架构；这不会初始化其他调用线程或模拟核工作线程。
  线程配置见[多核执行](#多核执行)。

A5 实现模拟特定归约顺序，不构成完整的 A5 真机逐位验证；
NaN payload、非正规数/flush-to-zero 等宿主机与设备间差异尚未证明等价。
A2A3 兼容路径也不承诺与真机逐位一致。依赖 A5 归约顺序时，不应启用 `-ffast-math` 等允许浮点重结合的选项。

## 多核执行

模拟核工作线程与单条指令内部的行并行线程不同。例如，`TROWSUM` 使用
`include/pto/cpu/parallel.hpp` 中的编译期控制项，`PTO_CPU_SIM_NUM_CORES` 不控制这组线程。
内部行并行需要至少两行、两个可用线程，且 `rows * cols >= PTO_CPU_PARALLEL_THRESHOLD_ELEMS`
（默认 16384）。`PTO_CPU_MAX_THREADS` 是编译期线程数上限（0 表示使用硬件并发数），
不是运行时环境变量。

`pto::cpu_sim::LaunchKernelMultiCore` 会为每个活跃的模拟核启动一个 CPU 工作线程，并初始化该线程的 block 和 subblock 执行上下文。默认配置的核数为 4，可通过 `PTO_CPU_SIM_NUM_CORES` 调整。应在初始化 CPU_SIM 运行时前设置该环境变量。

启动参数可指定请求核数、总工作项数、工作粒度、每个 block 的 subblock 数量以及显式 block 数量。`get_block_idx()`、`get_subblockid()` 和 `get_subblockdim()` 返回当前 worker 的 launch 上下文，`get_coreid()` 等同于 `get_block_idx()`。`get_block_num()` 返回 `PTO_CPU_SIM_NUM_CORES` 的配置值；当某次 launch 显式限制 block 数或根据工作项缩减活跃 block 时，该值可能大于实际活跃 block 数。

## 指令 Trace

指令 Trace 默认在构建时关闭。运行 CPU ST 时，可通过以下命令启用：

```bash
python3 tests/run_cpu.py --trace-mode
```

该参数会设置 CMake 选项 `PTO_CPU_SIM_TRACE_MODE`。启用 Trace 的构建会记录指令操作码、block 索引、指令序号、Tile 操作数和标量操作数。`LaunchKernelMultiCore` 会将合并后的 JSON Lines Trace 写入：

```text
cpu_sim_traces/<kernel_name>/launch_<id>/trace.jsonl
```

以下环境变量用于控制运行时 Trace：

- `PTO_CPU_SIM_TRACE_ENABLE`：对于已经启用 Trace 的构建，设置为 `0` 或 `false` 可关闭 Trace 收集。
- `PTO_CPU_SIM_TRACE_DIR`：覆盖默认的 `cpu_sim_traces` 输出目录。

应在初始化 CPU_SIM 运行时前设置这些环境变量。也可使用 `include/pto/cpu/trace.hpp` 中的接口重置、查看、复制或序列化当前线程的指令记录。
