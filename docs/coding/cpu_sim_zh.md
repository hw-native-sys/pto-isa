# CPU_SIM

CPU_SIM 是一个面向纯 CPU 系统执行的后端实现。

与 NPU 后端相比，CPU_SIM 当前存在以下差异和限制：

- 每个 CPU 工作线程内的 PTO 指令同步执行。对于已支持的同步和通信操作，CPU_SIM 会使用 CPU 同步原语进行模拟，其中包括 TileData `TPUSH`/`TPOP`/`TFREE` FIFO 流程。
- 使用特定的内存模型来模拟 NPU 内存层次（见下文）。
- 多线程支持尚不完整。`Tile` 对象的内存访问不具备线程间同步能力，因此除已支持的通信操作外，不建议跨线程共享 Tile。

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

当前支持的架构包括 A2A3 和 A5。可通过 `pto::NPUMemoryModel::Initialize` 为每个线程指定要模拟的架构；该函数应在每个线程中调用一次。

- 若不显式调用，则默认使用 A2A3 架构。

更多信息请参考 `include/pto/cpu/NPUMemoryModel.hpp`。

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

## 使用建议

对于常规 `Tile`，可采用以下两种策略之一：

- **直接内存绑定**：为每个 Tile 显式调用 `TASSIGN` 绑定内存，并手动计算合适的偏移。
- **惰性后备存储**：定义 `__PTO_AUTO__` 后不显式绑定 Tile，由首次 `data()` 调用分配私有主机内存。

直接绑定用于模拟指定的 NPU 内存区域；惰性后备存储适用于 CPU 侧私有 Tile 的正确性验证，不模拟具体片上地址。

## 已支持行为和后端差异

- TileData `TPUSH`/`TPOP`/`TFREE` 使用主机侧 `TPipe` FIFO 模型。该模型会等待空闲槽位和就绪数据，在 `Direction::DIR_BOTH` 下区分 C2V 和 V2C 流量，并通过模拟的 block/subblock 上下文协调 split lane。`TFREE` 会参与 CPU FIFO 的释放协议，不是 A2A3 TileData 路径中的空操作。CPU_SIM 当前不支持公共 GlobalData `TALLOC`/`TPUSH`/`TPOP`/`TFREE` 流程。
- CPU_SIM 中，Tile-vs-Tile `TCMPS` 重载逐元素比较 `src0[i,j]` 与 `src1[i,j]`。该行为与 A5 一致，与 A2/A3 的标量广播行为不同；标量重载仍按通常的标量比较语义执行。
- CPU arg-reduce 实现（`TCOLARGMIN`、`TCOLARGMAX`、`TROWARGMIN` 和 `TROWARGMAX`）支持 integral、`half`、`bfloat16_t` 和 `float` 源元素。索引输出必须为 `int32_t` 或 `uint32_t`，临时 Tile 参数在 CPU_SIM 中不使用。
- CPU_SIM 支持 `TGATHER` 和 `TSCATTER` 的 AIV 路径，但没有可用的 `CollEngine::CCU` 功能实现。CPU 头文件通过 deferred-fail 在编译期拒绝不支持的 CCU 调用；CCU 路径应使用 A5 NPU 后端。
- `SYNCALL`（包括带 workspace 的 Soft 形式）当前在 CPU_SIM 中只是兼容性空操作，不能作为 CPU 工作线程屏障。工作线程需要交换数据时，必须使用已有 CPU 实现的同步或通信操作。
- 外部 simulator 可通过 `pto::cpu_sim::register_hooks` 提供 subblock ID 和共享 `TPipe` 状态回调。未直接注册回调时，CPU_SIM 还会从主机进程解析 `pto_sim_get_subblock_id` 和 `pto_sim_get_pipe_shared_state` 符号。

## 多核执行

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
