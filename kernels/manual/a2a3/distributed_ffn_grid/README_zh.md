# Single-device Multi-block FFN GridPipe Demo

## 整体目标

本 demo 在 A2/A3 的单卡逻辑 FFN 网格上验证三个分布式 FFN GridPipe 集合通信接口 —— **TPUSH**、**TBROADCAST**、**TREDUCE**。host 在选定 device 上启动单进程，并 launch `gridRows * gridCols` 个 block；每个 block 对应一个逻辑 cell。共有 **四个例子**，每个对应一组 (接口, FFN 模式)，全部跑在同一套纯 1D N-cut 4×8 = 32-cell 拓扑上，并使用真实的 DeepSeek-v4 Pro 形状（M=T=8、H=7168、I=3072）：

| 例子（运行脚本 / 可执行文件） | 验证的接口 | FFN 模式 | 跨 cell 集合通信 |
| --- | --- | --- | --- |
| `run_tpush_reducesum.sh` / `distributed_ffn_grid_tpush_reducesum` | **TPUSH** | ReduceSum | 显式 `TPOP(pipe, ..., prodId)` + `TADD` + `TPUSH(pipe, ..., consId, isLastTransfer)` 中继 |
| `run_tpush_allgather.sh` / `distributed_ffn_grid_tpush_allgather` | **TPUSH** | AllGather | 最近邻 `TPUSH`/`TPOP` 中继 gather（fan-in-1 DAG） |
| `run_tbroadcast_allgather.sh` / `distributed_ffn_grid_tbroadcast_allgather` | **TBROADCAST** | AllGather | `TBROADCAST<GridGroup>` MPSC 组广播 |
| `run_treduce_reducesum.sh` / `distributed_ffn_grid_treduce_reducesum` | **TREDUCE** | ReduceSum | 分 channel 的 `TREDUCE<GridGroup, Sum>` N→1 扇入（4+3 源分批） |

每个例子都用 `1e-3` 容差把 `[T, H]` 输出与 `golden.bin` 比对。四个例子在 NPU 上全部 **位精确通过**（`max diff = 0`，用 `-r npu` 运行）；详见 [位精确性说明](#位精确性说明)。

跨 cell 的集合通信走 A2/A3 GridPipe mock 后端：mock 中由 GM 撑起的本地 SRAM windows、fake `HcclDeviceContext` window 指针、每 channel 的 ready/free/close 计数器、`dcci/dsb` fence 和自旋等待。该 demo 验证的是编程模型和同设备 mock 路径，不是多卡通信验证。

Unicast 在运行期显式传 peer id（TPUSH 的 `consId`、TPOP 的 `prodId`）；并发组广播仍使用 `TBROADCAST<GridGroup>` / `TPOP<GridGroup>`。独立的广播冒烟测试位于 `smoke/`。

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `README.md` / `README_zh.md` | 英文 / 中文说明文档。 |
| `CMakeLists.txt` | 构建四个 host 可执行文件及其 mixed Cube/Vec device kernel shared library（外加两个冒烟测试 target）。 |
| `run_treduce_reducesum.sh` / `run_tpush_reducesum.sh` | 配置 CANN、生成数据、配置 CMake、构建并运行 TREDUCE / TPUSH ReduceSum 例子。 |
| `run_tbroadcast_allgather.sh` / `run_tpush_allgather.sh` | 配置 CANN、生成数据、配置 CMake、构建并运行 TBROADCAST / TPUSH AllGather 例子。 |
| `ffn_config.hpp` | 编译期网格形状、tile 形状、GridPipe window 字节数、buffer 字节数、SwiGLU clamp 上下界、A3 精度映射表与 Batcher GM arena 字节数。 |
| `kernel_launch.hpp` | host 侧 mixed kernel launch 接口声明（每个例子一份）。 |
| `main_treduce_reducesum.cpp` / `main_tpush_reducesum.cpp` | ReduceSum host driver：ACL 初始化、fake HCCL context / 本地 GridPipe windows、工作 buffer、Batcher 加载/分发、kernel launch、golden 比对、资源清理。 |
| `distributed_ffn_grid_treduce_reducesum_compute_kernel.cpp` | TREDUCE ReduceSum kernel：EAST+SOUTH 扇入复用 C 个 TPUSH ring、前向绝对计数和反向原子 FREE 接力。 |
| `distributed_ffn_grid_tpush_reducesum_compute_kernel.cpp` | TPUSH ReduceSum kernel：EAST 与 SOUTH 跨 launch 共用一个 channel，以显式 TPOP + TADD + TPUSH 覆盖 close 与接力计数重绑定。 |
| `main_tbroadcast_allgather.cpp` / `main_tpush_allgather.cpp` | AllGather host driver。 |
| `distributed_ffn_grid_tbroadcast_allgather_compute_kernel.cpp` | TBROADCAST AllGather kernel：两个 gather 阶段用 `TBROADCAST<GridGroup>` + `TPOP<GridGroup>`。 |
| `distributed_ffn_grid_tpush_allgather_compute_kernel.cpp` | TPUSH AllGather kernel：两个 gather 阶段用双向 `TPUSH`/`TPOP` 中继。 |
| `batcher.hpp` | host 侧 **GM 模拟 Batcher**：在 GM 中持有全量输入 + 全量 DRAM 常驻权重，沿列切分成 per-cell shard，广播 x，并暴露输出收集区。 |
| `tpipe_tmov_inl.hpp` | 把 Cube↔Vec 的 C2V/V2C 搬运封装成方向化 `TMOV` 重载，内部转发到现有 `TPUSH`/`TPOP`，使 kernel 正文不再出现该 handshake。 |
| `gridpipe_payload_inl.hpp` | 本地 GridPipe payload 钩子与 fake-window 适配器：peer slot/SCB 解析、tile 到 producer L1 staging、producer L1 到对端 ring 的拷贝、本地接收 ring drain，以及 TPOP 本地性守卫。 |
| `smoke/` | Vec-only GridPipe 广播冒烟测试（`bcast_smoke_*` + `run_bcast_smoke.sh`）。 |
| `../../../../include/pto/npu/a2a3/grid_cce_intrinsic.hpp` | Grid CCE 门面：统一 L1 搬运、绝对值 `sync_hscb`、反向扇入 `atom_add_hscb` 与阻塞 `wait_ipc_scb`。A3 mock 用 GM window 表示 L1 地址段。 |
| `../../../../include/pto/npu/a2a3/grid_intrinsic.hpp` | GridPipe A2/A3 数据模型 + mock 支持：每 channel 的 ready/free/close SCB、生产者/消费者绑定、每消费者三态 FSM、持久化接力计数、mesh/group 解析器、fault 哨兵，以及 `GmSramArena` 的 TPOP 读本地性守卫。 |
| `scripts/gen_data.py` | 生成 Batcher 消费的全量 fp16 X/weight 张量（`x_full`、`w_gate_full`、`w_up_full`、`w_down_full`）以及 fp32 SwiGLU `golden` 参考结果。四个例子统一用 `--pure-ncut` 产出扁平全量张量。 |
| `build/` | 被忽略的生成 build 目录。 |
| `out/` | 被忽略的生成数据目录。 |

## 位精确性说明

用 `-r npu` 运行（`sim`/`camodel` 模式会在 `aclrtSetDevice` 报 507033）；共享主机上每次运行都要走 `task-submit`。四个例子全部产出 `max diff = 0`（对 `golden.bin`）——位精确，而不只是落在 `1e-3` 容差内。曾经遮住这一点的是两个真实 bug，现都已修复：

- **分 channel 的集合通信接力（TBROADCAST/TREDUCE）。** payload 不再按源 rank 分槽。源序号 `s` 使用普通 GridPipe ring `slotBase[s%C]`；任意时刻最多 C 个源并行，既不共享 payload 地址也不共享前向计分板。超过 C 的源只有在前任 CLOSE 且反向 FREE 证明 ring 已 drain 后才复用 channel。READY/CLOSE 用单调绝对值 `sync_hscb`，只有反向 FREE 扇入使用 `atom_add_hscb`。
- **phase-D 输出 T 步长（两个 AllGather kernel）。** AllGather 的 y-shard `[T, Hc]` 写进**完整** `[T, H]` 输出，所以其行步长必须是完整输出宽度 `kHfull`（= `H` = 7168）。从 `hidden_full` store 复制粘贴时遗留成 `kIfull`（= `I` = 3072），把 y 的第 1–7 行打乱（≈50 % 零输出 / 大漂移）。两个 AllGather kernel 的 `GY` store 各改一行 `kIfull` → `kHfull` 即修复。

`treduce` ReduceSum 还要求其 per-cell partial buffer（`partialBuf` / `rowPartialBuf`）以**段主序（segment-major）**布局——每个 `[T, kHBase]` H 段在偏移 `h*(T*kHBase)` 处连续存放——这样贡献者可把一段连续数据 stage 到所选 channel ring；只有最终的 `yFull` 保留 strided `[T, H]` golden 布局。

32 个 block 的 launch 仍然无法在 24 个物理 AICore 上一波跑完——单波 launch 的过载会让 phase C 死锁（COL 组跨满 4 行，首批 cell 自旋等待拿不到核的二批 row-3 门铃）。因此 host 按 `--phys-cores` 切波启动（`rowsPerWave = physCores/cols`、`colsPerWave = physCores/rows` → phase B、C 各 2 波，共 6 次 launch、~5 ms）。有了步长修复之后，分波只是调度问题，不再是可靠性问题。

## 运行流程

1. 每个 `run_*.sh` 解析参数。默认值是 4×8 = 32-cell mesh 上的真实 DeepSeek-v4 Pro 形状：`gridRows=4`、`gridCols=8`、`T=8`（token tile）、`H=7168`、`Fi=96`（per-cell I shard；完整 `I = Fi * cells = 3072`）、`n-ranks=1`、`phys-cores=24`。
2. 如果没有指定 `--build-only`，`scripts/gen_data.py --pure-ncut` 生成 Batcher 消费的全量扁平张量（`x_full`、`w_gate_full`、`w_up_full`、`w_down_full`）以及 SwiGLU `golden.bin`。
3. CMake 为每个例子构建两个 target——一个 `..._mixed_kernel` 的 `dav-c220` shared library 和对应的 host 可执行文件（例如 `distributed_ffn_grid_treduce_reducesum_mixed_kernel` + `distributed_ffn_grid_treduce_reducesum`）。两个 AllGather kernel 还会额外带 `-DCONFIG_FFN_GRID_ALLGATHER` 编译。
4. host 在选定 device 上初始化 ACL。
5. host 按 `gridRows * gridCols` 个 cell 分配连续 device buffers。
6. host 分配每个 cell 一个本地 GridPipe SRAM window（mock 中由 GM backing），并构造 fake `HcclDeviceContext`：

```text
windowsIn[cell] = reduce_pipe_windows_dev + cell * FFN_GRID_WINDOW_BYTES
rankNum = gridRows * gridCols
winSize = FFN_GRID_WINDOW_BYTES
```

7. host **Batcher**（`batcher.hpp`）把全量输入 + 全量 DRAM 常驻权重载入 GM，沿列切分成 per-col shard，并 per-row 广播 x（见 [Batcher（GM 模拟）](#batchergm-模拟swiglu-与-a3-精度映射)）。
8. host 通过 `rtGetC2cCtrlAddr()` 获取 FFTFS base address，并单次（或多波）launch `DistributedFfnGridMixedKernel`，共 `gridRows * gridCols` 个 block。
9. kernel 内 Cube 和 Vec 分支通过 A2/A3 `TPipe` FIFO 交换中间 tile。这些 C2V/V2C 搬运在 kernel 正文里以方向化 `TMOV` 表达（`TMOV(pipe, tile)` 生产、`TMOV(tile, pipe)` 消费），底层 `TPUSH`/`TPOP` 隐式完成（见 `tpipe_tmov_inl.hpp`）：

```text
Cube:
  X[row] @ W_gate[col] -> gatePartial[row,col] --TMOV C2V-->
  X[row] @ W_up[col]   -> upPartial[row,col]   --TMOV C2V-->

Vec:
  hidden[row,col] = fp16(SwiGLU(gatePartial) * upPartial)   # SiLU(clamp(gate)) * up
  hidden[row,col] --TMOV V2C-->

Cube:
  hidden[row,col] @ W_down[col] -> downPartial[row,col] --TMOV C2V-->

Vec:
  downPartial --跨列 GridPipe 归约--> yOutput[row] 落在最终列
```

跨 cell 的 reduce 与 gather 保持显式的 GridPipe 调用；只有 block 内 Cube↔Vec 的 C2V/V2C 搬运被收敛到 `TMOV`。其中 ReduceSum 的 EAST/SOUTH 归约：`treduce` 用 `TREDUCE<GridGroup, Sum>` 在 sink 做 N→1 扇入，`tpush` 用 peer-id TPOP + TADD + TPUSH 逐跳累加。

10. host 同步 stream，检查 GridPipe fault flags，拷回 `yOutput` 并与 `golden.bin` 比对。

## 关键设计

### Batcher（GM 模拟）、SwiGLU 与 A3 精度映射

本 demo 对齐 `WSE-FFN-tile级全展开图.svg`，该图把外部 **Batcher** 设定为全量输入与全量 DRAM 常驻权重的持有者，负责切分/分发到各核并收集输出。A2/A3 没有该硬件，因此 `batcher.hpp` 完全用 GM 模拟 Batcher：

- **全量权重常驻 GM**（`w_gate_full`/`w_up_full` `[H,F]`、`w_down_full` `[F,H]`），对应 SVG 的 `DRAM 常驻`。
- **分发（Distribute）**：把全量权重列切，写出一个连续的 per-col shard 到 per-col GM 区。每个核再 TLOAD 自己的 shard（DRAM→L1 流式），就像核流式读取 Batcher 投递的权重 tile。
- **广播（Broadcast）**：把全量 `x` 写入 GM；同一行的每列都读同一份 `x`（广播，"复制 broadcast → N 核"）。这也去掉了旧的 per-cell 冗余：`x` per-row、权重 per-col。
- **收集（Collect）**：AllGather 各核把 y shard、ReduceSum 的 EAST/SOUTH 归约把行内和直接写进 Batcher 的 `y` GM 区。

kernel 按 `(row, col)` 寻址 Batcher 存储：`x = xFull + row*…`、`w = wShards + col*…`、`y = yFull + row*… (+ col*Hc)`。

SVG 激活为 **SwiGLU = SiLU(clamp(gate)) · up**（"SiLU + clamp(max=10)"）。Vec 分支在 fp32 下用已有指令组合 SiLU：`SiLU(g) = g/(1+e^-g)`，经 `TMAXS`/`TMINS`（clamp ±10）、`TMULS(-1)` → `TEXP` → `TADDS(1)`（分母）、`TDIV`，再与 `up` `TMUL`。`gen_data.py` golden 使用完全一致的 clamp+SiLU。

SVG 还携带 A3 不支持的低位宽精度（FP4 权重、FP8 激活、BF16 I/O）。按扩展设计，tile 图中的每种精度都映射到**一个 A3 支持的精度**（见 `ffn_config.hpp` 表格）：FP4/FP8/BF16 → `half`，FP32 累加/输出保持 `float`。`act_quant` 与权重 `unpack` 因此在 kernel 中以具名的、零开销恒等阶段存在——只标注 SVG 转换本应所在的位置，不引入 A3 不支持的转换。fp16/fp32 数据通路本身即映射后的结果。

### Mixed Cube/Vec 单次 launch

device kernel 编译为 `dav-c220`。Cube 和 Vec 分支分别由 `__DAV_CUBE__`、`__DAV_VEC__` 保护，两个分支位于同一个 kernel source 中，通过 A2/A3 `TPipe` ready/free handshake 同步。

### 隐式 C2V/V2C `TMOV`

`tpipe_tmov_inl.hpp` 新增两个 `TMOV` 重载，让 kernel 正文用单条 tile-move 表达 Cube↔Vec 搬运，而不再显式写 `TPUSH`/`TPOP`：

- `TMOV(pipe, tile)`：生产侧，转发到 `TPUSH`（把 `tile` 写入 C2V/V2C FIFO）。
- `TMOV(tile, pipe)`：消费侧，转发到 `TPOP`（把下一个 slot 读入 `tile`）。

哪个物理核负责写/读、pipe 是 C2V 还是 V2C，都仍由 `TPipe` 类型及其 `__DAV_CUBE__`/`__DAV_VEC__` 保护决定，因此调用点与方向无关。这两个重载只接收 `(pipe, tile)`/`(tile, pipe)` 两个参数（没有 wait-event 包），比通用的 tile-to-tile `TMOV(dst, src, ...)` 更特化；重载决议因而会对任意 `TPipe`/tile 组合选中它们，而其它所有 `TMOV` 用法保持不变。这样既把 Cube↔Vec handshake 隐藏在调用点之后（贴近真实 WSE fabric move 隐藏生产/消费拆分的方式），又原样复用现有 `TPUSH`/`TPOP` 的同步与 record 机制。

### 单卡逻辑网格

`get_block_idx()` 是 row-major cell id：

```text
cell = get_block_idx()
row  = cell / gridCols
col  = cell % gridCols
```

所有 cell 都在同一个 device 上运行。`gridRows` 控制 data-parallel token tile 数，`gridCols` 控制 model-parallel FFN shard 数。

### 本地 GridPipe mock

host 分配 `gridRows * gridCols` 个本地 SRAM windows（mock 中由 GM backing）。`TPUSH(..., consId)` 通过 `ResolvePeerSlotAddr` 解析该消费者的 SRAM slot，写 payload 后发布 READY；`TPOP(..., prodId)` 等本地 READY、读取本地 slot，再向该生产者归还 FREE credit。

mock 使用 GM flag polling 和 cache maintenance 在 A2/A3 上模拟 LPU WSE 预期的 `SPR` / `WFE` 行为。

### NoC 只写不读的地址段 SRAM 模型（`GmSramArena`）

为了贴近真实硬件，mock 把未来硬件的"每核私有 SRAM"显式建模为一个 **GM 地址段（address segment）arena**：那块连续的 `gridRows*gridCols * FFN_GRID_WINDOW_BYTES` window 缓冲被切成等长的 per-core 段，于是第 `c` 段（即 `windowsIn[c]`）就是第 `c` 个核的私有 SRAM：

```text
段 c = [base + c*winSize, base + (c+1)*winSize)   // base == windowsIn[0]
```

每个段内部把接收端与生产者空间分开：

```text
control + record | unicast 接收 ring | 可选 broadcast 接收 ring | producer staging slot
                                                                                   [SlotStride bytes]
```

producer slot 固定追加在所有接收端区域之后。`TPUSH` / `TBROADCAST` 先把 tile 落到这块本核 L1，再把该 L1 源地址映射到对端 payload ring。这与真实 WSE 上 vector 数据和 L1 共用一份 SRAM 的模型一致，且 producer 源空间不会与本核正在消费的 ring 别名。

`GmSramArena`（位于 `include/pto/npu/a2a3/grid_intrinsic.hpp`）持有 `{base, segBytes, numSegs}` 以及 `SegmentOf` / `InSegment` 判定函数；demo 在 device 侧从 fake `HcclDeviceContext` 的 window 表构造它（`SramArenaFromCtx`）。它是"某地址归哪个核所有"的唯一真相来源。

这样就把真实硅片的 NoC 约束显式化并**强制**起来：fabric 只能跨核**写**，不能跨核**读**。

- `TPUSH(pipe, tile, consId, ...)` 从本核独立 producer slot 读源数据，写入**消费者核的接收 ring**——这是跨段写，正是 fabric 的行为。
- `TPOP(pipe, tile, prodId)` 只能 POP **本核自己**的段。`GRID_TRY_TPOP_IMPL` 在 payload 读取前先调用 `PopSlotIsLocal` 守卫；一旦发生跨段读，就写入 `kFaultPopNonLocal`（`0x205`，"pop non-local segment"）并放弃本次 pop，host 的 `CheckGridPipeFaults` 会报出来。

在 native 硬件上 `PopSlotIsLocal` 是恒为 `true` 的 no-op：TPOP 的读地址天然就是本地的，因为 fabric 根本没有远程读通路。这个守卫只是为了 A2/A3 mock——mock 用一块 GM window 模拟 SRAM，它物理上可以读任意地址，若没有守卫，demo 就可能悄悄依赖一次硅片做不到的远程读。每个 A2/A3 kernel 都会编入一条 `static_assert(GmSramArenaSelfCheck())`，因此段计算一旦回归就会在编译期失败，而不是把 pop 误路由出去。

> `pto::comm` 版本（`TREDUCE` / `TGATHER`）有意**不**遵守该约束：它们是 root 直接读取每个 rank 的 collective（HCCL/RDMA 式的远程读），与 WSE NoC 是不同的内存模型。只有 GridPipe `TPUSH`/`TPOP` 路径被约束成只写不读。

### IPC_SCB 计分板 intrinsic API

GridPipe 的 ready/free/close 同步走 V8 IPC_SCB 计分板路线，每个 channel 各有一条对应 SCB。握手 intrinsic 位于 `include/pto/npu/a2a3/grid_cce_intrinsic.hpp` 的薄 CCE 门面层——每条门面在 `PTO_GRID_CCE_NATIVE` 下 1:1 转发到 `__builtin_cce_*`，否则在 A2/A3 mock 中用 GM 字 + cache 维护（`dcci`/`dsb`）emulate 同语义：

- `copy_l1_to_neighbor_l1(dstNeighborSlot, srcProducerSlot, transferScratch, bytes)`（G1 / HW-DEP-0）：把独立的本核 producer L1 slot 写入对端接收 ring。`transferScratch` 只是 A3 GM mock 的 DMA 中转 UB，不是体系结构源地址。搬运不自同步，随后由 `sync_hscb(READY)` 发布 data-ready。
- `sync_hscb(peerScb, absCount)`（V8 `SYNC_HSCB`/`ST_HSCB`，G2——复用 HSCB store + 邻居 IPC_SCB 寻址 / HW-DEP-1）：把绝对计数 store 进对端的 `ready_scb`、`free_scb` 或 `close_scb`；目标种类与 peer 已由 `RemoteScbPtr` 解析进 `peerScb`。
- `atom_add_hscb(peerScb, delta)`：只为反向 FREE 扇入原子累加对端计分板；前向 READY/CLOSE 不使用它。A3 mock 下译为 s32 原子累加；native 门面映射 `__atom_add_hscb`，最终 WSE ISA 需确认 peer IPC_SCB 定址与唤醒语义。
- `wait_ipc_scb(localScb, threshold, slot)`（V8 `WAIT_SPR`，G3——复用 IPC_SCB 阻塞等待）：读+阻塞合**一条**指令——入口读本核 IPC_SCB，已 `≥ threshold` 即放行，否则阻塞当前 pipe 至对端 `sync_hscb` store 唤醒。V8 去掉了 V7 的 `MOV_SPR2X` 非阻塞 peek，无单独读步。demo 实际调 `wait_ipc_scb_sim(..., maxSpins)` 这层 mock 包装——加自旋超时哨兵，使握手死锁能以 fault 暴露而非挂死测试；文档化的硬件接口仍是上面的 void `wait_ipc_scb`。

payload 目标地址解析（把本地接收 ring slot / 计分板字解析为对端 GM window 中同字节偏移）是 `gridpipe_payload_inl.hpp` 中的普通 helper（`ResolvePeerSlotAddr` / `RemoteScbPtr`），非 intrinsic；源地址则独立固定为 `producerSlotBase`。TPOP 只 drain 本核接收 ring，`PopSlotIsLocal` 会拒绝误连的跨段读。

native lowering 对接真实 CCE HSCB/IPC_SCB 栈。编译器的 copy builtin 仍保留历史 `ubuf` 命名，但门面传入的始终是独立的统一 L1 producer 地址，不再把调用者 tile 地址当作 NoC 源映射。A3 mock 用 GM + cache maintenance 表示 SCB 和两个 L1 区域。

`TPUSH(pipe, tile, consId)` 先等本核生产者 channel 的 `free_scb`，把 tile stage 到 `producerSlotBase`，从该 L1 区域拷到对端独立选择的消费者 channel ring，再发布 `prod_idx`。`TPOP(pipe, tile, prodId)` 等本核消费者 channel 的 `ready_scb`，只读该本地接收 ring，然后把 `cons_idx` 发布到对端已协商的生产者 channel 的 `free_scb`。两端 channel 下标不要求相同。

### 时分 MPSC 接力计数

生产者为每个消费者保存一条 FSM（`UNBOUND`、`ACTIVE`、`CLOSED`），并维护本核生产者 channel 状态（`UNBOUND`、`ACTIVE`、生产者 `CLOSED`）。`ACTIVE` 直接走常规 TPUSH 快路径；对 `UNBOUND` 或 `CLOSED` 消费者，生产者先在本地选择未使用或生产者 `CLOSED` 的 channel，没有资源就等待。随后把 `[生产者 block id，本地生产者 channel，代次 token]` 入队到消费者按源编号索引的 bind-request 区；生产者端资源不再由消费者分配。

请求区为每个逻辑生产者保留一条独立 cache line（覆盖 runtime 最多 64 个逻辑 window），因此多个生产者同时到达时拥有不同的远端 writer，不会在线粒度 cache writeback 中互相覆盖。消费者以 round-robin 选择 pending 项（显式 TPOP 指定的生产者优先），完整处理一个 bind 及其回复后才选择下一项；回复写到生产者按消费者编号索引的 response 区。持久化的代次 token 可防止迟到回复误完成后续请求。

对选中的请求，消费者独立遍历自己的接收 channel：优先从未使用的 channel，否则只要求 `close_scb > closeBaseline`，不再要求 `cons_idx >= close_scb`。尚未 drain 的 turn 会把生产者身份和结束边界保存在持久记录中；bind 把旧结束值作为新 `prod_idx` 基线，并把当前 `cons_idx` 作为绝对 FREE 基线回传，因此常规 ring 背压会阻止新生产者覆盖旧 payload。请求和回复都最后写 commit 字；计数跨生产者单调接力，不清 ring，也不复位 SCB。

A2/A3 公共接口为 `TPUSH(pipe, tile, consId, isLastTransfer, events...)`；省略布尔参数等价于 `isLastTransfer == false`。只在当前生产者→消费者时段的最后一块上设为 `true`。最后一次 TPUSH 在 payload 与 READY 之后把同一绝对计数发布到对端消费者 channel 的 `close_scb`，再把该消费者 FSM 和本地生产者 channel 都置为 `CLOSED`。payload 所有权仍是时分 MPSC，但现在允许多条 bind request 并发 pending，由消费者串行处理。

### 组广播与归约数据搬运

`GRID_TBROADCAST` 只 stage 一次：group arena 对 block id 仿射时把所选 channel/slot 交给 `copy_l1_to_group`，否则按成员调 `copy_l1_to_neighbor_l1`。每个接收者等待该 channel 的绝对 READY/CLOSE，做本地 drain，再以原子 FREE 放行下一位 owner。组 `TREDUCE` 由每个成员调用：最多 C 个贡献核写入 sink 的不同 ring，sink 按源序归约，并用反向原子 FREE 放行各 channel 的下一源。

### fp32 归约

归约 slot 携带 fp32 `[T, H]`，所以 `FFN_SLOT_BYTES = T * H * 4`。这让 `downPartial`、`yOutput` 和 `golden.bin` 都保持 fp32，host 可直接做容差比较。ReduceSum 按 H 分段（`kHSegs` = 7）：`treduce` 把每段 stage 到分配的 sink ring，`tpush` 则用 peer-id TPOP + TADD + TPUSH 逐跳中继。ROW/COL 复用同一 window；跨阶段生产者身份变化由固定 channel bind 接力绝对 READY/FREE 基线。

### Peer-id unicast

Unicast 的方向由调度推导，不编码进指令类型。TPUSH 接收目标逻辑 block id（`consId`），TPOP 接收来源逻辑 block id（`prodId`），`RemoteScbPtr` 解析该 peer window 中对称的 slot/SCB 偏移。因此同一个 GridPipe 可以跨相位服务不同 peer。

### 并发组广播（TBROADCAST）

`TBROADCAST<GridGroup>`（`ROW`/`COL`/`SUBRECT`）把源序号 `s` 映射到 channel `s%C`，并写普通 GridPipe ring 的 `(sequence%SlotCount)`。整次扇出只付一次 publish fence，随后源以 `sequence+1` 绝对覆盖该 channel 的 READY/CLOSE；它不是按跳展开的 `TPUSH` 循环。

任意时刻最多 C 个源真实并发。源数超过 C 时，各 channel 的源按 owner 顺序分批。旧源 CLOSE 后下一源即可 bind；bind 将旧结束绝对序列和当前消费序号接力给下一源，反向 FREE 原子信用阻止其覆盖尚未消费的槽位。源集合就是实际调用 TBROADCAST 的 rank，不再需要额外的源范围配置。8 路行 AllGather 实际执行 4+4，4 路列 gather 一批完成；一批需要的成员必须在同一 hardware wave 常驻。

### GridPipe 冒烟测试

`bcast_smoke` 是 Vec-only 纯搬运冒烟测试（无 Cube、无 matmul、无数据文件），复用同一 GM-backed mock。源 cell 把带标记的 fp32 `[T, W]` tile 广播到行、列或子矩形，接收方取回并写出，host 校验 `out[cell] == in[source]`。

### AllGather 版本

两个 AllGather 例子（`run_tbroadcast_allgather.sh`、`run_tpush_allgather.sh`）与 ReduceSum 例子共用同一套 pure-N-cut 数据（`scripts/gen_data.py --pure-ncut`）；kernel 带 `-DCONFIG_FFN_GRID_ALLGATHER` 编译，使 host Batcher 把 `W_down` 沿输出 **H** 切（每个核拿一个 `[I_full, Hc]` shard，`Hc = H / cells`），并把跨 cell 工作变成两阶段 gather——在每个核上重建完整的 fp16 `hidden [T, I_full]` 后再做 down GEMM，于是每个核只写一个 `Y[:, Hc]` 输出 shard，不再需要 post-down ReduceSum。host 把 shard 拼接后与 `golden.bin` 比对。Pure-N-cut 要求 `--model-tile`（H）能被 cell 数（`grid-rows * grid-cols`）整除（保证 `Hc` 为整数宽度），且完整 `I`（`ffn-tile * cells`）能被 cell 数整除。

## 运行方法

### 仅编译

```bash
bash run_treduce_reducesum.sh    --build-only
bash run_tbroadcast_allgather.sh --build-only
```

### NPU 运行

脚本默认就是 DeepSeek-v4 Pro 形状（4×8 = 32 cell），所以直接调用跑的就是真实形状：

```bash
bash run_treduce_reducesum.sh    -r npu -v Ascend910B1 --device-id 0
bash run_tpush_reducesum.sh      -r npu -v Ascend910B1 --device-id 0
bash run_tbroadcast_allgather.sh -r npu -v Ascend910B1 --device-id 0
bash run_tpush_allgather.sh      -r npu -v Ascend910B1 --device-id 0
```

共享主机上，每次运行都要包一层 `task-submit`（例如 `task-submit bash run_treduce_reducesum.sh -r npu --device-id 0`）。

### GridPipe 冒烟测试

```bash
# 单源广播：1x5 行，源在 col 2（--span-col 1 + Rx1 网格切到列广播；--subrect 1 切到子矩形广播）
bash smoke/run_bcast_smoke.sh -r npu -v Ascend910B1 --device-id 0 --grid-cols 5 --src 2
```

该脚本支持 `--build-only`，且不需要生成数据。

### 常用参数

```text
-r, --run-mode      sim 或 npu，默认 npu（sim/camodel 在 aclrtSetDevice 报 507033）
-v, --soc-version   默认 Ascend910B1
-n, --n-ranks       固定为 1
-d, --device-id     ACL device id；默认依次取 TASK_DEVICE、FFN_GRID_DEVICE_ID、ASCEND_DEVICE_ID、DEVICE_ID、0
--grid-rows         单卡逻辑网格行数，默认 4
--grid-cols         单卡逻辑网格列数，默认 8
--token-tile        每个 cell 的 token tile T（M），默认 8
--model-tile        hidden dim H，默认 7168；pure-N-cut 要求 H % (grid-rows*grid-cols) == 0
--ffn-tile          每列 intermediate dim I_shard，默认 96（完整 I = ffn-tile*cells = 3072；须能被 cell 数整除）
--phys-cores        仿真用物理 AICore 数，默认 24（分波由此推导；<32 时强制多波 launch）
--build-only        只编译，不生成数据和运行
```

广播冒烟脚本复用 `-r/-v/-d`、`--grid-rows/--grid-cols`、`--token-tile/--model-tile`（tile `[T, W]`）和 `--build-only`；额外提供 `--src`、`--span-col`、`--subrect`，并用 `--rect-r0/r1/c0/c1` / `--rect-src` 圈定子矩形。

## 期望输出

成功时，每个 FFN 可执行文件会打印其位精确判定：

```text
[SUCCESS] 32-cell N-cut FFN GridPipe TREDUCE ReduceSum PASS.
[SUCCESS] 32-cell N-cut FFN GridPipe TPUSH ReduceSum PASS.
[SUCCESS] 32-cell N-cut FFN GridPipe TBROADCAST AllGather PASS.
[SUCCESS] 32-cell N-cut FFN GridPipe TPUSH AllGather PASS.
```

冒烟测试成功时打印：

```text
[SUCCESS] GridPipe single-source broadcast smoke PASS.
```
