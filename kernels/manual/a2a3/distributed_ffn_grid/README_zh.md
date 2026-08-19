# Single-device Multi-block FFN GridPipe Demo

## 整体目标

本 demo 在 A2/A3 的单卡逻辑 FFN 网格上验证三个分布式 FFN GridPipe 集合通信接口 —— **TPUSH**、**TBROADCAST**、**TREDUCE**。host 在选定 device 上启动单进程，并 launch `gridRows * gridCols` 个 block；每个 block 对应一个逻辑 cell。共有 **四个例子**，每个对应一组 (接口, FFN 模式)，全部跑在同一套纯 1D N-cut 4×8 = 32-cell 拓扑上，并使用真实的 DeepSeek-v4 Pro 形状（M=T=8、H=7168、I=3072）：

| 例子（运行脚本 / 可执行文件） | 验证的接口 | FFN 模式 | 跨 cell 集合通信 |
| --- | --- | --- | --- |
| `run_tpush_reducesum.sh` / `distributed_ffn_grid_tpush_reducesum` | **TPUSH** | ReduceSum | 显式 `TPOP(pipe, ..., prodId)` + `TADD` + `TPUSH(pipe, ..., consId, isLastTransfer)` 中继 |
| `run_tpush_allgather.sh` / `distributed_ffn_grid_tpush_allgather` | **TPUSH** | AllGather | 最近邻 `TPUSH`/`TPOP` 中继 gather（fan-in-1 DAG） |
| `run_tbroadcast_allgather.sh` / `distributed_ffn_grid_tbroadcast_allgather` | **TBROADCAST** | AllGather | `TBROADCAST<GridGroup>` MPSC 组广播 |
| `run_treduce_reducesum.sh` / `distributed_ffn_grid_treduce_reducesum` | **TREDUCE** | ReduceSum | 融合 `TREDUCE<GridGroup, Sum>` 的 N→1 组扇入（`mov_ubuf_group`，op=SUM） |

每个例子都用 `1e-3` 容差把 `[T, H]` 输出与 `golden.bin` 比对。四个例子在 NPU 上全部 **位精确通过**（`max diff = 0`，用 `-r npu` 运行）；详见 [位精确性说明](#位精确性说明)。

跨 cell 的集合通信走 A2/A3 GridPipe mock 后端：mock 中由 GM 撑起的本地 SRAM windows、fake `HcclDeviceContext` window 指针、每 channel 的 ready/free/close 计数器、`dcci/dsb` fence 和自旋等待。该 demo 验证的是编程模型和同设备 mock 路径，不是多卡通信验证。

Unicast 在运行期显式传 peer id（TPUSH 的 `consId`、TPOP 的 `prodId`）；并发组广播仍使用 `TBROADCAST<GridGroup>` / `TPOP<GridGroup>`。独立的广播冒烟测试位于 `smoke/`。

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `README.md` / `README_zh.md` | 英文 / 中文说明文档。 |
| [`GRIDPIPE_HANDSHAKE_DESIGN_zh.md`](GRIDPIPE_HANDSHAKE_DESIGN_zh.md) | 当前 GridPipe 的建链、核间传输、CLOSE/tenant 交接、broadcast ticket、group-reduce pull 握手及状态机设计。 |
| `CMakeLists.txt` | 构建四个 host 可执行文件及其 mixed Cube/Vec device kernel shared library（外加三个冒烟测试 target）。 |
| `run_treduce_reducesum.sh` / `run_tpush_reducesum.sh` | 配置 CANN、生成数据、配置 CMake、构建并运行 TREDUCE / TPUSH ReduceSum 例子。 |
| `run_tbroadcast_allgather.sh` / `run_tpush_allgather.sh` | 配置 CANN、生成数据、配置 CMake、构建并运行 TBROADCAST / TPUSH AllGather 例子。 |
| `ffn_config.hpp` | 编译期网格形状、tile 形状、GridPipe window 字节数、buffer 字节数、SwiGLU clamp 上下界、A3 精度映射表、Batcher GM arena 字节数，以及记分板 cache line 步长等常量。 |
| `kernel_launch.hpp` | host 侧 mixed kernel launch 接口声明（每个例子一份）。 |
| `main_treduce_reducesum.cpp` / `main_tpush_reducesum.cpp` | ReduceSum host driver：ACL 初始化、fake HCCL context / 本地 GridPipe windows、工作 buffer、Batcher 加载/分发、kernel launch、golden 比对、资源清理。 |
| `distributed_ffn_grid_treduce_reducesum_compute_kernel.cpp` | TREDUCE ReduceSum kernel：EAST+SOUTH 归约用融合的 `TREDUCE<GridGroup, Sum>` 组扇入（`mov_ubuf_group`，op=SUM）。 |
| `distributed_ffn_grid_tpush_reducesum_compute_kernel.cpp` | TPUSH ReduceSum kernel：EAST 与 SOUTH 跨 launch 共用一个 channel，以显式 TPOP + TADD + TPUSH 覆盖 close 与接力计数重绑定。 |
| `main_tbroadcast_allgather.cpp` / `main_tpush_allgather.cpp` | AllGather host driver。 |
| `distributed_ffn_grid_tbroadcast_allgather_compute_kernel.cpp` | TBROADCAST AllGather kernel：两个 gather 阶段用 `TBROADCAST<GridGroup>` + `TPOP<GridGroup>`。 |
| `distributed_ffn_grid_tpush_allgather_compute_kernel.cpp` | TPUSH AllGather kernel：两个 gather 阶段用双向 `TPUSH`/`TPOP` 中继。 |
| `batcher.hpp` | host 侧 **GM 模拟 Batcher**：在 GM 中持有全量输入 + 全量 DRAM 常驻权重，沿列切分成 per-cell shard，广播 x，并暴露输出收集区。 |
| `tpipe_tmov_inl.hpp` | 把 Cube↔Vec 的 C2V/V2C 搬运封装成方向化 `TMOV` 重载，内部转发到现有 `TPUSH`/`TPOP`，使 kernel 正文不再出现该 handshake。 |
| `gridpipe_payload_inl.hpp` | 本地 GridPipe payload 钩子与 fake-window 适配器：peer slot/SCB 解析、tile 到 producer L1 staging、producer L1 到对端 ring 的拷贝、本地接收 ring drain，以及 TPOP 本地性守卫。 |
| `smoke/` | Vec-only GridPipe 冒烟测试：广播（`bcast_smoke_*` + `run_bcast_smoke.sh`）、单播交接（`unicast_smoke_*` + `run_unicast_smoke.sh`）、归约↔单播通道接力（`relay_smoke_*` + `run_relay_smoke.sh`），以及覆盖矩阵 `run_bcast_smoke_matrix.sh`（组形态 × 发布者数 × ring 深度 × 票据批量 × 通道交接）。 |
| `../../../../include/pto/npu/a2a3/grid_cce_intrinsic.hpp` | Grid CCE 门面：`copy_l1_to_peer_l1` / `copy_l1_to_group` 处理统一 L1 上的外发搬运，`sync_hscb` 发布对端计数，`wait_ipc_scb` 阻塞等本地计分板，`mov_ubuf_group` 处理组归约。A3 mock 用 GM window 表示 L1 地址段。 |
| `../../../../include/pto/npu/a2a3/grid_intrinsic.hpp` | GridPipe A2/A3 数据模型 + mock 支持：每 channel 的 ready/free/close SCB、生产者/消费者绑定、每消费者三态 FSM、持久化接力计数、mesh/group 解析器、fault 哨兵，以及 `GmSramArena` 的 TPOP 读本地性守卫。 |
| `scripts/gen_data.py` | 生成 Batcher 消费的全量 fp16 X/weight 张量（`x_full`、`w_gate_full`、`w_up_full`、`w_down_full`）以及 fp32 SwiGLU `golden` 参考结果。四个例子统一用 `--pure-ncut` 产出扁平全量张量。 |
| `build/` | 被忽略的生成 build 目录。 |
| `out/` | 被忽略的生成数据目录。 |

## 位精确性说明

用 `-r npu` 运行（`sim`/`camodel` 模式会在 `aclrtSetDevice` 报 507033）；共享主机上每次运行都要走 `task-submit`。四个例子全部产出 `max diff = 0`（对 `golden.bin`）——位精确，而不只是落在 `1e-3` 容差内。曾经遮住这一点的是两个真实 bug，现都已修复：

- **缓存行门铃步长（TBROADCAST MPSC）。** `TBROADCAST<GridGroup>` 是真·同时 MPSC 通道——组内每个成员可**同一瞬间**调用，各自只敲自己的门铃。这些门铃原本被打包成连续 `u32`：`GroupMax` 个 × 4 B ≤ 32 B < 一条 64 B cache line。于是多个生产者各自写**同一条 line** 的不同 word，而消费者每次轮询都 `dcci` 失效+读取；AICore store 是 line 粒度的，一个生产者的写回就会踩掉另一个的 word，那个门铃便**从 GM 永久丢失**（由绕过消费者 `dcci` 的 D2H dump 证实）。现象：偶发 `wait ready timeout`。修复：让每个有独立外部写者的记分板**独占一条 cache line**（`grid_intrinsic.hpp` 的 `kScbLineStride = 64`）。phase B/C 的握手从 10–20 s（重试风暴）降到 ~40 µs。现在那套专用 lane 数组已彻底移除——组集合通信直接敲 pipe **保留广播通道**的 ready 记分板（凭票原子加，通道数与组宽无关），而这些记分板从一开始就是一条 line 一个。
- **phase-D 输出 T 步长（两个 AllGather kernel）。** AllGather 的 y-shard `[T, Hc]` 写进**完整** `[T, H]` 输出，所以其行步长必须是完整输出宽度 `kHfull`（= `H` = 7168）。从 `hidden_full` store 复制粘贴时遗留成 `kIfull`（= `I` = 3072），把 y 的第 1–7 行打乱（≈50 % 零输出 / 大漂移）。两个 AllGather kernel 的 `GY` store 各改一行 `kIfull` → `kHfull` 即修复。

`treduce` ReduceSum 还要求其 per-cell partial buffer（`partialBuf` / `rowPartialBuf`）以**段主序（segment-major）**布局——每个 `[T, kHBase]` H 段在偏移 `h*(T*kHBase)` 处连续存放——这样组扇入才能把每个同行成员的段当作一段连续字节读出来；只有最终的 `yFull` 保留 strided `[T, H]` golden 布局。

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
control + record | unicast 接收 ring | 可选 组集合通信接收 ring | producer staging slot
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

- `copy_l1_to_peer_l1(dstPeerSlot, srcProducerSlot, transferScratch, bytes)`（G1 / HW-DEP-0）：把独立的本核 producer L1 slot 写入对端接收 ring。`transferScratch` 只是 A3 GM mock 的 DMA 中转 UB，不是体系结构源地址。搬运不自同步，随后由 `sync_hscb(READY)` 发布 data-ready。
- `sync_hscb(peerScb, absCount)`（V8 `SYNC_HSCB`/`ST_HSCB`，G2——复用 HSCB store + 邻居 IPC_SCB 寻址 / HW-DEP-1）：把绝对计数 store 进对端的 `ready_scb`、`free_scb` 或 `close_scb`；目标种类与 peer 已由 `RemoteScbPtr` 解析进 `peerScb`。
- `wait_ipc_scb(localScb, threshold, slot)`（V8 `WAIT_SPR`，G3——复用 IPC_SCB 阻塞等待）：读+阻塞合**一条**指令——入口读本核 IPC_SCB，已 `≥ threshold` 即放行，否则阻塞当前 pipe 至对端 `sync_hscb` store 唤醒。V8 去掉了 V7 的 `MOV_SPR2X` 非阻塞 peek，无单独读步。demo 实际调 `wait_ipc_scb_sim(..., maxSpins)` 这层 mock 包装——加自旋超时哨兵，使握手死锁能以 fault 暴露而非挂死测试；文档化的硬件接口仍是上面的 void `wait_ipc_scb`。

payload 目标地址解析（把本地接收 ring slot / 计分板字解析为对端 GM window 中同字节偏移）是 `gridpipe_payload_inl.hpp` 中的普通 helper（`ResolvePeerSlotAddr` / `RemoteScbPtr`），非 intrinsic；源地址则独立固定为 `producerSlotBase`。TPOP 只 drain 本核接收 ring，`PopSlotIsLocal` 会拒绝误连的跨段读。

native lowering 对接真实 CCE HSCB/IPC_SCB 栈。编译器的 copy builtin 仍保留历史 `ubuf` 命名，但门面传入的始终是独立的统一 L1 producer 地址，不再把调用者 tile 地址当作 NoC 源映射。A3 mock 用 GM + cache maintenance 表示 SCB 和两个 L1 区域。

`TPUSH(pipe, tile, consId)` 先等本核生产者 channel 的 `free_scb`，把 tile stage 到 `producerSlotBase`，从该 L1 区域拷到对端独立选择的消费者 channel ring，再发布 `prod_idx`。`TPOP(pipe, tile, prodId)` 等本核消费者 channel 的 `ready_scb`，只读该本地接收 ring，然后把 `cons_idx` 发布到对端已协商的生产者 channel 的 `free_scb`。两端 channel 下标不要求相同。

### 时分 MPSC 接力计数

生产者为每个消费者保存一条 FSM（`UNBOUND`、`ACTIVE`、`CLOSED`），并维护本核生产者 channel 状态（`UNBOUND`、`ACTIVE`、生产者 `CLOSED`）。`ACTIVE` 直接走常规 TPUSH 快路径；对 `UNBOUND` 或 `CLOSED` 消费者，生产者先在本地选择未使用或生产者 `CLOSED` 的 channel，没有资源就等待。随后把 `[mode，本地生产者 channel，生产者 block id]` 打进**一条 32 位 store**（因而 payload 与 commit 就是同一次写），投到消费者 bind-request 队列中**属于自己的那个槽**；生产者端资源不再由消费者分配。

**建链邮箱两端都是队列。** `kBindRequestOffset`（消费者窗口内）与 `kBindResponseOffset`（生产者窗口内）各是 `kGridBindQueueDepth` 条 cache line 的数组，**按对端逻辑 block id 索引**：request 槽 `p` 只由生产者 `p` 写，response 槽 `c` 只由消费者 `c` 写。于是多个生产者可以同一瞬间请求同一个消费者而互不覆盖——单行邮箱做不到这件事，这正是组集合通信过去只能"派生"通道而不能协商的原因。消费者**每趟只处理一条请求**，并轮转扫描起点以避免饿死，其余请求原封不动地留在队列里等下一趟；暂时无法服务的请求（还没有空闲通道、或同一生产者的旧流尚未排空）就保持 pending，而不是把消费者堵住。按 block id 而不是按队头索引是结构性的：A2/A3 没有 fetch-and-add（`atom_add_hscb` 不返回旧值），fabric 也没有远端读，所以根本没有一张"号码牌"可以发给请求者。对端 block id 超出队列深度报 `0x50B`；mesh 更大时调大 `PTO_GRID_BIND_QUEUE_DEPTH`。

消费者遍历自己的接收 channel：优先从未使用的 channel，否则**只**要求 `close_scb > closeBaseline`，即旧生产者已发布 CLOSE（不会再有新项）。**这就是全部条件：不要求 drain，既不要求退休生产者的、也不要求更早任何一位的。**新生产者拿到的 `prod_idx` 是该 channel 幸存的 ready 计数、其 `free_scb` 基线是本核当前 `cons_idx`，因此它第一次槽位复用判定（`free >= prod_idx + 1 - SlotCount`）与仍留在 ring 里的尾巴处在**同一条绝对计数**上，它覆盖未消费 payload 的可能性并不比退休的那个生产者更大。读者侧同样不需要记住退休生产者的任何信息——交接后 channel 仍是**一条连续的流**（一个 ring、一条绝对计数、写者依次接力），所以之后的 `TPOP` 做的还是老三样：按 `cons_idx` 读本地槽，把新的 `cons_idx` 写进**当前**持有者的 `free_scb`；这些字节由谁写入，从头到尾不进入这套算术。对调用方唯一的推论是：一旦接受了新生产者，剩下的尾巴就用**新**生产者的名字去 drain——生产者 id 的作用只是选中 channel，而退休者已不持有任何 channel。写成退休者不会写坏数据，只会等一个不会到来的绑定并超时。它**先清空 request 槽再作答**（生产者拿到答复后才会再次请求，所以这次清空绝不会抹掉下一轮的请求），把本核 `cons_idx` 写入生产者的 `free_scb[生产者 channel]`，再把 `[ready_scb 基线，消费者 channel + granted]` 写进生产者的 response 槽——基线在前、commit 在后。生产者只轮询那个 commit 字，随后把 ready 基线装入 `prod_idx`，并记录"本地生产者 channel ↔ 对端消费者 channel"映射。各计数均为绝对值覆盖，不是累加；换生产者时接力延续，不清 ring，也不复位 SCB。

两端**所有邮箱等待都会在等待间隙服务自己的请求队列**，因此一个正在等授权的核仍然在发放授权。否则两个核同时向对方请求（组集合通信的常态）就会互锁。

A2/A3 公共接口为 `TPUSH(pipe, tile, consId, isLastTransfer, events...)`；省略布尔参数等价于 `isLastTransfer == false`。只在当前生产者→消费者时段的最后一块上设为 `true`。最后一次 TPUSH 在 payload 与 READY 之后把同一绝对计数发布到对端消费者 channel 的 `close_scb`，再把该消费者 FSM 和本地生产者 channel 都置为 `CLOSED`。unicast 流本身仍是时分的——一条通道同一时刻只有一个写者——但**请求**不再要求调用方自己串行化：它们排队。

### 组广播与归约数据搬运

组 COPY 与 reduce 仍共用 native group opcode，但门面分开表达本地地址合约：

- `copy_l1_to_group(srcProducerSlot, groupSlot, transferScratch, ...)` 是广播路径，从独立 producer L1 扇出；A3 mock 按成员展开，native 仍发射一条 group COPY。
- `mov_ubuf_group(..., op=SUM/MAX/MIN, ...)` 是当前组归约路径，sink 按升序 block id 折叠成员贡献，保持与 relay 相同的累加顺序。

`GRID_TBROADCAST` 只 stage 一次：group arena 对 block id 仿射时调 `copy_l1_to_group`，否则按成员调 `copy_l1_to_peer_l1`。`GRID_TREDUCE_GROUP_IMPL` 的 SUM/MAX/MIN 仍下译到 `mov_ubuf_group`，它是真正的 N→1 扇入，与 peer-id TPOP + TADD + TPUSH 中继不同。

**归约的门铃是"拉"回来的，不是推过去的**（`2026-08-12-组归约门铃归属方案分析-全员调用与sink-only的五种形态.md` 的方案 C）。`mov_ubuf_group(op=SUM)` 本来就是 sink 去**拉**各成员的贡献；方案 C 把就绪标志也一并拉过来：

- 成员发布"我第 `r` 轮的贡献已就位"的方式，是把 `r+1` 存进自己窗口里的 epoch 字——一次普通的**本地** store，**零跨核动作**；
- sink 逐个拉取各成员的 epoch 字，直到全部 ≥ `r+1`，然后折叠；
- 折叠完成后 sink **推** `r+1` 到每个成员的 `free_scb`，这就是"你的贡献已被读走、可以回填"的释放信号。这些计数器只有 sink 一个写者，所以是普通的绝对值覆盖 store，不需要原子加（广播那边每个计数器有 K 个写者才需要）。

信用推到哪里，正是**建链邮箱**负责传达的：每个成员每次集合通信向 sink 绑定一次（`GridBindMode::GROUP_PULL`），告诉它该给哪个 `free_scb` 发信用，并拿回 sink 已折叠的轮数以便对齐基线。这些请求与其它请求一样排队，每趟处理一条。pull 绑定**不分配 ring**——它的 payload 走调用方自己的 arena——但它的**计数器**取自与 unicast 相同的那个通道池，二者在同一条 channel 上**接力复用**：

- **归约后单播复用（置 0 规则）**：退场的集合通信留下的 `cons_idx` 数的是"折叠轮数"而不是 tile，且它从未写过这条 channel 的 ring、也没敲过它的门铃，因此没有值得中继的基线：消费者把 `cons_idx` 与该 channel 自己的 ready/close 记分板一并置 0（`MOVX2SPR`；此刻该 channel 没有任何外部写者，正是这条指令要求的独占前提），再以 `prod_idx = 0` 应答，并把 0 写进生产者的 `free_scb`。
- **单播后归约复用（先 drain，再基线 + round）**：只有当退场的单播流已发布 CLOSE **且被完全 drain** 之后，channel 才会被授予——归约会彻底放弃这条 ring，残留将永远无人读取，而旧消费者仍欠一次 FREE store，那一发会落进集合通信的信用计数器里。之后什么都不清零：幸存的 `cons_idx` 就是本次集合通信的基线，sink 把它**两次**交给每个成员——既作为轮次原点，也作为其 `free_scb` 的起始值——两侧此后都按"基线 + round"计数。成员在发出请求之前还会清掉自己的 epoch 字：这第三个计数器无法靠 sink 事后声明的基线来对齐。

`smoke/relay_smoke_kernel.cpp` 在同一条 channel 上把这条序列跑两遍。phase 0–2 每个阶段各用一次 launch，覆盖兼容
回收路径；phase 3 则在一个 launch 内连续执行归约 → 单播 → 归约。每次归约的最后一轮，所有参与者都传入
`isLastRound = true`：成员等到 sink 的最后一笔信用后释放生产者侧 channel，sink 在发完全部最终信用后释放
消费者侧 channel。后继单播或归约因此可立即建链，无需等待下一次 `InitGridPipeFromWindow`。

代价在那份分析里已经写明，并非实现细节：`WAIT_SPR` 只能阻塞等待**本核**的 IPC_SCB，而本核不能写自己的 IPC_SCB，所以**拉**回来的标志无法挂起等待——sink 从"阻塞"变成了"自旋"。`HW-DEP-B`（见 `grid_cce_intrinsic.hpp` 的 `mov_peer_word_to_gpr`）就是能把阻塞还回来的 ISA 诉求：给组指令加一个标量 `ALL_GE`/`MIN` 模式，并允许结果落到可 `WAIT_SPR` 的 IPC_SCB。epoch 拉取卡住报 `0x303`。

归约仍由**每个成员共同调用**，而非只由 sink 调用：**sink 还在读时不许回填**这个等待属于成员，sink-only 的 API 无处安放；并且这与 `TBROADCAST` 以及所有同形状的集合通信 API（含 `MPI_Reduce`）一致——所有参与方调用同一个操作，由 root/sink 参数选择角色。成员半不搬运任何 payload：一次信用等待 + 一次本地 store，外加整个集合通信一次建链握手。也不再有"轮次"可排：没有 ready 门铃，就没有需要串行化的共享计数。

### fp32 归约

归约 slot 携带 fp32 `[T, H]`，所以 `FFN_SLOT_BYTES = T * H * 4`。这让 `downPartial`、`yOutput` 和 `golden.bin` 都保持 fp32，host 可直接做容差比较。ReduceSum 的归约按 H 分段（`FFN_RS_REDUCE_SLOT_COUNT = kHSegs` = 7，每个 H 段一个 slot）：`treduce` 在 sink 折叠 partial（每段一轮，因此成员从不等 free 信用）；`tpush` 用 peer-id TPOP + TADD + TPUSH 逐跳中继。`tpush` 变体的 EAST 与 SOUTH 两次 launch 刻意复用同一物理 channel；前一时段最后一个 H 段 close，后一 launch 从接力后的计数继续。`treduce` 变体给每相各配一个 window——它的两相在同一批 cell 上构成不同的组；协商式绑定虽然已经能正确地为复用通道重设基线，但一相一 window 能让两个集合通信的簿记彼此可见地分开。

### Peer-id unicast

Unicast 的方向由调度推导，不编码进指令类型。TPUSH 接收目标逻辑 block id（`consId`），TPOP 接收来源逻辑 block id（`prodId`），`RemoteScbPtr` 解析该 peer window 中对称的 slot/SCB 偏移。因此同一个 GridPipe 可以跨相位服务不同 peer。

### 并发组广播（TBROADCAST）

`TBROADCAST<GridGroup>`（首个模板参数为 `ROW` / `COL` / `SUBRECT`）把本 cell 的 tile 一次性广播给组内所有其它 cell：逐目标写入批量发出且目标之间无 fence，整个广播只付一次 publish fence，随后 ready 门铃批量触发。它不是按跳展开的 `TPUSH` 循环。

与旧的单源多播 `TPUSH<GridSpan>`（fan-in=1，禁止并发写者）不同，`TBROADCAST` 是真·同时 MPSC 通道：组内每个成员可**同一瞬间**调用。它把三件事拆开各自解决——地址、计数、信用：

- **地址——调用方给的全局序号。** `TBROADCAST(pipe, tile, basek, …)` 写入**保留广播通道**的 `basek % SlotCount` 槽。地址里不含任何身份，因此接收侧 ring 的深度由**它自己的 SRAM** 决定，而不再由写者总数决定——这正是 `2026-08-13-TBROADCAST写入地址由身份推导的可扩展性缺陷与前缀偏移改造方案.md`（判据 M2/M3）指出的缺陷：旧式 `(r*K + rank) % BcastSlotCount` 暗含 `BcastSlotCount ≥ K`。`basek` 是生产者侧的值，在每个接收方窗口内偏移相同，所以扇出仍是**一条** `copy_l1_to_group`（判据 M4）。调用方需保证它在一次集合通信内**唯一、递增且稠密**（`round*K + rank`，单源时就是 `round`）：稠密性正是各接收方无需通信即可推出同一授票顺序的依据。
- **计数——保留通道上的票据。** 通道 `[0, kGridBcastChanCount)`（默认仅 channel 0）保留给组播，其余留给 unicast 与组归约。发布者通过**组邮箱**向每个接收方申请——请求/应答各按**组内 rank** 一行、深度 `GroupMax`，而不是每核一行——每个持票发布者用一次 `ATOM_ADD_HSCB` 把该通道 ready 计数加 1。接收方看到计数达到 `ticketEnd = ticketBase + 发票数` 即知本批全部落地：这一条比较就让"同一记分板上的多个并发写者"精确可判。
- **信用——反方向的原子加。** 接收方每排空一块 tile 就向发布者 `free_scb` 原子加 1，发布者在开启新一轮前等 `基线 + round × (K-1)`。单个接收方最多贡献 `round`，因此和达到阈值当且仅当**每个**接收方都到位——在求和计数器上，任何更松的阈值都不成立（快的接收方会把慢的顶掉）。

**授票即写权限。** 接收方只会把"上一任住户已被本地 caller 排空"的槽授出去。这恰好补上了逐发布者信用计数器看不见的那一类：一旦 ring 比组窄，某个槽的上一任住户就属于**别人**，发布者自己的任何计数器都观察不到。所以顺序是：先申请，等所有接收方都同意后再写。

**为什么"持有部分票、等待其余票"不会死锁。** 每个接收方只在稠密 `basek` 序列的窗口 `[grantHead, grantHead + n)` 内发票（`n = PTO_GRID_BCAST_TICKET_BATCH`，被 pipe 钳到 `SlotCount`），且未到达的票不超过 `n` 张。设 `g` 为尚未完成的最小 `basek`：所有小于 `g` 的都已完成，因而在每个接收方处都已授票，故各接收方 `grantHead ≥ g`；若 `g` 在某处尚未授票，则该处 `grantHead == g`。该接收方发出的票必然都 `< g + n`，而 `< g` 的都已到达，故未到达的至多 `n-1` 张，配额不会挡住 `g`。于是 `g` 在**每个**接收方处要么已授票、要么立即可授——其发布者必然完成，窗口随之滑动。无需优先级协议、无需释放重试、接收方之间零通信。

因此 `n` 不只是吞吐旋钮，它就是集合通信的**并发度**：`n = 1` 严格按 `basek` 串行发布，`n = SlotCount` 让一整个 ring 的发布者同时发布（demo 即取此值）。

`SlotCount ≥ K` 时调用方没有任何顺序义务：

```cpp
TBROADCAST<Group>(pipe, tile, /*basek=*/myRank);
for (除自己外的每个 srcRank) { TPOP<Group>(pipe, tile, srcRank); }   // 顺序随意
```

组**比 `SlotCount` 宽**时，必须按 `SlotCount` 分波发布、波间排空：接收方在自己 caller 阻塞于 `TBROADCAST` 期间无法释放槽位。这是该模式本身的性质而非本实现的限制——ring 物理上只能容纳 `SlotCount` 块未排空的 tile——而调用方正是用 `basek` 的分配方式来表达它。违反时表现为阻塞（并在自旋上限处落成故障哨兵），不会静默写坏数据。bcast 冒烟两种都覆盖：`--slot-count` 小于组宽时其 kernel 走分波循环。

集合通信内部**每一次等待都在等待中服务**（每趟一次发票扫描），因此正阻塞在发布上的核仍在发票——这正是两个成员能互相授票的原因。代价是这些等待变成轮询而非 `WAIT_SPR` 挂起：挂起的核不再服务，而它等的东西可能正排在这份服务后面（`HW-DEP-C`：一条能在一组记分板上任意唤醒的 `WAIT_SPR`）。unicast TPUSH/TPOP 路径不受影响，仍然是阻塞等待。

排空顺序自由——`TPOP<Group>(src)` 等的是**那个源**，读的是该源自己 `basek` 指定的槽——所以 `0x405`、`0x50A` 继续退役。新增 `0x406`：请求携带的 `basek` 本接收方已经授过票，即调用方序号重复或回退。

**TBROADCAST 没有 `isLast` 或 CLOSE 操作。** 广播 channel 按 index 保留，票据在到达时即失效，写
CLOSE 只会与下一批的原子加撞同一个记分板。组 TREDUCE 同样是 pull 集合通信而不是流，但它可选的
`isLastRound` 用途不同：最后一次折叠后释放共享信用 channel，使单播或另一轮归约能在同一 launch 内
重新建链；它并不发布 CLOSE。`TPUSH` 保留 `isLastTransfer`，这是唯一用 CLOSE 标记单播生产者时段结束
的接口。

group arena 仿射时，payload 扇出用一条 `copy_l1_to_group`（见 [组广播与归约数据搬运](#组广播与归约数据搬运)）。

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
# 单个配置：单源广播，1x5 行，源在 col 2（--span-col 1 + Rx1 网格切到列广播；--subrect 1 切到子矩形广播）
bash smoke/run_bcast_smoke.sh -r npu -v Ascend910B1 --device-id 0 --grid-cols 5 --src 2

# 覆盖矩阵：每个 case 对应协议里不同的一处
bash smoke/run_bcast_smoke_matrix.sh --list          # 只打印计划，不编译
bash smoke/run_bcast_smoke_matrix.sh --from 7 --to 9 # 跑一段，例如两个 24 核 case
```

`run_bcast_smoke.sh`（以及 `run_unicast_smoke.sh`）跑一个编译期配置；`run_bcast_smoke_matrix.sh` 跑合起来覆盖该接口的那一组——
组形态（ROW / COL / SUBRECT，以及一个**严格位于 mesh 内部**的 SUBRECT，用来证明矩形外的 cell 确实是 no-op）、
单源 vs 全员同一瞬间发布、单轮 vs 多轮（槽位复用与生产者侧信用）、`SlotCount >= K` vs `< K` vs `== 1`
（无顺序义务 vs 分波 vs 每块 tile 都换手）、票据批量 `n = SlotCount` vs `n = 1`，以及规模：
**3x8 = 24 核全员并发**——这才是门铃的真实负载：一条保留通道的 ready 计数上有 24 个并发原子加。
每个 case 都是一次完整重编译，所以用 `--from/--to` 跑一段以适配有时限的任务槽。三个脚本都支持 `--build-only`，无需生成数据。

最后两个 case 是**单播时分交接**（`run_unicast_smoke.sh`），也是本仓唯一能跑到它的测试：两个生产者在**同一条**消费者 channel 上轮换，
而后一个接手时前一个的 tile **尚未被 drain**。接力棒（`A -> B`，走 A 重新打开的生产者 channel）把"先 A 后 B"变成程序性质而非调度性质；
消费者唯一的 drain 名字是 B——这正是 channel 易主时 `cons_idx` 仍为 0 的原因。回绕变体（`--slot-count 2`）进一步让 B 的第一块 tile
正好落在 A 的第一个槽上，于是 B 必须先等它建链时拿到的信用基线才能写：这是接力计数的 payload 安全那一半。
两者都是**有牙齿**的回归测试——把旧的"必须 drain 完才能重绑"门槛加回去，它们会以消费者 `0x504`、生产者 B `0x505` 失败，
正是那道门槛在此处造成的死锁。

`run_relay_smoke.sh` 覆盖另一类交接：组**归约**与单播流在同一条 channel 上轮换——二者现在共用一个通道池，而不再各占一个 index。
一条共享池 channel、一个 window、三次 launch（归约 → 单播 → 归约），两条规则各跑一次；同时还有一条并发的旁路流，
让第二个成员带着一个**与折叠基线不相等**的信用残值进入 phase 2。它同样**有牙齿**：关掉"归约后单播复用"的置 0 规则，
phase 1 会以消费者 `0x301` 失败（它在等一个新生产者永远到不了的 ready 计数）；去掉 sink 声明成员 free 基线的那一发，
phase 2 会以两个成员 `0x302`、sink `0x303` 失败——成员在等永远不会来的信用，sink 在等他们因此没能发布的 epoch。

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
