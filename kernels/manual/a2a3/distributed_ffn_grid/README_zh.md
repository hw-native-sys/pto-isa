# Single-device Multi-block FFN GridPipe Demo

## 整体目标

本 demo 在 A2/A3 的单卡逻辑 FFN 网格上验证三个分布式 FFN GridPipe 集合通信接口 —— **TPUSH**、**TBROADCAST**、**TREDUCE**。host 在选定 device 上启动单进程，并 launch `gridRows * gridCols` 个 block；每个 block 对应一个逻辑 cell。共有 **四个例子**，每个对应一组 (接口, FFN 模式)，全部跑在同一套纯 1D N-cut 4×8 = 32-cell 拓扑上，并使用真实的 DeepSeek-v4 Pro 形状（M=T=8、H=7168、I=3072）：

| 例子（运行脚本 / 可执行文件） | 验证的接口 | FFN 模式 | 跨 cell 集合通信 |
| --- | --- | --- | --- |
| `run_tpush_reducesum.sh` / `distributed_ffn_grid_tpush_reducesum` | **TPUSH** | ReduceSum | 显式 `TPOP<Dir>` + `TADD` + `TPUSH<Dir>`（即 `TREDUCE` 的 A3 展开式，方向性中继） |
| `run_tpush_allgather.sh` / `distributed_ffn_grid_tpush_allgather` | **TPUSH** | AllGather | 最近邻 `TPUSH`/`TPOP` 中继 gather（fan-in-1 DAG） |
| `run_tbroadcast_allgather.sh` / `distributed_ffn_grid_tbroadcast_allgather` | **TBROADCAST** | AllGather | `TBROADCAST<GridGroup>` MPSC 组广播 |
| `run_treduce_reducesum.sh` / `distributed_ffn_grid_treduce_reducesum` | **TREDUCE** | ReduceSum | 融合 `TREDUCE<GridGroup, Sum>` 的 N→1 组扇入（`reduce_group_to_ubuf`） |

每个例子都用 `1e-3` 容差把 `[T, H]` 输出与 `golden.bin` 比对。四个例子在 NPU 上全部 **位精确通过**（`max diff = 0`，用 `-r npu` 运行）；详见 [位精确性说明](#位精确性说明)。

跨 cell 的集合通信走 A2/A3 GridPipe mock 后端：mock 中由 GM 撑起的本地 SRAM windows、fake `HcclDeviceContext` window 指针、ready/free 计数器、`dcci/dsb` fence 和自旋等待。该 demo 验证的是编程模型和同设备 mock 路径，不是多卡通信验证。

除这些 FFN 例子外，GridPipe 还支持路由 K-hop unicast（`TPUSH<dir, dist>` / `TPOP<dir, dist>`）与并发组广播（`TBROADCAST<GridGroup>` / `TPOP<GridGroup>`）。这两个能力各有一个独立的 Vec-only 冒烟测试，位于 `smoke/` 子目录（见 [GridPipe 冒烟测试](#gridpipe-冒烟测试)）。

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `README.md` / `README_zh.md` | 英文 / 中文说明文档。 |
| `CMakeLists.txt` | 构建四个 host 可执行文件及其 mixed Cube/Vec device kernel shared library（外加两个冒烟测试 target）。 |
| `run_treduce_reducesum.sh` / `run_tpush_reducesum.sh` | 配置 CANN、生成数据、配置 CMake、构建并运行 TREDUCE / TPUSH ReduceSum 例子。 |
| `run_tbroadcast_allgather.sh` / `run_tpush_allgather.sh` | 配置 CANN、生成数据、配置 CMake、构建并运行 TBROADCAST / TPUSH AllGather 例子。 |
| `ffn_config.hpp` | 编译期网格形状、tile 形状、GridPipe window 字节数、buffer 字节数、SwiGLU clamp 上下界、A3 精度映射表、Batcher GM arena 字节数，以及 lane stride 等常量。 |
| `kernel_launch.hpp` | host 侧 mixed kernel launch 接口声明（每个例子一份）。 |
| `main_treduce_reducesum.cpp` / `main_tpush_reducesum.cpp` | ReduceSum host driver：ACL 初始化、fake HCCL context / 本地 GridPipe windows、工作 buffer、Batcher 加载/分发、kernel launch、golden 比对、资源清理。 |
| `distributed_ffn_grid_treduce_reducesum_compute_kernel.cpp` | TREDUCE ReduceSum kernel：EAST+SOUTH 归约用融合的 `TREDUCE<GridGroup, Sum>` 组扇入（`reduce_group_to_ubuf`）。 |
| `distributed_ffn_grid_tpush_reducesum_compute_kernel.cpp` | TPUSH ReduceSum kernel：同样的计算，但 EAST+SOUTH 归约用显式 `TPOP<Dir>` + `TADD` + `TPUSH<Dir>` 拼出。 |
| `main_tbroadcast_allgather.cpp` / `main_tpush_allgather.cpp` | AllGather host driver。 |
| `distributed_ffn_grid_tbroadcast_allgather_compute_kernel.cpp` | TBROADCAST AllGather kernel：两个 gather 阶段用 `TBROADCAST<GridGroup>` + `TPOP<GridGroup>`。 |
| `distributed_ffn_grid_tpush_allgather_compute_kernel.cpp` | TPUSH AllGather kernel：两个 gather 阶段用双向 `TPUSH`/`TPOP` 中继。 |
| `batcher.hpp` | host 侧 **GM 模拟 Batcher**：在 GM 中持有全量输入 + 全量 DRAM 常驻权重，沿列切分成 per-cell shard，广播 x，并暴露输出收集区。 |
| `tpipe_tmov_inl.hpp` | 把 Cube↔Vec 的 C2V/V2C 搬运封装成方向化 `TMOV` 重载，内部转发到现有 `TPUSH`/`TPOP`，使 kernel 正文不再出现该 handshake。 |
| `gridpipe_payload_inl.hpp` | 本地 GridPipe payload 钩子与 fake-window 远端指针适配器 —— peer slot / 计分板字解析（`ResolvePeerSlotAddr`/`RemoteScbPtr`）、`copy_ubuf_to_neighbor_ubuf`/`copy_gm_to_ubuf` 的 tile 适配器（`CopyTileToNeighborSramSlot`/`CopyLocalSlotToTile`）、NoC 读本地性守卫（`PopSlotIsLocal`），以及 `TileUbPtr`（为取 raw UB 指针而非 tile 对象的 G4/G5 组 intrinsic 抽取 tile 的 `__ubuf__` 指针）。 |
| `smoke/` | GridPipe 特性冒烟测试。`khop_smoke_{config,kernel,launch}` + `main_khop_smoke.cpp` + `run_khop_smoke.sh` 覆盖路由 K-hop unicast；`bcast_smoke_*` + `run_bcast_smoke.sh` 覆盖单源行/列/子矩形广播。两者都通过父目录 `CMakeLists.txt` 构建。 |
| `../../../../include/pto/npu/a2a3/grid_cce_intrinsic.hpp` | V8 GridPipe CCE 门面层：三条握手 intrinsic `copy_ubuf_to_neighbor_ubuf`（G1 `COPY_UBUF_TO_NBR`）、`sync_hscb`（G2 `SYNC_HSCB`/`ST_HSCB`）、`wait_ipc_scb`/`wait_ipc_scb_sim`（G3 `WAIT_SPR`，读+阻塞合一条指令、无 `MOV_SPR2X` peek）——外加两条组语义 intrinsic `bcast_ubuf_to_group`（G4 `BCAST_UBUF_TO_GROUP`，单指令 1→N 扇出）与 `reduce_group_to_ubuf<Group,Op,T>`（G5 `REDUCE_GROUP_TO_UBUF`，N→1 逐元素扇入）。每条在 `PTO_GRID_CCE_NATIVE` 下 1:1 转发到 `__builtin_cce_*`，否则 emulate 同语义——G1–G3 用 GM 字 + cache 维护；G4–G5 用分块 UB/GM 拷贝，其中 G5 在 A3 mock 上还要用 per-member scratch 跑核内 `vadd`/`vmax`/`vmin` 归并。 |
| `../../../../include/pto/npu/a2a3/grid_intrinsic.hpp` | GridPipe A2/A3 数据模型 + mock 支持：Section 1 是 mesh 模型 + 邻居 / K-hop / group 解析器；Section 2 是 GM-mock 边界 fault 哨兵；Section 3 是 `GmSramArena` 地址段 SRAM 模型 + TPOP 读本地性守卫；此外定义了 ready/free lane 的缓存行步长 `kBcastLaneStride`（64 B，每 lane 独占一条 cache line）。 |
| `scripts/gen_data.py` | 生成 Batcher 消费的全量 fp16 X/weight 张量（`x_full`、`w_gate_full`、`w_up_full`、`w_down_full`）以及 fp32 SwiGLU `golden` 参考结果。四个例子统一用 `--pure-ncut` 产出扁平全量张量。 |
| `build/` | 被忽略的生成 build 目录。 |
| `out/` | 被忽略的生成数据目录。 |

## 位精确性说明

用 `-r npu` 运行（`sim`/`camodel` 模式会在 `aclrtSetDevice` 报 507033）；共享主机上每次运行都要走 `task-submit`。四个例子全部产出 `max diff = 0`（对 `golden.bin`）——位精确，而不只是落在 `1e-3` 容差内。曾经遮住这一点的是两个真实 bug，现都已修复：

- **缓存行门铃步长（TBROADCAST MPSC）。** `TBROADCAST<GridGroup>` 是真·同时 MPSC 通道——组内每个成员可**同一瞬间**调用，各自只敲自己的 per-source ready lane。这些 lane 原本被打包成连续 `u32`：`GroupMax` lane × 4 B ≤ 32 B < 一条 64 B cache line。于是多个生产者各自写**同一条 line** 的不同 word，而消费者每次轮询都 `dcci` 失效+读取；AICore store 是 line 粒度的，一个生产者的写回就会踩掉另一个的 word，那个门铃便**从 GM 永久丢失**（由绕过消费者 `dcci` 的 D2H lane dump 证伪/证实）。现象：偶发 `wait ready timeout`。修复：让每个 ready/free lane **独占一条 cache line**——`grid_intrinsic.hpp` 的 `kBcastLaneStride = 64` / `kBcastLaneStrideU32 = 16`，在 `ffn_config.hpp` 镜像为 `FFN_NCUT_LANE_STRIDE = 64`，并在 `GridTBroadcast.hpp` 的全部 4 个 lane 下标点应用。phase B/C 的握手从 10–20 s（重试风暴）降到 ~40 µs。
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

跨 cell 的 reduce 与 gather 保持显式的 GridPipe 调用；只有 block 内 Cube↔Vec 的 C2V/V2C 搬运被收敛到 `TMOV`。其中 ReduceSum 例子的 EAST/SOUTH 归约：`treduce` 用 `TREDUCE<GridGroup, Sum>` 组扇入（在 sink 处一次性 N→1 折叠），`tpush` 用 `TPOP<Dir>` + `TADD` + `TPUSH<Dir>` 方向性中继逐跳累加。

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

host 分配 `gridRows * gridCols` 个本地 SRAM windows（mock 中由 GM backing）。`TPUSH<EAST>` 通过 `ResolvePeerSlotAddr` 运行时 helper 解析 east neighbor 的 SRAM slot 后写入 payload，再发布 ready counter；`TPOP<EAST>` 等待本地 ready counter、读取本地 SRAM slot，并向 west neighbor 归还 free credit。

mock 使用 GM flag polling 和 cache maintenance 在 A2/A3 上模拟 LPU WSE 预期的 `SPR` / `WFE` 行为。

### NoC 只写不读的地址段 SRAM 模型（`GmSramArena`）

为了贴近真实硬件，mock 把未来硬件的"每核私有 SRAM"显式建模为一个 **GM 地址段（address segment）arena**：那块连续的 `gridRows*gridCols * FFN_GRID_WINDOW_BYTES` window 缓冲被切成等长的 per-core 段，于是第 `c` 段（即 `windowsIn[c]`）就是第 `c` 个核的私有 SRAM：

```text
段 c = [base + c*winSize, base + (c+1)*winSize)   // base == windowsIn[0]
```

`GmSramArena`（位于 `include/pto/npu/a2a3/grid_intrinsic.hpp`）持有 `{base, segBytes, numSegs}` 以及 `SegmentOf` / `InSegment` 判定函数；demo 在 device 侧从 fake `HcclDeviceContext` 的 window 表构造它（`SramArenaFromCtx`）。它是"某地址归哪个核所有"的唯一真相来源。

这样就把真实硅片的 NoC 约束显式化并**强制**起来：fabric 只能跨核**写**，不能跨核**读**。

- `TPUSH<dir>` 把 payload 写入**邻居核**的段——这是跨段写，正是 fabric 的行为。
- `TPOP<dir>` 只能 POP **本核自己**的段。`GRID_TRY_TPOP_IMPL` 在 payload 读取前先调用 `PopSlotIsLocal` 守卫；一旦发生跨段读，就写入 `kFaultPopNonLocal`（`0x205`，"pop non-local segment"）并放弃本次 pop，host 的 `CheckGridPipeFaults` 会报出来。

在 native 硬件上 `PopSlotIsLocal` 是恒为 `true` 的 no-op：TPOP 的读地址天然就是本地的，因为 fabric 根本没有远程读通路。这个守卫只是为了 A2/A3 mock——mock 用一块 GM window 模拟 SRAM，它物理上可以读任意地址，若没有守卫，demo 就可能悄悄依赖一次硅片做不到的远程读。每个 A2/A3 kernel 都会编入一条 `static_assert(GmSramArenaSelfCheck())`，因此段计算一旦回归就会在编译期失败，而不是把 pop 误路由出去。

> `pto::comm` 版本（`TREDUCE` / `TGATHER`）有意**不**遵守该约束：它们是 root 直接读取每个 rank 的 collective（HCCL/RDMA 式的远程读），与 WSE NoC 是不同的内存模型。只有 GridPipe `TPUSH`/`TPOP` 路径被约束成只写不读。

### IPC_SCB 计分板 intrinsic API

GridPipe 的 ready/free 同步走 V8 IPC_SCB 计分板路线。握手 intrinsic 位于 `include/pto/npu/a2a3/grid_cce_intrinsic.hpp` 的薄 CCE 门面层——每条门面在 `PTO_GRID_CCE_NATIVE` 下 1:1 转发到 `__builtin_cce_*`，否则在 A2/A3 mock 中用 GM 字 + cache 维护（`dcci`/`dsb`）emulate 同语义：

- `copy_ubuf_to_neighbor_ubuf(dstNeighborSlot, src, bytes)`（V8 `COPY_UBUF_TO_NBR`，G1——唯一新增机器指令 / HW-DEP-0）：把本核 UB payload 写入解析出的邻居 L1/SRAM slot。不自同步，data-ready 由随后的 `sync_hscb(READY)` 通告。
- `sync_hscb(peerScb, absCount)`（V8 `SYNC_HSCB`/`ST_HSCB`，G2——复用 HSCB store + 邻居 IPC_SCB 寻址 / HW-DEP-1）：把本核新的单调绝对计数 store 进对端的 `ready_scb`/`free_scb`（IPC_SCB）。`(kind, dir, dist)` 机器操作数由调用方的 `RemoteScbPtr` 运行时 helper 解析折进 `peerScb`，门面直接操作解析后的目标。
- `wait_ipc_scb(localScb, threshold, slot)`（V8 `WAIT_SPR`，G3——复用 IPC_SCB 阻塞等待）：读+阻塞合**一条**指令——入口读本核 IPC_SCB，已 `≥ threshold` 即放行，否则阻塞当前 pipe 至对端 `sync_hscb` store 唤醒。V8 去掉了 V7 的 `MOV_SPR2X` 非阻塞 peek，无单独读步。demo 实际调 `wait_ipc_scb_sim(..., maxSpins)` 这层 mock 包装——加自旋超时哨兵，使握手死锁能以 fault 暴露而非挂死测试；文档化的硬件接口仍是上面的 void `wait_ipc_scb`。

payload 的远端地址解析（把本地 slot / 计分板字解析为对端 GM window 中同字节偏移）是 demo `gridpipe_payload_inl.hpp` 中的普通运行时 helper（`ResolvePeerSlotAddr` / `RemoteScbPtr`），非 intrinsic。TPOP 的本地 drain 复用现成本地 `copy_gm_to_ubuf`——NoC 只写，故刻意**无跨核读** payload——并用 `GmSramArena` 段校验 `PopSlotIsLocal` 守卫，使误连的跨段读被拒绝而非悄悄服务。

native lowering 对接真实 CCE HSCB/IPC_SCB 栈（`__sync_hscb`/`__st_hscb`；阻塞等待用 `__builtin_cce___wait_ipc_scb`，或现成最贴近的 `__wait_ast_scb`——头文件尚未暴露 IPC_SCB 上的阻塞 `WAIT_SPR`）。当前 A2/A3 mock 用 GM 字 + cache 维护替代这些 IPC_SCB 计分板。native 硬件提供邻居 IPC_SCB 寻址（V8 HW-DEP-1）与 `COPY_UBUF_TO_NBR` builtin（V8 HW-DEP-0）后，上层 GridPipe 调用点不需要修改——打开 `PTO_GRID_CCE_NATIVE`，门面即路由到真实内建。

`TPUSH<EAST>` 先用 `wait_ipc_scb` 等待本核 `free_scb`，写 payload slot，再用 `sync_hscb` 把 `prod_idx` 发布到下游 `ready_scb`。`TPOP<EAST>` 先等待本核 `ready_scb`，读取 payload slot，再发布 `cons_idx` 到上游 `free_scb`。

### 组广播与归约 intrinsic（G4 / G5）

在三条握手门面（G1–G3）之上，同一个头文件还暴露两条**组数据搬运** intrinsic，各是某个 Tier-2 集合通信的单指令形态：

- `bcast_ubuf_to_group<Group>(groupSlotBase, src, bytes, memberCount, rect, memberStride)`（G4 `BCAST_UBUF_TO_GROUP`）：把一个本地 UB tile 拷进每个成员解析出的接收 slot——dtype 无关的字节拷贝（`void*` + bytes），不自同步（调用方仍要敲 `sync_hscb(READY)` 门铃）。`PTO_GRID_CCE_NATIVE` 下是一条 `__builtin_cce_bcast_ubuf_to_group`；A3 mock 下是分块 UB→GM-window 拷贝。
- `reduce_group_to_ubuf<Group, Op, T>(dst, groupSlotBase, bytes, memberCount, rect, memberStride, combineScratch)`（G5 `REDUCE_GROUP_TO_UBUF`）：把每个成员解析出的 contribution slot 用 `Op`（Sum/Max/Min）逐元素折叠进本地 UB。它必须知道元素位宽，所以按 `T` 模板化、按 `sizeof(T)` 派发 native builtin（half/bfloat16 走 `_b16`、float 走 `_b32`）。它按**升序**折叠成员（member 0 播种 `dst`），所以 SPMD 行/列扇入能逐位复现方向性中继的从左到右累加（IEEE-754 加法可交换）。A3 mock 下它把每个成员 GM→UB 拉进来，再用 per-member `combineScratch` 跑核内 `vadd`/`vmax`/`vmin`（mock 必需，native/`__CPU_SIM` 忽略）。

`GRID_TBROADCAST` 的 per-member 拷贝循环现在**坍缩成一条** `bcast_ubuf_to_group`——只要该组的接收 slot 构成等步长 arena：ROW 和 COL 永远如此（连续 rank → `memberStride = 解析出的 slot₁ − slot₀`）；SUBRECT 在单行/单列时等步长，多行矩形则回退到 per-member `copy_ubuf_to_neighbor_ubuf` 循环。`treduce` ReduceSum 的 EAST（行）与 SOUTH（列）阶段在 sink（`col == gridCols-1` / `row == gridRows-1`）处用 `GRID_TREDUCE_GROUP_IMPL<ROW/COL, Sum, float>` 扇入——这是真正的 N→1 扇入，与 `tpush` ReduceSum 例子用 `TPOP<Dir>` + `TADD` + `TPUSH<Dir>` 拼出的方向性中继是不同的集合通信**形状**。两条 Tier-2 门面都是 tile 无关的，所以 `gridpipe_payload_inl.hpp` 新增的 `TileUbPtr<T>` 负责抽取 tile 的 `__ubuf__` 指针交给 intrinsic。

### fp32 归约

归约 slot 携带 fp32 `[T, H]`，所以 `FFN_SLOT_BYTES = T * H * 4`。这让 `downPartial`、`yOutput` 和 `golden.bin` 都保持 fp32，host 可直接做容差比较。ReduceSum 的归约是按 H 分段的（`FFN_RS_REDUCE_SLOT_COUNT = kHSegs` = 7，每个 H 段一个 slot）：`treduce` 例子用 `GRID_TREDUCE_GROUP_IMPL` 在 sink 处折叠 segment-h 的 partial；`tpush` 例子用 `TPOP<EAST/SOUTH>` + `TADD` + `TPUSH<EAST/SOUTH>` 逐跳中继累加。在中继形式下 slot 数必须等于 `kHSegs`——跨段复用 slot 会在跨段 *free* 门铃上死锁。

### 路由 K-hop unicast

`TPUSH<dir, dist>` / `TPOP<dir, dist>` 把最近邻 pipe 扩展为沿 `dir` 方向 `dist` 跳的路由 unicast（Scheme A）。payload 写入解析到 `dist` 跳邻居段内的 slot，ready/free 计分板 store 经 `RemoteScbPtr` 解析到 `dist` 跳对端的 IPC_SCB、由 `sync_hscb` 发出，因此下游 `dist` 跳的接收方直接 pop，中间不需要任何中继。`dist == 1` 即原有最近邻行为。fan-in 保持为 1（每个方向/距离只有一个上游），无需 slot/flag 扩容。

### 并发组广播（TBROADCAST）

`TBROADCAST<GridGroup>`（首个模板参数为 `ROW` 或 `COL`）把本 cell 的 tile 一次性广播给所在行（`ROW`）或列（`COL`）的所有其它 cell：逐目标写入各接收方共享 ring 批量发出且目标之间无 fence，整个广播只付一次 publish fence，随后逐源 ready lane 批量触发。它不是按跳展开的 `TPUSH` 循环。

与旧的单源多播 `TPUSH<GridSpan>`（fan-in=1，禁止并发写者）不同，`TBROADCAST` 是真·同时 MPSC 通道（设计文档 `Grid_TPUSH_TPOP_WSE核间握手机制选型 §4 方案②·前缀偏移`）：组内每个成员可**同一瞬间**调用。每个源只写自己在每个接收方共享 ring 中的前缀偏移槽、只敲自己的 per-source ready lane，因此 K 个并发写者永不会踩同一个计数器。这正是"每个 AICORE 广播自己的分片"的 AllGather 得以正确的关键。接收方用 `TPOP<GridGroup>(pipe, tile, srcRank)` 取源 `srcRank` 的分片（按 srcRank 升序，使定向 free 通知链随消费推进）。per-member 的 payload 扇出本身在该组是等步长 arena 时坍缩成单条 `bcast_ubuf_to_group` intrinsic（见 [组广播与归约 intrinsic](#组广播与归约-intrinsicg4--g5)）。完整握手见 `Grid_TPUSH_TBROADCAST_TREDUCE_接口设计说明.md`。

### GridPipe 冒烟测试

上述两个能力在 `smoke/` 下各有一个 Vec-only 纯搬运冒烟测试（无 Cube、无 matmul、无数据文件，进程内校验，复用 FFN demo 的 GM-backed mock）：

- `khop_smoke`：`1 x cols` 行网格；每个 cell 向东 `DIST` 跳 push 一块带标记的 fp32 `[T, W]` tile，接收方 pop 后写出；host 校验 `out[c] == in[c-DIST]`。
- `bcast_smoke`：源 cell（`--src`）经 `TBROADCAST<GridGroup>` 向整行（`--span-col 1` 时为整列、`--subrect 1` 时为子矩形）广播带标记 tile；其余 cell 用 `TPOP<GridGroup>(pipe, tile, src)` 取回并写出；host 校验 `out[cell] == in[source]`。默认 1x5 行、源在 col 2，一次运行同时覆盖源两侧的接收方。

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
# 路由 K-hop unicast：1x4 行，整体右移 2 跳
bash smoke/run_khop_smoke.sh -r npu -v Ascend910B1 --device-id 0 --grid-cols 4 --dist 2
# 单源广播：1x5 行，源在 col 2（--span-col 1 + Rx1 网格切到列广播；--subrect 1 切到子矩形广播）
bash smoke/run_bcast_smoke.sh -r npu -v Ascend910B1 --device-id 0 --grid-cols 5 --src 2
```

两者均支持 `--build-only`，且不需要生成数据。

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

冒烟脚本复用 `-r/-v/-d`、`--grid-rows/--grid-cols`、`--token-tile/--model-tile`（tile `[T, W]`）和 `--build-only`；`run_khop_smoke.sh` 额外提供 `--dist`（跳数，默认 2）。`run_bcast_smoke.sh` 额外提供 `--src`（源下标，默认 2）、`--span-col`（1 为列组，默认 0 即行组）和 `--subrect`（1 为子矩形组，默认 0），并用 `--rect-r0/r1/c0/c1` / `--rect-src` 圈定子矩形。

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
[SUCCESS] GridPipe K-hop unicast smoke PASS.
[SUCCESS] GridPipe single-source broadcast smoke PASS.
```
