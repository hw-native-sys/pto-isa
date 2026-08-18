# Single-device Multi-block FFN GridPipe Demo

## 整体目标

本 demo 在 A2/A3 的单卡逻辑 FFN 网格上验证三个分布式 FFN GridPipe 集合通信接口 —— **TPUSH**、**TBROADCAST**、**TREDUCE**。host 在选定 device 上启动单进程，并 launch `gridRows * gridCols` 个 block；每个 block 对应一个逻辑 cell。共有 **四个例子**，每个对应一组 (接口, FFN 模式)，全部跑在同一套纯 1D N-cut 4×8 = 32-cell 拓扑上，并使用真实的 DeepSeek-v4 Pro 形状（M=T=8、H=7168、I=3072）：

| 例子（运行脚本 / 可执行文件） | 验证的接口 | FFN 模式 | 跨 cell 集合通信 |
| --- | --- | --- | --- |
| `run_tpush_reducesum.sh` / `distributed_ffn_grid_tpush_reducesum` | **TPUSH** | ReduceSum | 显式 `TPOP<Dir>` + `TADD` + `TPUSH<Dir>`（即 `TREDUCE` 的 A3 展开式，方向性中继） |
| `run_tpush_allgather.sh` / `distributed_ffn_grid_tpush_allgather` | **TPUSH** | AllGather | 最近邻 `TPUSH`/`TPOP` 中继 gather（fan-in-1 DAG） |
| `run_tbroadcast_allgather.sh` / `distributed_ffn_grid_tbroadcast_allgather` | **TBROADCAST** | AllGather | `TBROADCAST<GridGroup>` 串行组广播 |
| `run_treduce_reducesum.sh` / `distributed_ffn_grid_treduce_reducesum` | **TREDUCE** | ReduceSum | 融合 `TREDUCE<GridGroup, Sum>` 的 N→1 组扇入（`mov_ubuf_group`，op=SUM） |

每个例子都用 `1e-3` 容差把 `[T, H]` 输出与 `golden.bin` 比对。四个例子在 NPU 上全部 **位精确通过**（`max diff = 0`，用 `-r npu` 运行）；详见 [位精确性说明](#位精确性说明)。

跨 cell 的集合通信走 A2/A3 GridPipe mock 后端：mock 中由 GM 撑起的本地 SRAM windows、fake `HcclDeviceContext` window 指针、ready/free 计数器、`dcci/dsb` fence 和自旋等待。该 demo 验证的是编程模型和同设备 mock 路径，不是多卡通信验证。

除这些 FFN 例子外，GridPipe 还支持组广播（`TBROADCAST<GridGroup>` / `TPOP<GridGroup>`），它有一个独立的 Vec-only 冒烟测试，位于 `smoke/` 子目录（见 [GridPipe 冒烟测试](#gridpipe-冒烟测试)）。

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `README.md` / `README_zh.md` | 英文 / 中文说明文档。 |
| `CMakeLists.txt` | 构建四个 host 可执行文件及其 mixed Cube/Vec device kernel shared library（外加两个冒烟测试 target）。 |
| `run_treduce_reducesum.sh` / `run_tpush_reducesum.sh` | 配置 CANN、生成数据、配置 CMake、构建并运行 TREDUCE / TPUSH ReduceSum 例子。 |
| `run_tbroadcast_allgather.sh` / `run_tpush_allgather.sh` | 配置 CANN、生成数据、配置 CMake、构建并运行 TBROADCAST / TPUSH AllGather 例子。 |
| `ffn_config.hpp` | 编译期网格形状、tile 形状、GridPipe window 字节数、buffer 字节数、SwiGLU clamp 上下界、A3 精度映射表、Batcher GM arena 字节数，以及 scoreboard 缓存行步长等常量。 |
| `kernel_launch.hpp` | host 侧 mixed kernel launch 接口声明（每个例子一份）。 |
| `main_treduce_reducesum.cpp` / `main_tpush_reducesum.cpp` | ReduceSum host driver：ACL 初始化、fake HCCL context / 本地 GridPipe windows、工作 buffer、Batcher 加载/分发、kernel launch、golden 比对、资源清理。 |
| `distributed_ffn_grid_treduce_reducesum_compute_kernel.cpp` | TREDUCE ReduceSum kernel：EAST+SOUTH 归约用融合的 `TREDUCE<GridGroup, Sum>` 组扇入（`mov_ubuf_group`，op=SUM）。 |
| `distributed_ffn_grid_tpush_reducesum_compute_kernel.cpp` | TPUSH ReduceSum kernel：同样的计算，但 EAST+SOUTH 归约用显式 `TPOP<Dir>` + `TADD` + `TPUSH<Dir>` 拼出。 |
| `main_tbroadcast_allgather.cpp` / `main_tpush_allgather.cpp` | AllGather host driver。 |
| `distributed_ffn_grid_tbroadcast_allgather_compute_kernel.cpp` | TBROADCAST AllGather kernel：两个 gather 阶段用 `TBROADCAST<GridGroup>` + `TPOP<GridGroup>`。 |
| `distributed_ffn_grid_tpush_allgather_compute_kernel.cpp` | TPUSH AllGather kernel：两个 gather 阶段用双向 `TPUSH`/`TPOP` 中继。 |
| `batcher.hpp` | host 侧 **GM 模拟 Batcher**：在 GM 中持有全量输入 + 全量 DRAM 常驻权重，沿列切分成 per-cell shard，广播 x，并暴露输出收集区。 |
| `tpipe_tmov_inl.hpp` | 把 Cube↔Vec 的 C2V/V2C 搬运封装成方向化 `TMOV` 重载，内部转发到现有 `TPUSH`/`TPOP`，使 kernel 正文不再出现该 handshake。 |
| `gridpipe_payload_inl.hpp` | 本地 GridPipe payload 钩子与 fake-window 远端指针适配器 —— peer slot / 计分板字解析（`ResolvePeerSlotAddr`/`RemoteScbPtr`）、`copy_ubuf_to_neighbor_ubuf`/`copy_gm_to_ubuf` 的 tile 适配器（`CopyTileToNeighborSramSlot`/`CopyLocalSlotToTile`）、NoC 读本地性守卫（`PopSlotIsLocal`），以及 `TileUbPtr`（为取 raw UB 指针而非 tile 对象的 G4 组 intrinsic `mov_ubuf_group` 抽取 tile 的 `__ubuf__` 指针）。 |
| `smoke/` | GridPipe 特性冒烟测试。`bcast_smoke_*` + `run_bcast_smoke.sh` 覆盖单源与全源（`--all-src 1`）的行/列广播，通过父目录 `CMakeLists.txt` 构建。 |
| `../../../../include/pto/npu/a2a3/grid_cce_intrinsic.hpp` | V8 GridPipe CCE 门面层：三条握手 intrinsic `copy_ubuf_to_neighbor_ubuf`（G1 `COPY_UBUF_TO_NBR`）、`sync_hscb`（G2 `SYNC_HSCB`/`ST_HSCB`）、`wait_ipc_scb`/`wait_ipc_scb_sim`（G3 `WAIT_SPR`，读+阻塞合一条指令、无 `MOV_SPR2X` peek）——外加一条统一的组搬运 intrinsic `mov_ubuf_group`（G4 `MOV_UBUF_GROUP`，无模板、全运行期参数）。站在核的角度，广播与归约是*同一个动作*——在本核 UB tile 与已解析的 group 竞技场之间搬一块数据——故 NoC 通信模式做成运行期 `GridCollOp` operand（`COPY` = 1→N 复制扇出，`SUM`/`MAX`/`MIN` = N→1 逐元素扇入；数据通路方向由 `op` 隐含），而非第二条指令。每条在 `PTO_GRID_CCE_NATIVE` 下 1:1 转发到 `__builtin_cce_*`，否则 emulate 同语义——G1–G3 用 GM 字 + cache 维护；G4 用分块 UB/GM 拷贝，其中归约（`op != COPY`）在 A3 mock 上还要用 per-member scratch 跑核内 `vadd`/`vmax`/`vmin` 归并。（此合并把上一版的 `bcast_ubuf_to_group` G4 `BCAST_UBUF_TO_GROUP` + `reduce_group_to_ubuf<Group,Op,T>` G5 `REDUCE_GROUP_TO_UBUF`——两条机器指令、两组模板门面——收敛成一条；机器指令增量由 "+2 +复用 2" 降为 "+1 +复用 2"。设计文档：`2026-07-24-bcast-reduce-合并mov_ubuf_group方案.md`。） |
| `../../../../include/pto/npu/a2a3/grid_intrinsic.hpp` | GridPipe A2/A3 数据模型 + mock 支持：Section 1 是 mesh 模型 + 最近邻 / group 解析器；Section 2 是 GM-mock 边界 fault 哨兵；Section 3 是 `GmSramArena` 地址段 SRAM 模型 + TPOP 读本地性守卫；此外定义了 scoreboard 的缓存行步长 `kScbLineStride`（64 B，每个 scoreboard 独占一条 cache line）与组通道的 IPC_SCB 槽号 `kGroupReadyScbSlot` / `kGroupFreeScbSlot`。 |
| `scripts/gen_data.py` | 生成 Batcher 消费的全量 fp16 X/weight 张量（`x_full`、`w_gate_full`、`w_up_full`、`w_down_full`）以及 fp32 SwiGLU `golden` 参考结果。四个例子统一用 `--pure-ncut` 产出扁平全量张量。 |
| `build/` | 被忽略的生成 build 目录。 |
| `out/` | 被忽略的生成数据目录。 |

## 位精确性说明

用 `-r npu` 运行（`sim`/`camodel` 模式会在 `aclrtSetDevice` 报 507033）；共享主机上每次运行都要走 `task-submit`。四个例子全部产出 `max diff = 0`（对 `golden.bin`）——位精确，而不只是落在 `1e-3` 容差内。曾经遮住这一点的是两个真实 bug，现都已修复：

- **缓存行门铃步长。** 门铃字原本被打包成连续 `u32`，几个挤在一条 64 B cache line 里。两个不同的核写**同一条 line** 的两个 word 会互相丢更新：AICore store 是 line 粒度的，一个核的写回会用它自己那份（可能已过时的）副本盖掉邻居的 word，那个门铃便**从 GM 永久丢失**（由绕过消费者 `dcci` 的 D2H dump 证实）。现象：偶发 `wait ready timeout`。修复：让每个**有独立外部写者**的 scoreboard **独占一条 cache line**——`grid_intrinsic.hpp` 的 `kScbLineStride = 64` / `kScbLineStrideU32 = 16`，在 `ffn_config.hpp` 镜像为 `FFN_NCUT_SCB_LINE_STRIDE = 64`。phase B/C 的握手从 10–20 s（重试风暴）降到 ~40 µs。这个坑最早踩在 TBROADCAST 的 per-source lane 上（那些 lane 已退役，见下文），但规则比它们活得久，如今覆盖一个 window 里全部 8 个 scoreboard。
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

### PTO 指令面

一个 `GridPipe` 持有本 cell **全部**通道的 FIFO 状态——环按**方向**索引（`slotBase[dir]`、`prodIndex[dir]`、`consIndex[dir]`、`pushWindow[dir]`、`popWindow[dir]`，5 项），记分板按**网格边**索引（`readyScb[edge]`、`freeScb[edge]`，4 项），并且**不绑定任何对端身份**。组集合通信不持有任何自己的状态：它复用同一批方向环与同一批边记分板，因此既没有广播专用 ring，也没有组轮次计数器。

```cpp
GridPipe<Tile, SlotStride, SlotCount /*, DirMask = kGridDirAll */>

TPUSH<Dir>(pipe, tile)              TPOP<Dir>(pipe, tile)
TREDUCE<Dir, Op>(pipe, acc, recv)
TBROADCAST<Group>(pipe, tile)       TPOP<Group>(pipe, tile, srcBlockId)
TBWAIT<Group>(pipe)                 TBNOTIFY<Group>(pipe, dstBlockId)
TREDUCE<Group, Op, T>(pipe, acc, scratch, base, bytes, memberCount, sinkBlockId, memberStride)
```

这套接口里**凡是指代某个核的操作数，一律是逻辑 block id**——即 `get_block_idx()` 返回的那个整数 `row * gridCols + col`——而不是"组内第几个"：`srcBlockId` 指广播源，`dstBlockId` 指接过发布轮转的成员，`sinkBlockId` 指扇入的收集者。这里没有多卡 rank 的概念，只有一次 launch 内的 block。内核若想按组内位置遍历，用 `GroupMemberBlockId(Group, coord, shape, indexInGroup)` 换算（`IndexInGroup` 是其逆），传入组外的 block 会在**每个**成员上报 `0x405 kFaultGroupBadPeer`，而不是悄悄解析到一个陌生核。

上面每一条的通知信号量都是一对 **IPC_SCB scoreboard**——`ready_scb`（对端 store 的单调绝对计数，本地 `WAIT_SPR` 比较）与 `free_scb`（反向的信用）。全窗口**只有八个 scoreboard**，每条网格边一对——`GridDirection` 有五个枚举值，但只有四个是真实的边；`SOURCE` 是 GM/host/runtime 注入用的伪方向，**不占记分板**（`TPOP<SOURCE>` 把这一对解析为空指针，于是它的 ready 等待与 free 回写都由 runtime 带外把关）。环用 `GridDirectionIndex` 索引、记分板用 `GridEdgeIndex` 索引：组集合通信**不新增任何 scoreboard**，而是复用方向那一对，按坐标差的**主轴**把每条 (生产者, 消费者) 边归属到一个方向——`|dCol| > |dRow|` 时按消费者在东/西侧取 `EAST`/`WEST`，否则（**含相等**）按南/北取 `SOUTH`/`NORTH`——即 `GroupFlowDirection`。该方向命名的是**流向**，所以消费者在东侧的源写对方的 `ready_scb[EAST]`、对方也等自己的 `ready_scb[EAST]`，与 `TPUSH<EAST>`/`TPOP<EAST>` 的约定完全一致，只是跨了多跳、还可能不在同一轴上。没有任何一条走**按 rank 静态划分的 lane 数组**或组私有 scoreboard——这正是组集合通信必须把发布方串行化、而不能并发的原因（见下文）。

组集合通信的通知用的是 store 的**自增形式** `sync_hscb_add(peerScb, 1)`，而不是 `TPUSH`/`TPOP` 用的绝对写 `sync_hscb(peerScb, count)`（V8 §2b）。正是它才让一个**在不同时刻有多个写者**的 scoreboard 成为可能：绝对计数逼着所有写者对同一条共享序列达成一致，于是必须把参与者集合编码进数值——而只要参与者不是整组，这个前提立刻破产（单源广播算出的值与它那唯一接收方等待的值对不上）。自增可交换、且不携带任何关于"别人会写什么"的假设：源给每个接收方的 `ready_scb` 加 1，接收方给源的 `free_scb` 加 1，两侧的门限都退化成普通的 `TPUSH`/`TPOP` 算术。它同时废掉了旧的"**后继门铃最后发**"纪律：HSCB 只保证**同一对 (生产者, 消费者)** 之间的顺序，本核发往两个不同对端的 store 彼此无序——绝对计数下一个迟到的 store 会把共享计数**往回拉**，而自增无论以什么顺序落地都不会。

> **HW-DEP。** 自增形式的 HSCB 是对硬件的诉求，而且这个自增必须在 scoreboard 处**原子**：与绝对写不同，它的写者是真的会重叠的——一次广播的 K 个接收方会在同一瞬间给同一个生产者的 `free_scb` 记信用。因此 A2/A3 mock **不是**用读-改-写来模拟它，而是用后端最小的原子累加 DMA（`set_atomic_s32` + `set_atomic_add` + 4 字节 UB→GM 突发，加数放在 UB 顶部预留的一个字里）：读-改-写会丢信用（实测发出 10 次加、计数停在 9）。mock 还把自增**收敛到 sub-block 0**，因为一个 cell 是一个 block，但 mix 模式下一个 block 的**两个 AIV sub-block** 都会跑同一段 vector 程序——绝对写在这种重复执行下是幂等的，自增不是（每个计数都会翻倍）。

由于组借用了方向 scoreboard，**同一根 pipe 上归属到同一方向的组集合通信与单播通道会互相踩计数**——请分开用不同的 pipe。示例里的集合通信 pipe 都是 `DirMask = kGridDirNone`，没有别的东西碰它们的 scoreboard；而 scoreboard 本身与 `DirMask` 无关、恒被接好，这正是无 ring 的 pipe 也能用的原因。

`Dir` 命名的是**网格的一条边，而不是某个核**：`TPUSH<Dir>` 的目标是沿 `Dir` 的相邻 cell，`TPOP<Dir>` 所排空的生产者是沿 `-Dir` 的相邻 cell，两者都在调用点由 `(Dir, coord, shape)` 现算。整个指令族**没有跳数参数**——每次 grid 传输恰好一跳；更远的传递就是中继：一条边一次 `TPUSH`，于是每条边各自持有自己的信用与反压。因为类型里不含对端，同一个 pipe 在相位切换、对端换核之后仍然继续服务同一个方向。

四个 demo 与冒烟测试**一律调用上面这层 PTO 指令**，而不是 A2/A3 后端的 `GRID_*_IMPL` / `GRID_TRY_*_IMPL` 入口，这样跑一遍就同时验证了指令面本身（重载选择、`SOURCE` 的 `static_assert`、非 A2A3 profile 的 target-profile 拦截），而不只是下译逻辑。代价是指令不带自旋上限——它们像硬件 `WAIT_SPR` 一样一直阻塞——所以握手接错会表现为挂死，而不是 `kFaultWaitReadyTimeout` 哨兵。这在本 demo 里是安全的：host 把每个中继组／group 整体排进同一个 wave（见 `main_*.cpp` 的 `LaunchWave`）。调试时若要恢复超时哨兵，直接改调对应的 `GRID_TRY_*_IMPL(..., maxSpins)`。

`TREDUCE` 有两个重载，对应两种**不同形状**的归约：`TREDUCE<Dir, Op>(pipe, acc, recv)` 是逐跳中继（第一个模板参数是 `GridDirection`），`TREDUCE<Group, Op, T>(pipe, acc, scratch, base, bytes, memberCount, sinkBlockId, memberStride)` 是 N→1 组扇入（第一个模板参数是 `GridGroup`，落到单条 `mov_ubuf_group`）。扇入直接读贡献 arena，所以它不用环——但它要 pipe：告诉 sink"N 份贡献都已落地"的握手、以及告诉贡献者"sink 已取走"的信用，用的正是这条 pipe 的边记分板。两者互不干扰：各自的首个模板实参代入对方模板时都会推导失败而被丢弃。

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
- `sync_hscb(peerScb, absCount)`（V8 `SYNC_HSCB`/`ST_HSCB`，G2——复用 HSCB store + 邻居 IPC_SCB 寻址 / HW-DEP-1）：把本核新的单调绝对计数 store 进对端的 `ready_scb`/`free_scb`（IPC_SCB）。`(kind, dir)` 机器操作数由调用方的 `RemoteScbPtr` 运行时 helper 解析折进 `peerScb`，门面直接操作解析后的目标。
- `wait_ipc_scb(localScb, threshold, slot)`（V8 `WAIT_SPR`，G3——复用 IPC_SCB 阻塞等待）：读+阻塞合**一条**指令——入口读本核 IPC_SCB，已 `≥ threshold` 即放行，否则阻塞当前 pipe 至对端 `sync_hscb` store 唤醒。V8 去掉了 V7 的 `MOV_SPR2X` 非阻塞 peek，无单独读步。GridPipe 的握手序列走的是 `wait_ipc_scb_sim(..., maxSpins)` 这层 mock 包装（`maxSpins > 0` 时加自旋超时哨兵，使握手死锁能以 fault 暴露而非挂死测试）；PTO 指令下译时传 `maxSpins = 0`，即和硬件 `WAIT_SPR` 一样永久阻塞，只有直接调 `GRID_TRY_*_IMPL` 才拿得到超时哨兵。文档化的硬件接口仍是上面的 void `wait_ipc_scb`。

payload 的远端地址解析（把本地 slot / 计分板字解析为对端 GM window 中同字节偏移）是 demo `gridpipe_payload_inl.hpp` 中的普通运行时 helper（`ResolvePeerSlotAddr` / `RemoteScbPtr`），非 intrinsic。TPOP 的本地 drain 复用现成本地 `copy_gm_to_ubuf`——NoC 只写，故刻意**无跨核读** payload——并用 `GmSramArena` 段校验 `PopSlotIsLocal` 守卫，使误连的跨段读被拒绝而非悄悄服务。

native lowering 对接真实 CCE HSCB/IPC_SCB 栈（`__sync_hscb`/`__st_hscb`；阻塞等待用 `__builtin_cce___wait_ipc_scb`，或现成最贴近的 `__wait_ast_scb`——头文件尚未暴露 IPC_SCB 上的阻塞 `WAIT_SPR`）。当前 A2/A3 mock 用 GM 字 + cache 维护替代这些 IPC_SCB 计分板。native 硬件提供邻居 IPC_SCB 寻址（V8 HW-DEP-1）与 `COPY_UBUF_TO_NBR` builtin（V8 HW-DEP-0）后，上层 GridPipe 调用点不需要修改——打开 `PTO_GRID_CCE_NATIVE`，门面即路由到真实内建。

`TPUSH<EAST>` 先用 `wait_ipc_scb` 等待本核 `free_scb`，写 payload slot，再用 `sync_hscb` 把 `prod_idx` 发布到下游 `ready_scb`。`TPOP<EAST>` 先等待本核 `ready_scb`，读取 payload slot，再发布 `cons_idx` 到上游 `free_scb`。

### 组广播与归约 intrinsic（G4 / MOV_UBUF_GROUP）

在三条握手门面（G1–G3）之上，同一个头文件还暴露一条**组数据搬运** intrinsic `mov_ubuf_group`——广播与归约两种 Tier-2 集合通信的统一单指令形态。站在发起核的角度，二者是同一个动作——在本核 UB tile 与已解析的每成员 group 竞技场之间搬一块 `bytes` 字节的数据——区分它们的只是 NoC 通信模式，且该模式是**运行期** operand，而非第二条指令。2026-07-23 的设计曾把二者下沉成 `bcast_ubuf_to_group<Group>`（G4 `BCAST_UBUF_TO_GROUP`）+ `reduce_group_to_ubuf<Group,Op,T>`（G5 `REDUCE_GROUP_TO_UBUF`）一对；2026-07-24 的合并把二者收敛成一条机器指令、一个无模板门面（`mov_ubuf_group`，机器指令增量由 "+2 +复用 2" 降为 "+1 +复用 2"；设计文档 `2026-07-24-bcast-reduce-合并mov_ubuf_group方案.md`）：

- `mov_ubuf_group(ubTile, groupSlotBase, bytes, memberCount, memberStride, op, eltype, rect, combineScratch, groupDesc)`（G4 `MOV_UBUF_GROUP`，**无模板**——`Group`/`Op`/`T` 模板参数都没了；`Group` 由 Tier-2 调用方解析进 `groupSlotBase`/`memberCount` 或 `groupDesc`）。`op` 是运行期 `GridCollOp`，选 NoC 通信模式**并**隐含数据通路方向：
  - `op == COPY`——广播：本核为**源**，把 UB tile 复制到每个成员 slot（1→N 扇出，push）。它是字节级纯拷贝（不读元素值），故 `eltype` 被忽略——正是旧的 `bcast_ubuf_to_group`。
  - `op == SUM/MAX/MIN`——归约：本核为**汇**，读每个成员的贡献 slot，按 op 逐元素折叠进 UB（N→1 扇入，pull）。逐元素合并必须知道元素位宽，故 `eltype`（1/2/4 字节）选合并粒度——正是旧的 `reduce_group_to_ubuf`，模板 `T` 换成运行期 `eltype` = `sizeof(T)` 分派。
  `GridCollOp` 数值故意取 `comm::ReduceOp{Sum,Max,Min}` + 1，故 Tier-2 归约调用方用 `static_cast<GridCollOp>(uint32_t(commOp) + 1)` 映射（`COPY=0` 留给广播）。非自同步：data-ready 仍由调用方在 publish fence 后单独 `sync_hscb(READY)` 发出。它按**升序**折叠成员（member 0 播种 `ubTile`），所以 SPMD 行/列扇入能逐位复现方向性中继的从左到右累加（IEEE-754 加法可交换）。A3 mock 下归约把每个成员 GM→UB 拉进来，再用 per-member `combineScratch` 跑核内 `vadd`/`vmax`/`vmin`（mock 必需，COPY 与 native/`__CPU_SIM` 忽略）。三态函数体与旧的两个门面逐字一致——只是模板 `T`/`Op` 换成运行期 `eltype`/`op` 分派——故行为按构造保持 bit-exact。

`GRID_TBROADCAST` 的 payload 扇出是**每个 window 行一条** `mov_ubuf_group(..., op=COPY, eltype=1)`：组是整行或整列，成员占据连续 rank，接收 slot 因此**恒为**等步长 arena（`memberStride = 解析出的 slot₁ − slot₀`）。不再有 per-member `copy_ubuf_to_neighbor_ubuf` 回退——那条路只为多行子矩形组存在，而子矩形组已被移除。`treduce` ReduceSum 的 EAST（行）与 SOUTH（列）阶段在 sink（`col == gridCols-1` / `row == gridRows-1`）处用 `TREDUCE<ROW/COL, Sum, float>` 扇入，后者下译成 `mov_ubuf_group(..., op=SUM, eltype=sizeof(float))`——这是真正的 N→1 扇入，与 `tpush` ReduceSum 例子用 `TPOP<Dir>` + `TADD` + `TPUSH<Dir>` 拼出的方向性中继（`TREDUCE<Dir, Op>`）是不同的集合通信**形状**。组内**每个**成员都要调用它、而不只是 sink，各自把自己的位置与 `sinkBlockId` 操作数一比即得角色：该值指向的 block 收集，其余是贡献者，其半程只有握手（数据早已在 arena 里）。sink 之所以是运行期操作数、而非旧的"末位成员"约定，是因为结果落在哪里是**调用方**的摆放决策——它就在这个值下一步被用到的地方，而那通常并不是行末——且发起 gather 的核按构造就是收集者。每个成员必须传入相同的值；传入组外的 block 会报 `0x405 kFaultGroupBadPeer`。贡献者→sink 是扇入，而有了自增门铃它**根本不需要轮转**：每个贡献者给 sink 的 `ready_scb` 加 1，sink 等的是**总数** `cons_idx + 贡献者数`。每个贡献者用的那一对由**它自己**到 sink 的边方向索引；于是**内部** sink 两侧都有贡献者，整股扇入落在**两个** scoreboard 上——它按侧分别等待（空的一侧跳过），与 `TBROADCAST` 的前后向拆分同构。两侧各记各的数、互不干扰；当 sink 恰在组的末端时有一侧为空，退化成单次等待。随后 sink 在每个贡献者的 `free_scb` 上各加 1 作为回信用，这就是让下一个 H 段轮次得以开始的反压。它的 pipe 完全不带 ring（`DirMask = kGridDirNone`），window 就是那段 flag 头；phase B / C 各占一个子 window，互不继承对方的计数。两条 Tier-2 门面（`GRID_TBROADCAST` / `GRID_TREDUCE_GROUP_IMPL`）保留各自的 `<Group[,Op,T]>` 模板——该结构性信息属于 PTO 层，不在 CCE 指令层——但它们 tile 无关，所以 `gridpipe_payload_inl.hpp` 的 `TileUbPtr<T>` 负责抽取 tile 的 `__ubuf__` 指针交给 intrinsic。

### fp32 归约

归约 slot 携带 fp32 `[T, H]`，所以 `FFN_SLOT_BYTES = T * H * 4`。这让 `downPartial`、`yOutput` 和 `golden.bin` 都保持 fp32，host 可直接做容差比较。ReduceSum 的归约是按 H 分段的（`FFN_RS_REDUCE_SLOT_COUNT = kHSegs` = 7，每个 H 段一个 slot）：`treduce` 例子用 `TREDUCE<Group, Sum, float>` 在 sink 处折叠 segment-h 的 partial；`tpush` 例子用 `TPOP<EAST/SOUTH>` + `TADD` + `TPUSH<EAST/SOUTH>` 逐跳中继累加。在中继形式下 slot 数必须等于 `kHSegs`——跨段复用 slot 会在跨段 *free* 门铃上死锁。

### 串行组广播（TBROADCAST）

`TBROADCAST<GridGroup>`（首个模板参数为 `ROW` 或 `COL`）把本 cell 的 tile 一次性广播给所在行（`ROW`）或列（`COL`）的所有其它 cell：逐目标写入各接收方共享 ring 批量发出且目标之间无 fence，整个广播只付一次 publish fence，随后批量触发 ready 门铃。它不是按跳展开的 `TPUSH` 循环。

它里面**没有任何组私有的寻址**。源落进每个接收方的普通方向环、位置就是 `prod_idx % SlotCount`——`TPUSH<dir>` 会落的那一格，并敲对方的 `ready_scb[dir]`（`TPUSH` 用的那八个 scoreboard 之一），`dir` 按上面的规则逐条 (源, 接收方) 边归属。`SlotCount` 就是 `TPUSH` 的 `SlotCount`：没有单独的 `BcastSlotCount`、没有按 rank 的前缀偏移、也没有 per-source lane。`TPOP<GridGroup>(pipe, tile, srcBlockId)` 同样是普通 `TPOP` 算术：`srcBlockId` 只用于两件事——定该源那条边的**方向**、以及回收信用要送回的**地址**；等待门限与 slot 都取接收方自己的 `cons_idx`，于是接收侧既不需要知道调度、也不需要知道参与者集合。

把一个计数拆到最多四个 scoreboard 上依然天然正确：每个 scoreboard 只看到映射到它的那个**子序列**，而递增序列的子序列仍然递增——前提是各源按 rank 升序发布，这正是下面的轮转纪律已经要求的。

这一切成立的前提是每个写者只做 **+1**，而代价是调用方要还一笔债：**SPSC 调度**——任一瞬间一个组里只有一个成员在 `TBROADCAST` 内部，且它的 tile 在下一个成员发布之前已被所有接收方排空。单源广播（一个成员发、其余只 `TPOP`，如冒烟测试）天然满足。多个成员同时调用 `TBROADCAST` 属于**调用方错误**，不是受支持的模式；接收半程会把自己能就地看见的违约拦下来——`ready_scb` 已经**越过** `cons_idx + 1`，说明前一个源还没被排空、后一个源就发布了，于是报 `0x404 kFaultGroupOutOfOrder`，而不是默默排空错误的 tile。

**消费只由 TPOP 完成，而 `TBROADCAST` 在所有接收方都 TPOP 完之前不会返回。** 敲完门铃后它按侧阻塞在 `rounds * peerCount` 个信用上——每个接收方每次 `TPOP` 末尾 `+1`。注意这个门限**不含 `SlotCount` 项**：它不是单播 `TPUSH` 那条可流水的 `prod - SlotCount + 1` 信用测试，因为组的约定比"槽位可复用"更强——**上一条广播在所有接收端消费完之前，不允许有下一条广播**。代价是源放弃了跨轮流水，这就是"整组共用一条 ring"要付的钱。（顺带，这也是绝对写根本表达不了的：信用要由**每一个**接收方回流到那**一个**源，"K 个写者的最小值"没有任何单条 `WAIT_SPR` 能测，一个计数总和可以。）

原语靠自己做不到的是**告诉下一个源**：下一个源本地没有任何一个计数会因为别人的 tile 被排空而变化，也读不到对端状态。所以这个事实必须走一跳——但它不需要包、不需要第二条 pipe，更不需要接收端多做一次 `TPOP`：`TBROADCAST` 排空等待一结束，由 **`TBNOTIFY<Group>(pipe, dstBlockId)`** 给它点名的那个 block 的**轮转计分板 +1**（一次 `sync_hscb_add`，4 B，无 payload），而那个成员正阻塞在 **`TBWAIT<Group>`** 上。

**递轮转是一条独立指令，而不是发布的尾巴**，因为两者说的不是同一件事：`TBROADCAST` 确立的是关于**本次 tile 的事实**——所有接收方都已排空；而**下一个由谁发布**是**调度**，调度属于调用方。rank+1 环绕只是 AllGather 这一种形态；只有部分成员发布、或发布顺序依赖数据、或轮转跨阶段交接的集合通信，直接点名后继即可。唯一的定序要求是：`TBNOTIFY` 必须跟在它所代言的那次 `TBROADCAST` 之后。

`TBWAIT` 就是"第二次 `TBROADCAST` 的前半段"单独拿出来：**按反压判断下一条能不能写，但什么都不写**。判据本身属于上一个源（接收方排空时记的是**它**的 `free_scb`），所以由它评估、由它的 `TBNOTIFY` 把结论递过来，这边只消费一次结论。轮转计分板取的是**这个组不走的那条轴**——ROW 组用 NORTH、COL 组用 EAST；组只碰 EAST/WEST（或 NORTH/SOUTH），另一轴的那对 scb 一直闲着，因此这条纪律**不新增 scb、不新增 ring、不新增 window、不新增包**，组自己的 EAST/WEST 计数也原封不动（顺序检查照旧精确）。两个 AllGather 阶段仍在同一次升序遍历里走完（轮到自己就发布，轮到别人就 `TPOP`），调用侧的全部义务就是发布前一条 `TBWAIT<Group>`、发布后一条 `TBNOTIFY<Group>`；单源广播没有"下一个源"，两条都不调用即可。

**一条 `TBWAIT` 对应一条 `TBNOTIFY`。** 等待端不含任何针对成员、位置或轮次的特殊分支——两边的计数都只是"调用了几次"，这正是两侧无需商定绝对值也能对齐的原因。（早先版本曾把"组内位置 0 且尚未发布过"直接豁免，用来代表组的第一次发布；那等于把调度悄悄钉死在"从位置 0 开始"，并且在任何不从 0 开始的调度里会提前放行位置 0，因此已删除。）代价是遍历的**两端**由调用方留空——令牌必须先被制造才能被消费：**首发者不调用 `TBWAIT`**（没人通知它；或者它先用 `TBNOTIFY<Group>(pipe, 自己的 blockId)` 给自己造一个令牌，然后与其他成员一样等待），**有限遍历的末位发布者不调用 `TBNOTIFY`**。后一条不是洁癖：没被消费的令牌会**留在**对方计分板里——pipe 初始化只清零 `consIndex`，不清 GM 计数——从而满足后一轮、或复用同一 window 的后一次 launch 的第一条 `TBWAIT`。只有真正逐轮循环的调度才环绕，由末位通知首位。只有一个成员的组既是首发者也是末位，两条都不调用。

**组集合通信的 ring 配置。** 组本身不拥有 ring，它用 `DirMask` 指名的那两条方向 ring，每条深 `SlotCount`。发送端寻址 `prod_idx % SlotCount`（**自己**的发布计数），接收端寻址 `cons_idx % SlotCount`（它在该方向上的到达计数）。多源喂同一个方向时这两个计数并不相等——每个源的 `prod_idx` 都从 0 起，而接收端的 `cons_idx` 要跑遍所有源——所以**多源组的 `SlotCount` 只能是 1**，示例正是每方向一格（`FFN_NCUT_GROUP_SLOTS_P1/P2 = 1`；`SlotStride` 取 `FFN_NCUT_SLOT_BYTES_P1` = 1536 B 对应 `[8,96]` 分片、`FFN_NCUT_SLOT_BYTES_P2` = 12288 B 对应 `[8,768]` 行块；window = 1024 B flag 头 + 2 方向 × 1 槽）。更深的 ring 只对**单源**组有意义——那时两个计数才重合。per-member 的 payload 扇出本身在该组是等步长 arena 时仍坍缩成单条 `mov_ubuf_group` intrinsic（见 [组广播与归约 intrinsic](#组广播与归约-intrinsicg4--mov_ubuf_group)）。完整握手见 `Grid_TPUSH_TBROADCAST_TREDUCE_接口设计说明.md`。

### GridPipe 冒烟测试

组广播能力在 `smoke/` 下有一个 Vec-only 纯搬运冒烟测试（无 Cube、无 matmul、无数据文件，进程内校验，复用 FFN demo 的 GM-backed mock）：
- `bcast_smoke`：同一个 kernel 两种模式：
  - **单源**（默认）：源 cell（`--src`）经 `TBROADCAST<GridGroup>` 向整行（`--span-col 1` 时为整列）广播带标记 tile；其余 cell 用 `TPOP<GridGroup>(pipe, tile, src)` 取回并写出；host 校验 `out[cell] == in[source]`。默认 1x5 行、源在 col 2，一次运行同时覆盖源两侧的接收方。只有一个发布者就没有"下一个源"，因此不需要取轮转、也不需要递轮转，`TBWAIT` 与 `TBNOTIFY` 都不调用。
  - **全源**（`--all-src 1`）：每个成员依次广播——一个微缩版 AllGather，也是唯一覆盖**调用侧义务**的模式。每个成员在自己 `TBROADCAST` 前调用 `TBWAIT<Group>` 取轮转、之后调用 `TBNOTIFY<Group>` 把轮转递给下一个源，遍历两端留空（成员 0 不等、末位不递）。在默认 1x5 网格上加 `--span-col 1` 可让组只剩一个成员，正好覆盖"两端是同一个 cell"这一退化情形。host 逐 (接收方, 源) 校验 `out[cell][src] == in[src cell]`——这样"被后来的源覆盖掉的那一格"才会立刻暴露；求和式校验分不清"源 3 收了两次"和"源 3、源 4 各一次"。把这条 `TBWAIT` 删掉，32-cell FFN AllGather 就从 `max diff 0` 变成 `max diff 17648`（注意 demo 自带的 PASS 判据是错误率比例，那一跑照样打印 SUCCESS——必须看 max diff）。

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
# 单源广播：1x5 行，源在 col 2（--span-col 1 + Rx1 网格切到列广播）
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

冒烟脚本复用 `-r/-v/-d`、`--grid-rows/--grid-cols`、`--token-tile/--model-tile`（tile `[T, W]`）和 `--build-only`。`run_bcast_smoke.sh` 额外提供 `--src`（源下标，默认 2）、`--all-src`（1 = 每个成员都广播，默认 0 = 单源）与 `--span-col`（1 为列组，默认 0 即行组）。

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
