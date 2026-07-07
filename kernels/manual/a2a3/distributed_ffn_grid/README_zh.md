# Single-device Multi-block FFN GridPipe Demo

## 整体目标

`distributed_ffn_grid_reducesum` 当前用于验证 A2/A3 上的单卡逻辑 FFN 网格。host 在选定 device 上启动单进程，并 launch `gridRows * gridCols` 个 block。每个 block 对应一个逻辑 cell：

- `gridRows` 是 data-parallel token 分片。
- `gridCols` 是 model-parallel FFN intermediate 分片。
- mixed Cube/Vec kernel 在单次 launch 中完成 gate/up、activation、down projection 和行内 `EAST` reduce。
- 每行最右列写出 fp32 `[T, H]` 输出 tile，host 使用 `1e-3` 容差与 `golden.bin` 比对。

`EAST` reduce 使用 A2/A3 GridPipe mock backend：本地 SRAM windows（当前由 GM 分配模拟）、fake `HcclDeviceContext` window 指针、ready/free counter、`dcci/dsb` fence 和 spin wait。该 demo 验证的是单设备 mock 路径和 GridPipe 编程模型，不是多卡通信验证。

仓内同时提供一个 AllGather 版本：`run_allgather.sh` / `distributed_ffn_grid_allgather`。它在 hidden 阶段先沿列做 fp16 AllGather，再让 down 阶段按输出 `H` 维切片，去掉 post-down ReduceSum。

除最近邻 FFN demo 外，GridPipe 还支持路由 K-hop unicast（`TPUSH<dir, dist>` / `TPOP<dir, dist>`）和单源行/列广播（`TPUSH<GridSpan>`）。这两个能力各有一个独立的 Vec-only 冒烟测试，位于 `smoke/` 子目录（见下文「GridPipe 冒烟测试」）。

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `README_zh.md` / `README.md` | 中文 / 英文说明文档。 |
| `CMakeLists.txt` | 构建 host 可执行文件和 mixed Cube/Vec device kernel shared libraries。 |
| `run_reducesum.sh` | ReduceSum 版本的一键构建、数据生成和运行脚本。 |
| `run_allgather.sh` | AllGather 版本的一键构建、数据生成和运行脚本。 |
| `ffn_config.hpp` | 编译期配置：逻辑网格尺寸、tile 尺寸、GridPipe window 字节数、buffer 字节数、PReLU alpha。 |
| `kernel_launch.hpp` | host 侧 mixed kernel launch 接口声明。 |
| `main_reducesum.cpp` | ReduceSum host driver：ACL 初始化、fake HCCL context/local GridPipe windows、device buffer 分配、数据加载、kernel launch、golden 校验和资源清理。 |
| `distributed_ffn_grid_reducesum_compute_kernel.cpp` | ReduceSum mixed Cube/Vec kernel。Cube 负责 GEMM，Vec 负责 activation/cast 和 GridPipe `EAST` reduce。 |
| `main_allgather.cpp` | AllGather host driver。 |
| `distributed_ffn_grid_allgather_compute_kernel.cpp` | AllGather mixed Cube/Vec kernel。Vec 负责 hidden shard 收集，Cube 写出 output-H shard。 |
| `tpipe_tmov_inl.hpp` | 把 Cube↔Vec 的 C2V/V2C 搬运封装成方向化 `TMOV` 重载，内部转发到现有 `TPUSH`/`TPOP`，使 kernel 正文不再出现该 handshake。 |
| `gridpipe_payload_inl.hpp` | 本地 GridPipe payload/remote pointer adaptor。 |
| `smoke/` | GridPipe 特性冒烟测试。`khop_smoke_{config,kernel,launch}` + `main_khop_smoke.cpp` + `run_khop_smoke.sh` 覆盖路由 K-hop unicast；`bcast_smoke_*` + `run_bcast_smoke.sh` 覆盖单源行/列广播。两者都通过父目录 `CMakeLists.txt` 构建。 |
| `../../../../include/pto/npu/a2a3/grid_intrinsic.hpp` | 合并后的 GridPipe A2/A3 intrinsic 层。Section 3 是 V6 IPC_SCB 计分板 API（`sync_neighbor_scb`/`wait_local_spr`/`mov_local_spr`），GridPipe ready/free wait 与 notify 经过这里；Section 4 是 neighbor SRAM 地址解析与 payload 搬运 API（`copy_ubuf_to_neighbor_ubuf`/`copy_local_slot_to_ubuf`），并声明 `GmSramArena` 地址段 SRAM 模型与 `sram_pop_is_local` 这条 TPOP 读本地性守卫。 |
| `scripts/gen_data.py` | 生成每个 cell 的 fp16 X/weight shard，以及 fp32 golden reference。 |
| `build/` | 被忽略的生成 build 目录。 |
| `out/` | 被忽略的生成数据目录。 |

## 运行流程

1. `run_reducesum.sh` 解析参数。默认 `gridRows=2`、`gridCols=2`、`T=16`、`H=64`、`Fi=64`、`n-ranks=1`。
2. 如果没有指定 `--build-only`，`scripts/gen_data.py` 生成 per-cell 输入、权重文件和 `golden.bin`。
3. CMake 构建两个 target：
   - `distributed_ffn_grid_reducesum_mixed_kernel`：`dav-c220` mixed Cube/Vec kernel。
   - `distributed_ffn_grid_reducesum`：host 可执行文件。
4. host 在选定 device 上初始化 ACL。
5. host 按 `gridRows * gridCols` 个 cell 分配连续 device buffers。
6. host 分配每个 cell 一个本地 GridPipe SRAM window（当前用 GM backing），并构造 fake `HcclDeviceContext`：

```text
windowsIn[cell] = reduce_pipe_windows_dev + cell * FFN_GRID_WINDOW_BYTES
rankNum = gridRows * gridCols
winSize = FFN_GRID_WINDOW_BYTES
```

7. host 加载每个 cell 的 X 和 weight shard。
8. host 通过 `rtGetC2cCtrlAddr()` 获取 FFTS base address，并单次 launch `DistributedFfnGridMixedKernel`。
9. kernel 内 Cube 和 Vec 分支通过 A2/A3 `TPipe` FIFO 交换中间 tile。这些 C2V/V2C 搬运在 kernel 正文里以方向化 `TMOV` 表达（`TMOV(pipe, tile)` 生产、`TMOV(tile, pipe)` 消费），底层 `TPUSH`/`TPOP` 隐式完成（见 `tpipe_tmov_inl.hpp`）：

```text
Cube:
  X[row] @ W_gate[col] -> gatePartial[row,col] --TMOV C2V-->
  X[row] @ W_up[col]   -> upPartial[row,col]   --TMOV C2V-->

Vec:
  hidden[row,col] = fp16(PReLU(gatePartial) * upPartial)
  hidden[row,col] --TMOV V2C-->

Cube:
  hidden[row,col] @ W_down[col] -> downPartial[row,col] --TMOV C2V-->

Vec:
  downPartial --GridPipe EAST reduce across cols--> yOutput[row] on final col
```

跨 cell 的 `EAST`/`WEST` reduce 与 gather 仍保留显式的 GridPipe `TPUSH`/`TPOP`；只有 block 内 Cube↔Vec 的 C2V/V2C 搬运被收敛到 `TMOV`。

10. host 同步 stream，检查 GridPipe fault flags，拷回 `yOutput` 并与 `golden.bin` 比对。

## 关键设计

### 1. Mixed Cube/Vec 单次 launch

device kernel 编译为 `dav-c220`。Cube 和 Vec 分支分别由 `__DAV_CUBE__`、`__DAV_VEC__` 保护，两个分支位于同一个 kernel source 中，通过 A2/A3 `TPipe` ready/free handshake 同步。

### 2. 隐式 C2V/V2C `TMOV`

`tpipe_tmov_inl.hpp` 新增两个 `TMOV` 重载，让 kernel 正文用单条 tile-move 表达 Cube↔Vec 搬运，而不再显式写 `TPUSH`/`TPOP`：

- `TMOV(pipe, tile)`：生产侧，转发到 `TPUSH`（把 `tile` 写入 C2V/V2C FIFO）。
- `TMOV(tile, pipe)`：消费侧，转发到 `TPOP`（把下一个 slot 读入 `tile`）。

哪个物理核负责写/读、pipe 是 C2V 还是 V2C，都仍由 `TPipe` 类型及其 `__DAV_CUBE__`/`__DAV_VEC__` 保护决定，因此调用点与方向无关。这两个重载只接收 `(pipe, tile)`/`(tile, pipe)` 两个参数（没有 wait-event 包），比通用的 tile-to-tile `TMOV(dst, src, ...)` 更特化；重载决议因而会对任意 `TPipe`/tile 组合选中它们，而其它所有 `TMOV` 用法保持不变。这样既把 Cube↔Vec handshake 隐藏在调用点之后（贴近真实 WSE fabric move 隐藏生产/消费拆分的方式），又原样复用现有 `TPUSH`/`TPOP` 的同步与 record 机制。

### 3. 单卡逻辑网格

`get_block_idx()` 是 row-major cell id：

```text
cell = get_block_idx()
row  = cell / gridCols
col  = cell % gridCols
```

所有 cell 都在同一个 device 上运行。`gridRows` 控制 data-parallel token tile 数，`gridCols` 控制 model-parallel FFN shard 数。

### 4. 本地 GridPipe mock

host 分配 `gridRows * gridCols` 个本地 SRAM windows（当前由 GM backing）。`TPUSH<EAST>` 通过 `get_neighbor_sram_addr` 解析 east neighbor 的 SRAM slot 后写入 payload，再发布 ready counter；`TPOP<EAST>` 等待本地 ready counter、读取本地 SRAM slot，并向 west neighbor 归还 free credit。

mock 使用 GM flag polling 和 cache maintenance 在 A2/A3 上模拟 LPU WSE 预期的 `SPR` / `WFE` 行为。

### 5. NoC 只写不读的地址段 SRAM 模型（`GmSramArena`）

为了贴近真实硬件，mock 把未来硬件的"每核私有 SRAM"显式建模为一个 **GM 地址段（address segment）数据结构**：那块连续的 `gridRows*gridCols * FFN_GRID_WINDOW_BYTES` window 缓冲被切成等长的 per-core 段，于是第 `c` 段（即 `windowsIn[c]`）就是第 `c` 个核的私有 SRAM：

```text
段 c = [base + c*winSize, base + (c+1)*winSize)   // base == windowsIn[0]
```

`GmSramArena`（位于 `include/pto/npu/a2a3/grid_intrinsic.hpp`）持有 `{base, segBytes, numSegs}` 以及 `SegmentOf` / `InSegment` 判定函数；demo 在 device 侧从 fake `HcclDeviceContext` 的 window 表构造它（`SramArenaFromCtx`）。它是"某地址归哪个核所有"的唯一真相来源。

这样就把真实硅片的 NoC 约束显式化并**强制**起来：fabric 只能跨核**写**，不能跨核**读**。

- `TPUSH<dir>` 把 payload 写入**邻居核**的段——这是跨段写，正是 fabric 的行为。
- `TPOP<dir>` 只能 POP **本核自己**的段。`GRID_TRY_TPOP_IMPL` 在 payload 读取前先调用 `sram_pop_is_local` 守卫；一旦发生跨段读，就写入 `kFaultPopNonLocal`（`0x205`，"pop non-local segment"）并放弃本次 pop，host 的 `CheckGridPipeFaults` 会报出来。

在 native 硬件上 `sram_pop_is_local` 是恒为 `true` 的 no-op：TPOP 的读地址天然就是本地的，因为 fabric 根本没有远程读通路。这个守卫只是为了 A2/A3 mock——mock 用一块 GM window 模拟 SRAM，它物理上可以读任意地址，若没有守卫，demo 就可能悄悄依赖一次硅片做不到的远程读。每个 A2/A3 kernel 都会编入一条 `static_assert(GmSramArenaSelfCheck())`，因此段计算一旦回归就会在编译期失败，而不是把 pop 误路由出去。

> `pto::comm` 版本（`TREDUCE` / `TGATHER`）有意**不**遵守该约束：它们是 root 直接读取每个 rank 的 collective（HCCL/RDMA 式的远程读），与 WSE NoC 是不同的内存模型。只有 GridPipe `TPUSH`/`TPOP` 路径被约束成只写不读。

### 6. IPC_SCB 计分板 intrinsic API

GridPipe 的 ready/free 同步走 V6 IPC_SCB 计分板路线，以 CCE intrinsic 风格 API 暴露在 `include/pto/npu/a2a3/grid_intrinsic.hpp`（Section 3）。规范调用形态把硬件语义参数放在前面，mock backend operand 放在末尾：

- `sync_neighbor_scb(kind, dir, dist, abs_count, operand)`（V6 `SYNC_HSCB`/`ST_HSCB`）：把本核新的绝对计数 store 进 `dist` 跳外对端的 `ready_scb`/`free_scb`（IPC_SCB）。
- `wait_local_spr(kind, dir, threshold, operand, maxSpins)`（V6 `WAIT_SPR`）：阻塞等待**本核** `ready_scb`/`free_scb`（IPC_SCB）达到 `threshold`。
- `mov_local_spr(kind, dir, operand)`（V6 `MOV_SPR2X`）：对本核计分板的非阻塞 peek，用于 `wait_local_spr` 前的快路径。

GridPipe payload 的远端 SRAM 地址解析走同一头文件（Section 4）：

- `get_neighbor_sram_addr(dst, src, dir, peerRank, operand)`：将本地 slot offset 解析为邻居 SRAM slot 地址寄存器。
- `copy_ubuf_to_neighbor_ubuf(dst, src, bytes, config)`（V6 `COPY_UBUF_TO_NBR`）：把本核 UB payload 写入邻居核的 UB/L1 slot。
- `copy_local_slot_to_ubuf(dst, src, bytes, config)`：把本核自己的 slot drain 进 tile。V6 **无跨核读** payload：native lowering 即现成本地 TLOAD/TMOV（此处为接口占位）；A2/A3 mock 读 GM-backed 本地 slot。
- `sram_pop_is_local(slot, bytes, callerRank, operand)`：TPOP 读本地性守卫（见上文地址段模型）。native lowering 恒返回 `true`；A2/A3 mock 用 `callerRank` 的 `GmSramArena` 段校验 `slot`，跨段读会被拒绝而不是被悄悄服务。

native lowering 对接真实 CCE HSCB/IPC_SCB 栈（`__sync_hscb`/`__st_hscb`、`get_ipc_scb_*`，等待用 `try_wait(CROSS_CORE)`——因头文件尚未暴露 IPC_SCB 上的阻塞 `WAIT_SPR`）；当前 A2/A3 mock 用 GM 字 + cache 维护替代这些 IPC_SCB 计分板，并把 `WAIT_SPR` 退化为自旋轮询。native 硬件提供邻居 IPC_SCB 寻址（V6 HW-DEP-1）与 `COPY_UBUF_TO_NBR` builtin（V6 HW-DEP-0）后，上层 GridPipe 调用点不需要修改。

`TPUSH<EAST>` 先用 `mov_local_spr`/`wait_local_spr` peek/等待本核 `free_scb`，写 payload slot，然后用 `sync_neighbor_scb` 把 `prod_idx` 发布到下游 `ready_scb`。`TPOP<EAST>` 先等待本核 `ready_scb`，读取 payload slot，再用 `sync_neighbor_scb` 把 `cons_idx` 发布到上游 `free_scb`。

当前 A2/A3 上，`ScbOperand::addr` 指向本地/fake peer GridPipe window 中的 GM-backed 计分板字，`NeighborSramOperand::runtimeCtx` 指向 fake HCCL context。真实硬件支持邻居 IPC_SCB store/`WAIT_SPR` 与 neighbor SRAM address register 后，应分别通过 `PTO_GRID_COUNTER_NATIVE_INTRINSIC`、`PTO_GRID_SRAM_NATIVE_INTRINSIC` 编译 GridPipe，并由编译器/头文件提供对应 `__sync_hscb`/`get_ipc_scb_*`/`try_wait` 与 `__builtin_cce_copy_ubuf_to_neighbor_ubuf`。此时 mock operand 被忽略，host/device 侧应从 fake GM window 切到硬件提供的 per-neighbor IPC_SCB 与 SRAM slot base；GridPipe `TPUSH/TPOP` 调用点不需要变化。

### 7. fp32 EAST reduce

reduce slot 携带 fp32 `[T, H]`，所以 `FFN_SLOT_BYTES = T * H * 4`。`downPartial`、`yOutput` 和 `golden.bin` 都保持 fp32，host 可直接做容差比较。

### 8. 路由 K-hop unicast

`TPUSH<dir, dist>` / `TPOP<dir, dist>` 把最近邻 pipe 扩展为沿 `dir` 方向 `dist` 跳的路由 unicast（Scheme A）。payload 写入解析到 `dist` 跳邻居段内的 slot，ready/free 计分板 store 通过 `sync_neighbor_scb` 携带同样的 `dist` 操作数，因此下游 `dist` 跳的接收方直接 pop，中间不需要任何中继。`dist == 1` 即原有最近邻行为。fan-in 保持为 1（每个方向/距离只有一个上游），无需 slot/flag 扩容。

### 9. 单源行/列广播

`TPUSH<GridSpan>`（首个模板参数为 `ROW` 或 `COL`，类型本身即选中多播重载）让单个源 cell 把 tile 一次性广播给所在行或列的所有其它 cell：逐目标写入批量发出且目标之间无 fence，整个广播只付一次 publish fence，随后所有 ready doorbell 批量触发。它不是按跳展开的 `TPUSH` 循环。接收方用普通 `TPOP<dir, dist>` 朝源方向取数（行广播为 `EAST`/`WEST`，列广播为 `NORTH`/`SOUTH`）。

### 10. GridPipe 冒烟测试

上述两个能力在 `smoke/` 下各有一个 Vec-only 纯搬运冒烟测试（无 Cube、无 matmul、无数据文件，进程内校验，复用 FFN demo 的 GM-backed mock）：

- `khop_smoke`：`1 x cols` 行网格；每个 cell 向东 `DIST` 跳 push 一块带标记的 fp32 `[T, W]` tile，接收方 pop 后写出；host 校验 `out[c] == in[c-DIST]`。
- `bcast_smoke`：源 cell（`--src`）向整行（`--span-col 1` 时为整列）广播带标记 tile；其余 cell 各按自己的方向/距离 pop 并写出；host 校验 `out[cell] == in[source]`。默认 1x5 行、源在 col 2，一次运行同时覆盖东西两臂。

### 11. AllGather 版本

`run_allgather.sh` 使用 `scripts/gen_data.py --split-mode allgather` 生成数据。此时 `W_down` 按 `[F, Hc]` 切列，`GridPipe` 搬运的是 fp16 `hidden [T, Fi]`，最终输出是每列一个 `Y[:, Hc]` shard，host 仍与完整 `golden.bin` 做拼接后的比对。AllGather 要求 `--model-tile` 能被 `--grid-cols` 整除，保证 `Hc = H / gridCols` 是整数 tile 宽度。

## 运行方法

### 仅编译

```bash
bash run_reducesum.sh --build-only -v Ascend910B1 --grid-rows 2 --grid-cols 2
bash run_allgather.sh --build-only -v Ascend910B1 --grid-rows 2 --grid-cols 2
```

### NPU 运行

```bash
bash run_reducesum.sh -r npu -v Ascend910B1 --device-id 0 --grid-rows 3 --grid-cols 3
bash run_allgather.sh -r npu -v Ascend910B1 --device-id 0 --grid-rows 3 --grid-cols 3 --model-tile 96
```

### GridPipe 冒烟测试

```bash
# 路由 K-hop unicast：1x4 行，整体右移 2 跳
bash smoke/run_khop_smoke.sh -r npu -v Ascend910B1 --device-id 0 --grid-cols 4 --dist 2
# 单源广播：1x5 行，源在 col 2（--span-col 1 + Rx1 网格可切到列广播）
bash smoke/run_bcast_smoke.sh -r npu -v Ascend910B1 --device-id 0 --grid-cols 5 --src 2
```

两者均支持 `--build-only`，且不需要生成数据。

### 常用参数

```text
-r, --run-mode      sim 或 npu，默认 npu
-v, --soc-version   默认 Ascend910B1
-n, --n-ranks       固定为 1
-d, --device-id     ACL device id；默认依次取 FFN_GRID_DEVICE_ID、ASCEND_DEVICE_ID、DEVICE_ID、0
--grid-rows         单卡逻辑网格行数，默认 2
--grid-cols         单卡逻辑网格列数，默认 2
--token-tile        每个 cell 的 token tile T，默认 16
--model-tile        hidden dim H，默认 64；AllGather 要求 H % gridCols == 0
--ffn-tile          每列 intermediate dim Fi，默认 64
--build-only        只编译，不生成数据和运行
```

冒烟脚本复用 `-r/-v/-d`、`--grid-rows/--grid-cols`、`--token-tile/--model-tile`（tile `[T, W]`）和 `--build-only`；`run_khop_smoke.sh` 额外提供 `--dist`（跳数，默认 2），`run_bcast_smoke.sh` 额外提供 `--src`（源下标，默认 2）和 `--span-col`（1 为列广播，默认 0）。

## 期望输出

ReduceSum 成功时最终打印：

```text
[SUCCESS] Single-device multi-block FFN GridPipe PASS.
```

AllGather 成功时最终打印：

```text
[SUCCESS] Single-device multi-block FFN GridPipe AllGather PASS.
```

冒烟测试成功时分别打印：

```text
[SUCCESS] GridPipe K-hop unicast smoke PASS.
[SUCCESS] GridPipe single-source broadcast smoke PASS.
```
