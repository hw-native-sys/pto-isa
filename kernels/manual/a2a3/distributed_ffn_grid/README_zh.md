# Single-device Multi-block FFN GridPipe Demo

## 整体目标

`distributed_ffn_grid_reducesum` 当前用于验证 A2/A3 上的单卡逻辑 FFN 网格。host 在选定 device 上启动单进程，并 launch `gridRows * gridCols` 个 block。每个 block 对应一个逻辑 cell：

- `gridRows` 是 data-parallel token 分片。
- `gridCols` 是 model-parallel FFN intermediate 分片。
- mixed Cube/Vec kernel 在单次 launch 中完成 gate/up、activation、down projection 和行内 `EAST` reduce。
- 每行最右列写出 fp32 `[T, H]` 输出 tile，host 使用 `1e-3` 容差与 `golden.bin` 比对。

`EAST` reduce 使用 A2/A3 GridPipe mock backend：本地 SRAM windows（当前由 GM 分配模拟）、fake `HcclDeviceContext` window 指针、ready/free counter、`dcci/dsb` fence 和 spin wait。该 demo 验证的是单设备 mock 路径和 GridPipe 编程模型，不是多卡通信验证。

仓内同时提供一个 AllGather 版本：`run_allgather.sh` / `distributed_ffn_grid_allgather`。它在 hidden 阶段先沿列做 fp16 AllGather，再让 down 阶段按输出 `H` 维切片，去掉 post-down ReduceSum。

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
| `../../../../include/pto/common/grid_counter_intrinsic.hpp` | CCE intrinsic 风格的 neighbor counter API，GridPipe ready/free wait 与 notify 会经过这里。 |
| `../../../../include/pto/common/grid_sram_intrinsic.hpp` | CCE intrinsic 风格的 neighbor SRAM 地址解析和 payload 搬运 API，GridPipe payload movement 会经过这里。同时声明 `GmSramArena` 地址段 SRAM 模型与 `sram_pop_is_local` 这条 TPOP 读本地性守卫。 |
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

`GmSramArena`（位于 `include/pto/common/grid_sram_intrinsic.hpp`）持有 `{base, segBytes, numSegs}` 以及 `SegmentOf` / `InSegment` 判定函数；demo 在 device 侧从 fake `HcclDeviceContext` 的 window 表构造它（`SramArenaFromCtx`）。它是"某地址归哪个核所有"的唯一真相来源。

这样就把真实硅片的 NoC 约束显式化并**强制**起来：fabric 只能跨核**写**，不能跨核**读**。

- `TPUSH<dir>` 把 payload 写入**邻居核**的段——这是跨段写，正是 fabric 的行为。
- `TPOP<dir>` 只能 POP **本核自己**的段。`GRID_TRY_TPOP_IMPL` 在 payload 读取前先调用 `sram_pop_is_local` 守卫；一旦发生跨段读，就写入 `kFaultPopNonLocal`（`0x205`，"pop non-local segment"）并放弃本次 pop，host 的 `CheckGridPipeFaults` 会报出来。

在 native 硬件上 `sram_pop_is_local` 是恒为 `true` 的 no-op：TPOP 的读地址天然就是本地的，因为 fabric 根本没有远程读通路。这个守卫只是为了 A2/A3 mock——mock 用一块 GM window 模拟 SRAM，它物理上可以读任意地址，若没有守卫，demo 就可能悄悄依赖一次硅片做不到的远程读。每个 A2/A3 kernel 都会编入一条 `static_assert(GmSramArenaSelfCheck())`，因此段计算一旦回归就会在编译期失败，而不是把 pop 误路由出去。

> `pto::comm` 版本（`TREDUCE` / `TGATHER`）有意**不**遵守该约束：它们是 root 直接读取每个 rank 的 collective（HCCL/RDMA 式的远程读），与 WSE NoC 是不同的内存模型。只有 GridPipe `TPUSH`/`TPOP` 路径被约束成只写不读。

### 6. Neighbor counter intrinsic API

GridPipe 的 ready/free 同步统一经过 `include/pto/common/grid_counter_intrinsic.hpp` 中两个 CCE intrinsic 风格 API。规范调用形态把硬件语义参数放在前面，mock backend operand 放在末尾：

- `mtspr_neighbor_counter(kind, dir, value, operand)`：向 `dir` 对应的 neighbor-visible counter 发布单调递增的 `Ready` 或 `Free` 值。
- `wfe_neighbor_counter(kind, dir, threshold, operand, maxSpins)`：等待本地 counter mirror 达到 `threshold`。

GridPipe payload 的远端 SRAM 地址解析统一经过 `include/pto/common/grid_sram_intrinsic.hpp`：

- `get_neighbor_sram_addr(dst, src, dir, peerRank, operand)`：将本地 slot offset 解析为邻居 SRAM slot 地址寄存器。
- `copy_sram_to_neighbor_sram(dst, src, bytes, config)`：本核 SRAM 到邻居核 SRAM 的写入接口，不再区分 UB/CBUF。
- `copy_neighbor_sram_to_sram(dst, src, bytes, config)`：对应的跨核读接口。native lowering 当前仅提供接口占位；A2/A3 mock 用它做 TPOP 侧 GM-backed slot 校验。
- `sram_pop_is_local(slot, bytes, callerRank, operand)`：TPOP 读本地性守卫（见上文地址段模型）。native lowering 恒返回 `true`；A2/A3 mock 用 `callerRank` 的 `GmSramArena` 段校验 `slot`，跨段读会被拒绝而不是被悄悄服务。

当前 A2/A3 mock 中，SRAM 写入/读取路径通过 GM-backed fake window 上的 MTE 搬运实现。native 硬件提供对应写 builtin 后，上层 GridPipe 调用点不需要修改。

`TPUSH<EAST>` 先用 `wfe_neighbor_counter` 等待 `Free` credit，写 payload slot，然后用 `mtspr_neighbor_counter` 发布 `Ready`。`TPOP<EAST>` 先等待 `Ready`，读取 payload slot，再向上游发布 `Free`。

当前 A2/A3 上，`NeighborCounterOperand::addr` 指向本地/fake peer GridPipe window 中的 GM-backed counter，`NeighborSramOperand::runtimeCtx` 指向 fake HCCL context。真实硬件支持 neighbor SPR/WFE counter 和 neighbor SRAM address register 后，应分别通过 `PTO_GRID_COUNTER_NATIVE_INTRINSIC`、`PTO_GRID_SRAM_NATIVE_INTRINSIC` 编译 GridPipe，并由编译器提供对应 `__builtin_pto_*`。此时 mock operand 被忽略，host/device 侧应从 fake GM window 切到硬件提供的 per-neighbor counter/event register 与 SRAM slot base；GridPipe `TPUSH/TPOP` 调用点不需要变化。

### 7. fp32 EAST reduce

reduce slot 携带 fp32 `[T, H]`，所以 `FFN_SLOT_BYTES = T * H * 4`。`downPartial`、`yOutput` 和 `golden.bin` 都保持 fp32，host 可直接做容差比较。

### 8. AllGather 版本

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

## 期望输出

ReduceSum 成功时最终打印：

```text
[SUCCESS] Single-device multi-block FFN GridPipe PASS.
```

AllGather 成功时最终打印：

```text
[SUCCESS] Single-device multi-block FFN GridPipe AllGather PASS.
```
