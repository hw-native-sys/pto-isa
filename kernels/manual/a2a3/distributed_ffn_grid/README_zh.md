# Single-device Multi-block FFN GridPipe Demo

## 整体目标

`distributed_ffn_grid` 当前用于验证单卡多核 FFN，并在单卡内使能 `gridCols > 1` 的 GridPipe EAST reduce mock。它不再需要多卡、`mpirun` 或真实 HCCL 通信；host 会在 device 0 上构造一个逻辑 `gridRows x gridCols` 网格，每个 grid cell 对应一个 block。

默认配置：

- `gridRows=2`：data-parallel 行数，不同行处理不同 token tile。
- `gridCols=2`：model-parallel 列数，每列持有 FFN intermediate dim 的一个 shard。
- 启动 `gridRows * gridCols` 个 block。
- 每个 cell 计算本列 FFN partial，随后同行按 `EAST` 方向 reduce。
- 每行最右列写出 `[T, H]` fp32 `yOutput`，host 与 `golden.bin` 做 `1e-3` 容差校验。

该 demo 关注单设备多核调度、cube/vec 分阶段执行，以及 GridPipe ready/free/slot 语义在单卡 GM window 上的 mock。它不是多卡通信验证。

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `README_zh.md` / `README.md` | 中文 / 英文说明文档。 |
| `CMakeLists.txt` | 构建 host 可执行文件，以及 cube/compute 和 vec/comm 两个 device kernel shared library。 |
| `run.sh` | 一键设置 CANN 环境、生成输入数据、配置 CMake、编译并在单进程中启动 demo。 |
| `ffn_config.hpp` | 编译期配置中心：逻辑网格尺寸、tile 尺寸、GridPipe window 字节数、buffer 字节数、PReLU alpha。 |
| `kernel_launch.hpp` | host 侧 kernel launch 接口声明。 |
| `main.cpp` | host driver：ACL 初始化、fake HCCL context/local GridPipe windows、device buffer 分配、数据加载、分阶段 launch、golden 校验和资源清理。 |
| `distributed_ffn_grid_compute_kernel.cpp` | cube kernel：phase 0 计算 `X @ W_gate` 和 `X @ W_up`；phase 1 计算 `hidden @ W_down -> downPartial[cell]`。 |
| `distributed_ffn_grid_comm_kernel.cpp` | vec/comm kernel：phase 0 执行 `PReLU(gate) * up -> hidden`；phase 1 所有列并行执行 EAST reduce。 |
| `gridpipe_payload_inl.hpp` | 本地 GridPipe payload/remote pointer adaptor。会从 fake `windowsIn[]` 反推当前 cell window，并解析 peer cell 的同 offset 地址。 |
| `scripts/gen_data.py` | 生成每个 cell 的 fp16 X/weight shard，以及 fp32 golden reference。 |

## 运行流程

1. `run.sh` 解析参数。默认 `gridRows=2`、`gridCols=2`、`T=16`、`H=64`、`Fi=64`。
2. `scripts/gen_data.py` 生成输入、权重和 golden：
   - `pe_<cell>_x.bin`: 当前 cell 所在 row 的 `[T, H]` fp16 输入。
   - `pe_<cell>_w_gate.bin`: 当前 cell 所在 col 的 `[H, Fi]` fp16 权重 shard。
   - `pe_<cell>_w_up.bin`: `[H, Fi]` fp16 权重 shard。
   - `pe_<cell>_w_down.bin`: `[Fi, H]` fp16 权重 shard。
   - `golden.bin`: `[gridRows * T, H]` fp32 输出。
3. host 初始化 ACL 并固定使用 device 0。
4. host 分配连续 device buffers，大小按 `gridRows * gridCols` 个 cell 扩展：
   - `x_dev`
   - `w_gate_dev` / `w_up_dev` / `w_down_dev`
   - `gate_partial_dev` / `up_partial_dev`
   - `hidden_dev`
   - `down_partial_dev`
   - `y_output_dev`，只按 `gridRows` 个输出 tile 分配
5. host 分配 `gridRows * gridCols` 个本地 GridPipe GM window，并构造 fake `HcclDeviceContext`：

```text
windowsIn[cell] = reduce_pipe_windows_dev + cell * FFN_GRID_WINDOW_BYTES
rankNum = gridRows * gridCols
winSize = FFN_GRID_WINDOW_BYTES
```

6. host 顺序 launch：
   - compute phase 0：所有 cell 计算 gate/up partial。
   - comm phase 0：所有 cell 计算 hidden。
   - compute phase 1：所有 cell 计算 downPartial。
   - comm phase 1：单次 launch 所有列并行执行 EAST reduce step。
7. host D2H 拷回完整 `yOutput`，与 `golden.bin` 比对。

数据流摘要：

```text
compute phase 0, all cells:
  X[row] @ W_gate[col] -> gatePartial[row,col]
  X[row] @ W_up[col]   -> upPartial[row,col]

comm phase 0, all cells:
  hidden[row,col] = fp16(PReLU(gatePartial[row,col]) * upPartial[row,col])

compute phase 1, all cells:
  hidden[row,col] @ W_down[col] -> downPartial[row,col]

comm phase 1, all cols in one launch:
  col 0       : TPUSH<EAST>(downPartial[row,0])
  middle cols : TPOP<EAST> + TADD + TPUSH<EAST>
  final col   : TPOP<EAST> + TADD + TSTORE(yOutput[row])
```

## 关键设计

### 1. 单卡逻辑网格

`get_block_idx()` 是 row-major cell id：

```text
cell = get_block_idx()
row = cell / gridCols
col = cell % gridCols
```

所有 cell 都在同一个 device 上运行。`gridRows` 控制 data-parallel token tile 数，`gridCols` 控制 model-parallel FFN shard 数。

### 2. 本地 GridPipe mock

原多卡路径依赖 HCCL remote window。当前单卡路径用一段连续 GM 内存模拟每个 cell 的 window，并用 fake `HcclDeviceContext::windowsIn[]` 做地址解析。

`TPUSH<EAST>` 仍然写入 east neighbor 的 slot，随后写 ready flag；`TPOP<EAST>` 仍然等待本地 ready flag、读取 slot、写 upstream free flag。差别只是 peer window 都在同一张卡上。

### 3. 并行 reduce 与超时保护

reduce 阶段单次 launch 所有列，各列之间通过 GridPipe ready/free flag 同步：

```text
launch comm phase 1
  col 0 producer TPUSH<EAST>
  col > 0 consumer TPOP<EAST> 阻塞等待上游 ready
```

当前 A2/A3 backend 使用 GM flag + spin wait 模拟 LPU WSE SPR/WFE。为避免上板时因为 block 调度或 flag
可见性问题导致无限卡死，mock wait 带有最大自旋次数；若等待超时，kernel 会在对应 GridPipe window flag
中写入 fault code，host 在 reduce 后检查这些 flag 并失败退出。

## 运行方法

### 仅编译

```bash
bash run.sh --build-only -v Ascend910B1 --grid-rows 2 --grid-cols 2
```

### NPU 运行

```bash
task-submit --device auto --run "cd $(pwd)/kernels/manual/a2a3/distributed_ffn_grid && bash run.sh -r npu -v Ascend910B1 --grid-rows 2 --grid-cols 2"
```

### 常用参数

```text
-r, --run-mode      sim 或 npu，默认 npu
-v, --soc-version   默认 Ascend910B1
-n, --n-ranks       固定为 1
--grid-rows         单卡逻辑网格行数，默认 2
--grid-cols         单卡逻辑网格列数，默认 2
--token-tile        每个 cell 的 token tile T，默认 16
--model-tile        hidden dim H，默认 64
--ffn-tile          每列 intermediate dim Fi，默认 64
--build-only        只编译，不生成数据和运行
```

## 期望输出

成功时最终打印：

```text
[SUCCESS] Single-device multi-block FFN GridPipe PASS.
```
