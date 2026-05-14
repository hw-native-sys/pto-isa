# Single-device Multi-block FFN GridPipe Demo

## 整体目标

`distributed_ffn_grid` 当前用于验证 A2/A3 上的单卡逻辑 FFN 网格。host 在 device 0 上启动单进程，并 launch `gridRows * gridCols` 个 block。每个 block 对应一个逻辑 cell：

- `gridRows` 是 data-parallel token 分片。
- `gridCols` 是 model-parallel FFN intermediate 分片。
- mixed Cube/Vec kernel 在单次 launch 中完成 gate/up、activation、down projection 和行内 `EAST` reduce。
- 每行最右列写出 fp32 `[T, H]` 输出 tile，host 使用 `1e-3` 容差与 `golden.bin` 比对。

`EAST` reduce 使用 A2/A3 GridPipe mock backend：本地 GM windows、fake `HcclDeviceContext` window 指针、ready/free counter、`dcci/dsb` fence 和 spin wait。该 demo 验证的是单设备 mock 路径和 GridPipe 编程模型，不是多卡通信验证。

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `README_zh.md` / `README.md` | 中文 / 英文说明文档。 |
| `CMakeLists.txt` | 构建 host 可执行文件和 mixed Cube/Vec device kernel shared library。 |
| `run.sh` | 一键设置 CANN 环境、生成输入数据、配置 CMake、编译并在单进程中启动 demo。 |
| `ffn_config.hpp` | 编译期配置：逻辑网格尺寸、tile 尺寸、GridPipe window 字节数、buffer 字节数、PReLU alpha。 |
| `kernel_launch.hpp` | host 侧 mixed kernel launch 接口声明。 |
| `main.cpp` | host driver：ACL 初始化、fake HCCL context/local GridPipe windows、device buffer 分配、数据加载、kernel launch、golden 校验和资源清理。 |
| `distributed_ffn_grid_compute_kernel.cpp` | mixed Cube/Vec kernel。Cube 负责 GEMM，Vec 负责 activation/cast 和 GridPipe `EAST` reduce。 |
| `gridpipe_payload_inl.hpp` | 本地 GridPipe payload/remote pointer adaptor。 |
| `../../../../include/pto/common/grid_counter_intrinsic.hpp` | CCE intrinsic 风格的 neighbor counter API，GridPipe ready/free wait 与 notify 会经过这里。 |
| `scripts/gen_data.py` | 生成每个 cell 的 fp16 X/weight shard，以及 fp32 golden reference。 |
| `build/` | 被忽略的生成 build 目录。 |
| `out/` | 被忽略的生成数据目录。 |

## 运行流程

1. `run.sh` 解析参数。默认 `gridRows=2`、`gridCols=2`、`T=16`、`H=64`、`Fi=64`、`n-ranks=1`。
2. 如果没有指定 `--build-only`，`scripts/gen_data.py` 生成 per-cell 输入、权重文件和 `golden.bin`。
3. CMake 构建两个 target：
   - `distributed_ffn_grid_mixed_kernel`：`dav-c220` mixed Cube/Vec kernel。
   - `distributed_ffn_grid`：host 可执行文件。
4. host 初始化 ACL 并固定使用 device 0。
5. host 按 `gridRows * gridCols` 个 cell 分配连续 device buffers。
6. host 分配每个 cell 一个本地 GridPipe GM window，并构造 fake `HcclDeviceContext`：

```text
windowsIn[cell] = reduce_pipe_windows_dev + cell * FFN_GRID_WINDOW_BYTES
rankNum = gridRows * gridCols
winSize = FFN_GRID_WINDOW_BYTES
```

7. host 加载每个 cell 的 X 和 weight shard。
8. host 通过 `rtGetC2cCtrlAddr()` 获取 FFTS base address，并单次 launch `DistributedFfnGridMixedKernel`。
9. kernel 内 Cube 和 Vec 分支通过 A2/A3 `TPipe` FIFO 交换中间 tile：

```text
Cube:
  X[row] @ W_gate[col] -> gatePartial[row,col] --TPipe C2V-->
  X[row] @ W_up[col]   -> upPartial[row,col]   --TPipe C2V-->

Vec:
  hidden[row,col] = fp16(PReLU(gatePartial) * upPartial)
  hidden[row,col] --TPipe V2C-->

Cube:
  hidden[row,col] @ W_down[col] -> downPartial[row,col] --TPipe C2V-->

Vec:
  downPartial --GridPipe EAST reduce across cols--> yOutput[row] on final col
```

10. host 同步 stream，检查 GridPipe fault flags，拷回 `yOutput` 并与 `golden.bin` 比对。

## 关键设计

### 1. Mixed Cube/Vec 单次 launch

device kernel 编译为 `dav-c220`。Cube 和 Vec 分支分别由 `__DAV_CUBE__`、`__DAV_VEC__` 保护，两个分支位于同一个 kernel source 中，通过 A2/A3 `TPipe` ready/free handshake 同步。

### 2. 单卡逻辑网格

`get_block_idx()` 是 row-major cell id：

```text
cell = get_block_idx()
row  = cell / gridCols
col  = cell % gridCols
```

所有 cell 都在同一个 device 上运行。`gridRows` 控制 data-parallel token tile 数，`gridCols` 控制 model-parallel FFN shard 数。

### 3. 本地 GridPipe mock

host 分配 `gridRows * gridCols` 个本地 GM windows。`TPUSH<EAST>` 通过 fake HCCL window table 写入 east neighbor 的 slot 和 ready counter；`TPOP<EAST>` 等待本地 ready counter、读取 slot，并向 west neighbor 归还 free credit。

mock 使用 GM flag polling 和 cache maintenance 在 A2/A3 上模拟 LPU WSE 预期的 `SPR` / `WFE` 行为。

### 4. Neighbor counter intrinsic API

GridPipe 的 ready/free 同步统一经过 `include/pto/common/grid_counter_intrinsic.hpp` 中两个 CCE intrinsic 风格 API：

- `mtspr_neighbor_counter(operand, kind, dir, value)`：向 `dir` 对应的 neighbor-visible counter 发布单调递增的 `Ready` 或 `Free` 值。
- `wfe_neighbor_counter(operand, kind, dir, threshold)`：等待本地 counter mirror 达到 `threshold`。

`TPUSH<EAST>` 先用 `wfe_neighbor_counter` 等待 `Free` credit，写 payload slot，然后用 `mtspr_neighbor_counter` 发布 `Ready`。`TPOP<EAST>` 先等待 `Ready`，读取 payload slot，再向上游发布 `Free`。

当前 A2/A3 上，`NeighborCounterOperand::addr` 指向本地/fake peer GridPipe window 中的 GM counter，API 会落到 mock 的 `dcci/dsb` 与 spin-wait 实现。真实硬件支持 neighbor SPR/WFE counter 后，本示例应通过 `PTO_GRID_COUNTER_NATIVE_INTRINSIC` 编译 GridPipe，并由编译器提供 `__builtin_pto_mtspr_neighbor_counter` 和 `__builtin_pto_wfe_neighbor_counter`。此时 counter operand 地址由 API 忽略，host/device 侧应从 fake GM ready/free counter 切到硬件提供的 neighbor counter/event register；GridPipe `TPUSH/TPOP` 调用点不需要变化。payload slot 映射仍可由后端选择 GridPipe window 或真实硬件 slot 地址机制，但 counter 发布和等待不应再走 GM polling。

### 5. fp32 EAST reduce

reduce slot 携带 fp32 `[T, H]`，所以 `FFN_SLOT_BYTES = T * H * 4`。`downPartial`、`yOutput` 和 `golden.bin` 都保持 fp32，host 可直接做容差比较。

## 运行方法

### 仅编译

```bash
bash run.sh --build-only -v Ascend910B1 --grid-rows 2 --grid-cols 2
```

### NPU 运行

```bash
task-submit --device auto --run "cd $(pwd)/kernels/manual/a2a3/distributed_ffn_grid && bash run.sh -r npu -v Ascend910B1 --grid-rows 3 --grid-cols 3"
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
