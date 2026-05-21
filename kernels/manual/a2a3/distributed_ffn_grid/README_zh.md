# Single-device Multi-block FFN GridPipe Demo

## 整体目标

`distributed_ffn_grid` 当前用于验证 A2/A3 上的单卡逻辑 FFN 网格。host 在选定 device 上启动单进程，并 launch `gridRows * gridCols` 个 block。每个 block 对应一个逻辑 cell：

- `gridRows` 是 data-parallel token 分片。
- `gridCols` 是 model-parallel FFN intermediate 分片。
- mixed Cube/Vec kernel 在单次 launch 中完成 gate/up、activation、down projection 和行内 `EAST` reduce。
- 每行最右列写出 fp32 `[T, H]` 输出 tile，host 使用 `1e-3` 容差与 `golden.bin` 比对。

`EAST` reduce 使用 A2/A3 GridPipe mock backend：本地 SRAM windows（当前由 GM 分配模拟）、fake `HcclDeviceContext` window 指针、ready/free counter、`dcci/dsb` fence 和 spin wait。该 demo 验证的是单设备 mock 路径和 GridPipe 编程模型，不是多卡通信验证。

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
| `../../../../include/pto/common/grid_sram_intrinsic.hpp` | CCE intrinsic 风格的 neighbor SRAM 地址解析和 payload 搬运 API，GridPipe payload movement 会经过这里。 |
| `scripts/gen_data.py` | 生成每个 cell 的 fp16 X/weight shard，以及 fp32 golden reference。 |
| `build/` | 被忽略的生成 build 目录。 |
| `out/` | 被忽略的生成数据目录。 |

## 运行流程

1. `run.sh` 解析参数。默认 `gridRows=2`、`gridCols=2`、`T=16`、`H=64`、`Fi=64`、`n-ranks=1`。
2. 如果没有指定 `--build-only`，`scripts/gen_data.py` 生成 per-cell 输入、权重文件和 `golden.bin`。
3. CMake 构建两个 target：
   - `distributed_ffn_grid_mixed_kernel`：`dav-c220` mixed Cube/Vec kernel。
   - `distributed_ffn_grid`：host 可执行文件。
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

host 分配 `gridRows * gridCols` 个本地 SRAM windows（当前由 GM backing）。`TPUSH<EAST>` 通过 `get_neighbor_ubuf_addr` 解析 east neighbor 的 Vector UB slot 后写入 payload，再发布 ready counter；`TPOP<EAST>` 等待本地 ready counter、读取本地 UB slot，并向 west neighbor 归还 free credit。

mock 使用 GM flag polling 和 cache maintenance 在 A2/A3 上模拟 LPU WSE 预期的 `SPR` / `WFE` 行为。

### 4. Neighbor counter intrinsic API

GridPipe 的 ready/free 同步统一经过 `include/pto/common/grid_counter_intrinsic.hpp` 中两个 CCE intrinsic 风格 API。规范调用形态把硬件语义参数放在前面，mock backend operand 放在末尾：

- `mtspr_neighbor_counter(kind, dir, value, operand)`：向 `dir` 对应的 neighbor-visible counter 发布单调递增的 `Ready` 或 `Free` 值。
- `wfe_neighbor_counter(kind, dir, threshold, operand, maxSpins)`：等待本地 counter mirror 达到 `threshold`。

GridPipe payload 的远端 SRAM 地址解析统一经过 `include/pto/common/grid_sram_intrinsic.hpp`：

- `get_neighbor_ubuf_addr(dst, src, dir, peerRank, operand)` / `get_neighbor_cbuf_addr(dst, src, dir, peerRank, operand)`：将本地 slot offset 解析为邻居 Vector UB 或 Cube L1/CBUF slot 地址寄存器。
- `copy_ubuf_to_neighbor_ubuf(dst, src, bytes, config)` / `copy_ubuf_to_neighbor_cbuf(dst, src, bytes, config)`：Vector UB 到邻居 Vector UB 或 Cube L1 的写入接口。
- `copy_cbuf_to_neighbor_ubuf(dst, src, bytes, config)` / `copy_cbuf_to_neighbor_cbuf(dst, src, bytes, config)`：Cube L1 到邻居 Vector UB 或 Cube L1 的写入接口。
- `copy_neighbor_ubuf_to_ubuf(dst, src, bytes, config)`：A2/A3 mock 中 TPOP 侧从本地 GM-backed slot 加载到 consumer Vector UB tile 的接口；native Grid TPOP 预期通过硬件 SRAM 地址机制绑定本地 slot。

当前 A2/A3 mock 中，UB 源路径通过 GM-backed fake window 上的 MTE 搬运实现；`__cbuf__` 源路径通过现有 `copy_cbuf_to_gm` 写入 GM-backed fake window。native 硬件提供对应 builtin 后，上层 GridPipe 调用点不需要修改。

`TPUSH<EAST>` 先用 `wfe_neighbor_counter` 等待 `Free` credit，写 payload slot，然后用 `mtspr_neighbor_counter` 发布 `Ready`。`TPOP<EAST>` 先等待 `Ready`，读取 payload slot，再向上游发布 `Free`。

当前 A2/A3 上，`NeighborCounterOperand::addr` 指向本地/fake peer GridPipe window 中的 GM-backed counter，`NeighborSramOperand::runtimeCtx` 指向 fake HCCL context。真实硬件支持 neighbor SPR/WFE counter 和 neighbor SRAM address register 后，应分别通过 `PTO_GRID_COUNTER_NATIVE_INTRINSIC`、`PTO_GRID_SRAM_NATIVE_INTRINSIC` 编译 GridPipe，并由编译器提供对应 `__builtin_pto_*`。此时 mock operand 被忽略，host/device 侧应从 fake GM window 切到硬件提供的 per-neighbor counter/event register 与 SRAM slot base；GridPipe `TPUSH/TPOP` 调用点不需要变化。

### 5. fp32 EAST reduce

reduce slot 携带 fp32 `[T, H]`，所以 `FFN_SLOT_BYTES = T * H * 4`。`downPartial`、`yOutput` 和 `golden.bin` 都保持 fp32，host 可直接做容差比较。

## 运行方法

### 仅编译

```bash
bash run.sh --build-only -v Ascend910B1 --grid-rows 2 --grid-cols 2
```

### NPU 运行

```bash
bash run.sh -r npu -v Ascend910B1 --device-id 0 --grid-rows 3 --grid-cols 3
```

### 常用参数

```text
-r, --run-mode      sim 或 npu，默认 npu
-v, --soc-version   默认 Ascend910B1
-n, --n-ranks       固定为 1
-d, --device-id     ACL device id，默认 0
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
