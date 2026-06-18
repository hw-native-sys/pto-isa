# Python DSL Flash Attention 用例

## 概览

本用例展示如何使用 PTO Python DSL（`ptodsl`）实现高性能 Flash Attention。该用例的目标是提供一个性能最优的 Python-DSL Flash Attention 实现。该实现围绕四阶段软流水构建：

```text
compute_qk（Cube） -> compute_p（Vector） -> compute_pv（Cube） -> compute_gu（Vector）
```

它表明 Python DSL 能够表达一个完整经过性能调优的、生产级 Flash Attention 流水线，包括 Cube/Vector 协同、运行时 S1 循环、通过全局内存进行软件 FIFO 暂存、结果正确性校验，以及与 `torch_npu.npu_fused_infer_attention_score` 的性能对比。

## 支持平台

- Ascend A3 类目标（`compile.sh` 中使用 `--pto-arch=a3`、`--npu-arch=dav-2201`）
- 已配置 `bisheng` 的 CANN 环境
- PTO 汇编器 `ptoas`
- 包含 `ptodsl`、`torch`、`torch_npu` 的 Python 环境

## 外部依赖

本用例不是自包含示例。它依赖外部 `huawei-csl/pto-dsl` 仓提供的 PTO Python DSL 包：

```text
https://github.com/huawei-csl/pto-dsl
```

本用例已验证的版本信息如下：

```text
pto-dsl package version: 0.1.2
upstream branch: main
commit: 6755794cfc145c8ffe4fae92483aa20148e57327
commit date: 2026-05-18 11:18:56 +0200
```

`ptodsl` 在两处被使用：

- `kernels/fa_builder.py` 导入 `ptodsl` 来构造 PTO MLIR module。
- `run.py` 导入 `ptodsl` 的 benchmark 和 NPU device helper。

构建或运行本用例前，需要先安装兼容版本的 `pto-dsl` 包。例如安装当前上游 main 分支：

```bash
python3 -m pip install --user --upgrade git+https://github.com/huawei-csl/pto-dsl.git
```

可以用下面的命令确认 Python 实际导入的是哪个 `ptodsl` 包：

```bash
python3 -c "import ptodsl, inspect; from ptodsl import to_ir_module; print(ptodsl.__file__); print(inspect.signature(to_ir_module))"
```

`compile.sh` 使用的 `PTO_LIB_PATH` 与 Python `ptodsl` 包是两类不同依赖。`PTO_LIB_PATH` 必须指向当前 PTO-ISA tile library 仓库 checkout，以便 `bisheng` 找到 `include/` 下的头文件。

完整的构建/运行依赖包括：

- 来自 `https://github.com/huawei-csl/pto-dsl` 的 `ptodsl`
- `torch` 和 `torch_npu`
- 包含 `bisheng` 的 Ascend CANN runtime/toolkit
- 与已安装 `ptodsl` 生成的 MLIR 兼容的 `ptoas`
- 通过 `PTO_LIB_PATH` 提供的本仓 PTO C++ 头文件

### 已验证环境版本（用于精确复现）

下表为最近一次复现性能与正确性结果所用的精确版本。复现时请按此固定版本；混用
其他版本通常需要配套的 `ptoas` 与头文件组合（见表后说明）。

| 依赖 | 版本（建议固定） | 说明 |
| --- | --- | --- |
| PTO-ISA 仓（本仓，经 `PTO_LIB_PATH` 提供头文件） | commit `b9122ec586b5a7b4b686ca7498874a1c94b3573c`（2026-06-12） | 还包含 `kernels/python/flash_atten/` 下的 Flash-Attention 工作区改动，以及 `include/pto/common/pto_instr.hpp` 中的 GlobalData 头文件修复（见下） |
| `pto-dsl` Python 包（`ptodsl`） | `0.1.2`，commit `6755794cfc145c8ffe4fae92483aa20148e57327`（2026-05-18），分支 `main` | 唯一的 `ptodsl` 运行/构建依赖 |
| `ptoas` | `0.45` | 必须与 `ptodsl` 生成的 MLIR 及本仓 C++ 头文件兼容 |
| CANN toolkit（`Ascend-cann-toolkit`） | `9.0.0`（inner `V100R001C10SPC001B250`） | 提供 `bisheng` |
| `bisheng` | clang `15.0.5`（CANN 9.0.0 自带） | |
| `torch` | `2.9.0` | |
| `torch_npu` | `2.9.0` | |
| Python | `3.10.19` | |
| Ascend 驱动 | `26.0.rc1`（ascendhal `7.35.23`） | |
| 目标 NPU | Ascend A3 类（`--pto-arch=a3`、`--npu-arch=dav-2201`） | |

**`ptoas` 与头文件的配套关系。** `ptoas` 的 codegen 必须与本仓 C++ 头文件兼容。
使用 `ptoas 0.45` 时，NPU 路径上的公共 GlobalData FIFO 指令
（`TALLOC` / `TPUSH` / `TPOP` / `TFREE`）在 `include/pto/common/pto_instr.hpp` 中
调用其 `*_IMPL` 时必须显式带上模板参数 `<Pipe, GlobalData, Split>`；缺少该 4 行
修复（与现有 `__CPU_SIM` 分支保持一致）时 `bisheng` 会报 `no member named 'cons'`。
该修复是使用 `ptoas >= 0.45` 构建本用例的前提。

## 目录结构

```text
kernels/python/flash_atten/
├── caller.cpp              # Host 侧 shim，导出供 ctypes 调用的 call_kernel
├── compile.sh              # 生成 MLIR/C++，并构建 build_artifacts/fa.so
├── kernels/
│   ├── fa_builder.py       # PTO Python DSL Flash Attention kernel 构造器
│   └── ptodsl_compat.py    # ptodsl 0.1.2 本地兼容层
├── scripts/
│   └── patch_vec_barriers.py
│                           # 生成 C++ 的 PIPE_V barrier patch helper
└── run.py                  # 构建、运行、校验和性能测试入口
```

生成产物位于 `build_artifacts/`：

```text
build_artifacts/fa.mlir     # fa_builder.py 生成的 MLIR
build_artifacts/fa.cpp      # ptoas 生成的 C++
build_artifacts/fa_patched.cpp
                           # compile.sh 删除 barrier 后的可选 C++ 产物
build_artifacts/fa.so       # run.py 加载的动态库
build_artifacts/fa_summary_*.tsv
```

## Kernel 范围

当前的形状与功能约束。该 kernel 被特意特化到单一形状族，以便实现能够针对峰值性能进行调优：

- `HEAD = 128`
- 每个 Q block 的 `S0 = 128`
- 默认 `S1_TILE = 256`；可通过 `FA_S1_TILE=512` 进行实验
- `CUBE_S1 = 128`
- DSL 运行路径使用经过性能调优的默认值 `QK_PRELOAD = 3`（可通过 `FA_QK_PRELOAD=4` 进行实验）
- `EXP_RING = QK_PRELOAD` 是硬性不变量（`FA_EXP_RING` 必须等于 `FA_QK_PRELOAD`）：softmax 与 `compute_gu` 复用配套的 rescale slot，因此二者必须共享相同的 ring 深度
- `KV_SPLIT` 默认为 `1`（`FA_KV_SPLIT` 可取 `1`、`2`、`4`）。当 `> 1` 时，每个 Q block 的 KV（S1）会被切分到 `KV_SPLIT` 个 work-unit，使得 `NUM_Q_BLOCKS * KV_SPLIT` 个 unit 填满 cube core 的波次；随后第二次 grid launch（`call_reduce`）会按 Q block 将各 unit 的部分结果 flash-combine 成最终归一化的 `O`。默认集合会按形状与设备自动选择（见[构建与运行](#构建与运行)）
- 仅支持非 causal attention
- Q 总行数通过 `FA_Q_ROWS` 配置，并且必须是 `128` 的整数倍
- KV 总行数由 `run.py` 在运行时传入；每个 S1 长度必须能被所选 `S1_TILE` 整除，且在 `KV_SPLIT = 1` 时至少切出 `QK_PRELOAD` 个 tile。当 `KV_SPLIT > 1` 时，tile 数还必须能被 `KV_SPLIT` 整除，且每个分片仍至少包含 `QK_PRELOAD` 个 tile（即 `S1 / S1_TILE / KV_SPLIT >= QK_PRELOAD`）

生成的动态库会针对当前 `FA_Q_ROWS` 特化，S1 长度则在运行时处理。

## 构建与运行

1. 配置 Ascend CANN 环境。

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. 进入用例目录并设置 PTO 头文件路径。

```bash
cd ${git_clone_path}/kernels/python/flash_atten
export PTO_LIB_PATH=${git_clone_path}
```

如果 `ptoas` 或 `bisheng` 不在 `PATH` 中，可以显式设置：

```bash
export PTOAS=/path/to/ptoas
export BISHENG=/path/to/bisheng
```

3. 运行一个默认 benchmark case。

```bash
python3 run.py --case case1
```

4. 运行完整默认 benchmark 集合。

```bash
python3 run.py
```

默认集合会运行 `case1` 到 `case8`，并针对每个 `FA_Q_ROWS` 重新编译 kernel。每个 case 会按形状选择两个参数：

- `S1_TILE`：`S1 < 32768`（`case1`–`case5`）使用 `256`，`S1 >= 32768`（`case6`–`case8`）使用 `512`。
- `KV_SPLIT`：按形状**和设备**（cube core 波次利用率）自动选择。在参考的 24 cube core A3 目标上，只有 `case5` 会启用 `KV_SPLIT=2`，其余保持 `KV_SPLIT=1`；cube core 数不同的设备可能选择不同。可用 `FA_KV_SPLIT` 覆盖。

| Case | Q rows（S0 total） | KV rows（S1） | `S1_TILE` | `KV_SPLIT`\* |
| --- | ---: | ---: | ---: | ---: |
| `case1` | 1024 | 1024 | 256 | 1 |
| `case2` | 2048 | 2048 | 256 | 1 |
| `case3` | 4096 | 4096 | 256 | 1 |
| `case4` | 8192 | 8192 | 256 | 1 |
| `case5` | 16384 | 16384 | 256 | 2 |
| `case6` | 32768 | 32768 | 512 | 1 |
| `case7` | 65536 | 65536 | 512 | 1 |
| `case8` | 131072 | 131072 | 512 | 1 |

\*`KV_SPLIT` 列对应参考的 24 cube core A3 目标；它会按 cube core 数自动选择，其他设备可能不同。

### Split-KV（`KV_SPLIT`）波次量化修复

当某个 Q block 产生的 cube work-unit 不足以填满所有 cube core 波次（波次利用率偏低）时，`KV_SPLIT > 1` 会把每个 Q block 的 KV 区间切成 `KV_SPLIT` 个 work-unit，使 `TOTAL_UNITS = NUM_Q_BLOCKS * KV_SPLIT`。每个 unit 计算未归一化的部分结果（`O_acc`、running max、running sum）；随后第二次 grid launch（`call_reduce`，由 `compile.sh` 的 `-DFA_KV_SPLIT` 与 `caller.cpp` 的 `#if FA_KV_SPLIT > 1` 控制）按 Q block 将这些部分结果 flash-combine 成最终归一化的 `O`。`KV_SPLIT=1` 即精确基线（不切分、不做第二次 launch）。只有当切分能显著提升利用率、且每个分片仍满足 `QK_PRELOAD` prologue 下限时才有收益，这也是默认集合在参考设备上仅对 `case5` 启用它的原因。

## 自定义 Case

通过 `FA_Q_ROWS` 和 `FA_BENCH_LENGTHS` 运行自定义形状：

```bash
FA_Q_ROWS=1024 FA_BENCH_LENGTHS=1024 python3 run.py
```

对同一个 Q 形状测试多个 S1 长度：

```bash
FA_Q_ROWS=2048 FA_BENCH_LENGTHS=1024,2048,4096 python3 run.py
```

控制 benchmark 迭代次数：

```bash
FA_Q_ROWS=1024 FA_BENCH_LENGTHS=1024 FA_BENCH_WARMUP=20 FA_BENCH_ITERS=200 python3 run.py
```

覆盖 split-KV 因子（自定义单次运行默认 `KV_SPLIT=1`，仅默认集合会自动选择）：

```bash
FA_Q_ROWS=16384 FA_BENCH_LENGTHS=16384 FA_S1_TILE=256 FA_KV_SPLIT=2 python3 run.py
```

通过环境变量选择 benchmark 计时模式（默认 `event`；`sync` 为更保守的 host 计时模式，等价于 `--timing sync`）：

```bash
FA_Q_ROWS=1024 FA_BENCH_LENGTHS=1024 FA_BENCH_TIMING=sync python3 run.py
```

当 `build_artifacts/fa.so` 已经按相同 `FA_Q_ROWS` 编译过时，可以复用已有动态库：

```bash
FA_Q_ROWS=1024 bash compile.sh
FA_Q_ROWS=1024 FA_BENCH_LENGTHS=1024 python3 run.py --no-build
```

## 删除冗余 PIPE_V Barrier

`compile.sh` 支持在 `ptoas` 生成 C++ 后、`bisheng` 编译前删除选定的 `pipe_barrier(PIPE_V);`。推荐使用基于 op pattern 的方式，而不是依赖生成文件中的固定行号：

稳定的 `gu` pattern 已作为 `compile.sh` 和 `run.py` 的默认行为启用。需要额外 pattern 时可以显式传入：

```bash
python3 run.py --case case1
python3 run.py --remove-vec-barrier-patterns gu,softmax-sum-add
```

也可以通过环境变量传入：

```bash
FA_REMOVE_VEC_BARRIER_PATTERNS=gu,softmax-sum-add python3 run.py
```

需要关闭默认 `gu` patch 做对比时，可以传入 `--remove-vec-barrier-patterns none`。

当前支持的 pattern：

| Pattern | 别名 | 作用 | 说明 |
| --- | --- | --- | --- |
| `gu` | `trowexpandmul-tadd` | 删除 `TROWEXPANDMUL -> pipe_barrier(PIPE_V) -> wait_flag(PIPE_MTE2, PIPE_V) -> TADD` 中的 vector barrier | 当前稳定策略，用于减少 `compute_gu` 阶段中 MTE wait 已经提供间隔时的冗余 V pipe barrier。 |
| `softmax-exp-sum` | `texp-trowsum` | 尝试删除 `TEXP -> pipe_barrier(PIPE_V) -> TROWSUM` 中的 barrier | 带直接 tile 依赖检查。若 `TROWSUM` 读取 `TEXP` 写出的 tile，该候选会被保留并在编译日志中报告 skipped。 |
| `softmax-sum-add` | `trowsum-tadd` | 删除 `TROWSUM -> pipe_barrier(PIPE_V) -> TADD` 中无直接 tile 依赖的 barrier | 实验策略，可与 `gu` 组合验证；对小/中尺寸 case 可能有收益，大尺寸通常接近持平。 |

编译日志会输出实际删除数量，例如：

```text
Patched generated C++ -> .../fa_patched.cpp (removed 74 PIPE_V barriers; lines=0, patterns=74)
Skipped PIPE_V barrier pattern candidates: softmax-exp-sum:direct-tile-dependency=2
```

需要按行号复现实验时，也可以使用 `--remove-vec-barriers line1,line2,...` 或 `FA_REMOVE_VEC_BARRIERS`，但该方式依赖当前生成 C++ 的行号，不适合作为默认方案。

性能验证时建议至少运行默认全量 case。`ptodsl` event timing 对 `ctypes` 自定义 kernel launch 偶发可能出现不可信的 0.x us 结果；此时应使用更保守的同步计时模式：

```bash
python3 run.py --timing sync
```

## 输出与正确性

`run.py` 会输出每个形状的时延、吞吐、相对 `torch_npu` 融合 attention 的加速比，以及最大误差。正确性对比包括：

- 当 `Q_ROWS * S1` 不太大时，使用 host 侧 FP32 PyTorch reference
- 所有 benchmark 尺寸都会对比 `torch_npu.npu_fused_infer_attention_score`

TFLOP/s 由 matmul、scale 和 softmax 的操作量统计得到（与上游参考 `run.py` 采用相同的操作计数口径）。**TFLOP/s 是随输入形状与数据变化的实测值，而非固定指标**——不同形状、不同运行之间的绝对数值会变化，因此对比时应以各形状的时延以及相对 `torch_npu.npu_fused_infer_attention_score` 的加速比为准。

默认集合会自动生成 summary TSV。也可以通过 `FA_SUMMARY_TSV` 指定输出路径：

```bash
FA_SUMMARY_TSV=/tmp/fa_summary.tsv python3 run.py --case case1
```

## 期望性能

> 下表为上文[版本表](#已验证环境版本用于精确复现)所列固定参考环境（Ascend A3 类，
> 24 cube core；`ptoas 0.45`；`torch_npu 2.9.0`）下的**目标**区间。正如
> [输出与正确性](#输出与正确性)所述，绝对时延与 TFLOP/s 会随形状、数据和设备变化——
> 请将其视为需要复现的参考区间，而非硬性指标（SLA）。

将 `include/pto/npu/a2a3/TPush.hpp` 中的 `TPipe::SyncPeriod` 设置优化后（详情见
[issue #172](https://github.com/hw-native-sys/pto-isa/issues/172)），默认集合预期可在
参考目标上达到下表区间。小 case 使用 `event` 计时（设备时间）；大 case 使用 `sync`
计时，以规避超大 `ctypes` launch 上偶发的 `event` 读数 glitch（见
[删除冗余 PIPE_V Barrier](#删除冗余-pipe_v-barrier)）。

| Case | S0 = S1 | DSL 时延（µs） | `torch_npu`（µs） | 加速比 | 计时 |
| --- | ---: | ---: | ---: | :---: | :---: |
| `case1` | 1024 | 21 | 54 | 2.55× | event |
| `case2` | 2048 | 38 | 73 | 1.91× | event |
| `case3` | 4096 | 105 | 139 | 1.32× | event |
| `case4` | 8192 | 239 | 291 | 1.22× | event |
| `case5` | 16384 | 1001 | 985 | 0.98× | sync |
| `case6` | 32768 | 3185 | 3184 | 1.00× | sync |
| `case7` | 65536 | 12106 | 12158 | 1.00× | sync |
| `case8` | 131072 | 48062 | 48135 | 1.00× | sync |

期望形态为：小尺寸大幅领先（`torch_npu` 有固定的 host 派发开销），随尺寸增大收窄，
在最大尺寸基本持平（~185 TFLOP/s）。

## 注意事项

- `compile.sh` 默认将 `PTO_LIB_PATH` 设为 `/sources/pto-isa`；在本仓工作时建议显式设置 `PTO_LIB_PATH=${git_clone_path}`。
- `--no-build` 只适合单个已选 case，因为 `fa.so` 会按 `FA_Q_ROWS` 重新构建。
- 长序列可能跳过 host 侧 FP32 reference，以避免分配过大的 QK 矩阵；此时会用更宽松阈值对比 NPU fused reference。
