# Python DSL Flash Attention 用例

## 概览

本用例展示如何使用 PTO Python DSL（`ptodsl`）实现高性能 Flash Attention。该实现是 `kernels/manual/common/flash_atten` 手写 kernel 的 Python-DSL 迁移与对齐实验，保留了手写版本中的四阶段软流水：

```text
compute_qk（Cube） -> compute_p（Vector） -> compute_pv（Cube） -> compute_gu（Vector）
```

实现过程中也参考了 Huawei CSL PTO DSL AOT Flash Attention 140 TFLOPS 示例：

```text
https://github.com/huawei-csl/pto-dsl/tree/main/examples/aot/flash_attention/140tflops
```

该用例的意义在于验证 Python DSL 对高性能 Flash Attention 这类复杂算子的表达能力，包括 Cube/Vector 协同、运行时 S1 循环、通过全局内存进行软件 FIFO 暂存、结果正确性校验，以及与 `torch_npu.npu_fused_infer_attention_score` 的性能对比。

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

当前形状和功能约束有意与手写版本对齐：

- `HEAD = 128`
- 每个 Q block 的 `S0 = 128`
- 默认 `TILE_S1 = 256`；可通过 `FA_S1_TILE=512` 进行实验
- `CUBE_S1 = 128`
- DSL 调优后的运行路径使用 `QK_PRELOAD = 3`（可通过 `FA_QK_PRELOAD=4` 进行实验；`MANUAL_QK_PRELOAD = 4` 仅保留为 parity 目标元数据）
- 仅支持非 causal attention
- Q 总行数通过 `FA_Q_ROWS` 配置，并且必须是 `128` 的整数倍
- KV 总行数由 `run.py` 在运行时传入；每个 S1 长度需要不小于 `S1_TILE * QK_PRELOAD`，并且能被所选 `S1_TILE` 整除

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

默认集合会运行 `case1` 到 `case8`，并针对每个 `FA_Q_ROWS` 重新编译 kernel。`case1` 到 `case4` 使用 `TILE_S1=256`，`case5` 到 `case8`（`S1 >= 16384`）使用 `TILE_S1=512`：

| Case | Q rows（S0 total） | KV rows（S1） |
| --- | ---: | ---: |
| `case1` | 1024 | 1024 |
| `case2` | 2048 | 2048 |
| `case3` | 4096 | 4096 |
| `case4` | 8192 | 8192 |
| `case5` | 16384 | 16384 |
| `case6` | 32768 | 32768 |
| `case7` | 65536 | 65536 |
| `case8` | 131072 | 131072 |

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

TFLOP/s 统计包含 matmul、scale 和 softmax 操作量，保持与 140 TFLOPS 参考脚本一致的计数口径。

默认集合会自动生成 summary TSV。也可以通过 `FA_SUMMARY_TSV` 指定输出路径：

```bash
FA_SUMMARY_TSV=/tmp/fa_summary.tsv python3 run.py --case case1
```

## 注意事项

- `compile.sh` 默认将 `PTO_LIB_PATH` 设为 `/sources/pto-isa`；在本仓工作时建议显式设置 `PTO_LIB_PATH=${git_clone_path}`。
- `--no-build` 只适合单个已选 case，因为 `fa.so` 会按 `FA_Q_ROWS` 重新构建。
- 长序列可能跳过 host 侧 FP32 reference，以避免分配过大的 QK 矩阵；此时会用更宽松阈值对比 NPU fused reference。
