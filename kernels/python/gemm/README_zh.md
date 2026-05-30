# PTO-DSL GEMM Kernel

## 概览

本目录包含一个使用 PTO-DSL 编写的 shape-specialized GEMM kernel，以及用于构建、
正确性验证和 benchmark 的 host 侧脚本。`run.py` 会在 `case_builds/<case>/`
下为每个 case 生成专用 kernel，并在记录的构建签名与当前配置不一致时自动重新构建。

## 支持的 AI 处理器

- A2/A3

## 目录结构

```text
kernels/python/gemm/
├── kernels/
│   └── gemm_builder.py      # 生成 PTO IR 的 PTO-DSL 源码
├── compile.sh               # 构建 build_artifacts/gemm.{pto,cpp,so}
├── caller.cpp               # 共享库 launcher 包装层
├── run.py                   # 正确性测试和 benchmark 入口
├── build_artifacts/         # compile.sh 基于 build() 默认参数的输出
└── case_builds/             # run.py 生成的分 case kernel
```

生成产物位于：

```text
build_artifacts/gemm.pto       # compile.sh 生成的 IR
build_artifacts/gemm.cpp       # ptoas 生成的 C++
build_artifacts/gemm.so        # compile.sh 生成的动态库
case_builds/<case>/gemm.pto    # run.py 为某个 case 生成的专用 IR
case_builds/<case>/gemm.cpp
case_builds/<case>/gemm.so
case_builds/<case>/build_config.json
```

## 算子约定

该 kernel 计算：

```text
C = A x B
```

- `A`：`[m, k]`，`float16`，ND 布局
- 逻辑上的 `B`：`[k, n]`，`float16`
- `B` 在 GM 中的存储：`[n, k]`，传入转置后的连续 tensor，并以 `layout="DN"` 读取
- `C`：`[m, n]`，`float16`，ND 布局
- 累加：fp32 ACC tile，store 时转换为 fp16

Python runner 会构造逻辑上的 `B[k, n]`，将 `B.T.contiguous()` 传给 kernel，
并使用逻辑 `B` 参与正确性计算。

## Kernel 结构

`kernels/gemm_builder.py` 会根据选定 shape 和 tiling 生成一个 `Gemm` 符号。
当前实现包含：

- 在全局 `baseM x baseN` 输出 tile 网格上的 persistent 遍历
- 作用于 base tile 的 N 分组 swizzle，用于提高 B tile 的 L2 复用
- A/B 使用双缓冲 L1 MAT tile 暂存
- L0A/L0B 使用 ping-pong tile buffer
- 每个输出 tile 使用一个 fp32 L0C ACC tile
- K panel 宽度为 `baseK * stepKa`
- 转置 B 存储通过 DN GM tensor view 读取

builder 和 runner 会检查以下约束：

- `m % baseM == 0`
- `n % baseN == 0`
- `k % baseK == 0`
- `stepKa == stepKb == 4`
- `(k / baseK) % (stepKa * 2) == 0`
- `1 <= blockDim <= (m / baseM) * (n / baseN)`
- `1 <= swizzleCountN <= n / baseN`

## 运行时调度

Case 只描述 shape 和 base tiling。`run.py` 会在运行时补充：

- `blockDim`：根据所选 SoC 的 `cube_core_cnt` 与 base-tile 总数取较小值
- `swizzleCountN`：根据 L2 容量计算 N 分组宽度，并受配置的目标值上限约束

可以通过 `--soc` 指定 CANN platform config。未指定时，`run.py` 会优先使用 SoC
相关环境变量，否则默认使用 `Ascend910B2`。

## build() 默认参数

- `m = 6144`
- `k = 6144`
- `n = 6144`
- `baseM = 128`
- `baseK = 64`
- `baseN = 256`
- `stepKa = 4`
- `stepKb = 4`
- base-tile 网格足够大时，运行时 `blockDim = cube_core_cnt`

## 依赖条件

在构建或运行前，需要具备：

- Ascend CANN 工具链环境
- `bisheng`
- `ptoas`
- MLIR/PTO Python 模块（`mlir.ir`、`mlir.dialects.pto`）
- `run.py` 需要 `torch` 与 `torch_npu`

`compile.sh` 在 `PATH` 中找不到 `bisheng` 时，会尝试自动加载本地 Ascend 环境；
同时它也支持：

- `PTOAS_ROOT`：将 PTO assembler 的二进制和库目录追加到 `PATH` / `LD_LIBRARY_PATH`
- `PTO_LIB_PATH`：覆盖 `include/` 所使用的仓库根目录
- `NPU_ARCH`：传给 `bisheng` 的目标 NPU 架构，默认 `dav-2201`

## 构建

基于 `build()` 默认参数构建动态库：

```bash
cd ${git_clone_path}/kernels/python/gemm
bash compile.sh
```

指定其他目标架构：

```bash
NPU_ARCH=dav-2301 bash compile.sh
```

预期构建步骤：

1. 由 `kernels/gemm_builder.py` 生成 `build_artifacts/gemm.pto`
2. 将 PTO IR 汇编为 `build_artifacts/gemm.cpp`
3. 将 `caller.cpp` 编译成 `build_artifacts/gemm.so`

## 运行

列出可用 case：

```bash
cd ${git_clone_path}/kernels/python/gemm
python3 run.py --list-cases
```

仅做正确性测试：

```bash
cd ${git_clone_path}/kernels/python/gemm
python3 run.py
```

正确性测试加 PTO benchmark：

```bash
python3 run.py --benchmark
```

使用相同 ABt 输入存储运行可选 Torch-NPU 计时：

```bash
python3 run.py --torch-npu
```

两条 benchmark 路径都分配 `[m, k]` 的 A 和 `[n, k]` 的 B 存储。Torch-NPU 路径
调用 `torch.matmul(a, b.transpose(0, 1))`。计时使用 NPU event，并报告 fp16 输出
的平均时延和 TFLOPS。

运行指定 case：

```bash
python3 run.py --case a2a3_perf_6144
```

运行全部 case：

```bash
python3 run.py --all_cases --benchmark
```

覆盖自动计算的 swizzle 宽度：

```bash
python3 run.py --case a2a3_perf_6144 --swizzle-count-n 5
```

强制重新构建所选 case：

```bash
python3 run.py --case a2a3_perf_6144 --rebuild
```

指定共享库路径：

```bash
python3 run.py --case a2a3_perf_6144 --lib ./build_artifacts/gemm.so
```

`--lib` 只能在只选择一个 case 时使用，因为 runner 仍然使用该 case 的 shape
和运行时 launch 参数。

设备可通过以下环境变量指定：

- `PTODSL_TEST_DEVICE_ID`
- `TASK_DEVICE`

如果两者都未设置，runner 默认使用 `npu:0`。

## 预期输出

正确性模式会打印当前 shape 的 `PASS` / `FAIL`，并最终输出：

```text
Result: ALL PASS
```

开启 benchmark 后，还会额外打印平均时延和 TFLOPS。
