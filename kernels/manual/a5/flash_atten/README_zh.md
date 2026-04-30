# Flash Attention 性能 Kernel（A5）

## 概览

本示例演示如何使用 PTO 在 Ascend A5 平台上实现混合精度的 Flash Attention（FA）算子，包含工程结构、构建与运行方式。

关于 Flash Attention 算子的详细说明、优化策略和流水线编排，请参考通用版本文档：[../../common/flash_atten/README_zh.md](../../common/flash_atten/README_zh.md)

## 支持的 AI 处理器

- Ascend A5

## 目录结构

```
kernels/manual/a5/flash_atten/
├── scripts/
│   ├── gen_data.py                  # 生成输入与 golden 输出
│   ├── generate_cases.py            # 生成测试用例
│   ├── pipeline_log_analysis.py     # 流水线日志分析
│   ├── pipeline_schedule_gen.py     # 流水线调度生成
│   └── run_timeline.sh              # 时间线分析脚本
├── CMakeLists.txt                   # 构建配置
├── fa_performance_kernel.cpp        # Kernel 实现
├── fa_performance_kernel.h          # Kernel 头文件
├── main.cpp                         # Host 侧入口
├── pto_macro_fa_gu.hpp              # FA GU 宏定义
├── pto_macro_fa_softmax.hpp         # FA Softmax 宏定义
├── pto_macro_matmul.hpp             # Matmul 宏定义
└── run.sh                           # 便捷脚本
```

## 构建与运行

1. 配置 Ascend CANN 环境（示例路径）：

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. 运行示例：

```bash
cd ${git_clone_path}/kernels/manual/a5/flash_atten

# 运行默认 case（与 generated_cases.* 中内置集合一致）
bash run.sh -r npu -v Ascend910_9599

# 从内置集合中只运行一个 case
bash run.sh -r npu -v Ascend910_9599 -c case_float_H_128_S0_128_S1_1024

# 提供自定义 case（用分号分隔：HEAD_SIZE,S0,S1,CUBE_S0,TILE_S1）
# TILE_S1：支持 128（=CUBE_S1）、256、512
bash run.sh -r npu -v Ascend910_9599 --cases "128,128,1024,128,128;128,2048,2048,128,512"

# 提供自定义 case，并只运行其中一个
bash run.sh -r npu -v Ascend910_9599 --cases "128,128,1024,128,128;128,512,2048,128,128" \
  -c case_float_H_128_S0_128_S1_1024
```

成功时输出：

```text
test success
```

## 性能

A5 版本的参考性能数据尚未发布。

当前 910B2（A2/A3 代际）Flash Attention 性能参考请见 [../../common/flash_atten/README_zh.md](../../common/flash_atten/README_zh.md)。

## 算子说明

Flash Attention 算子的详细实现说明、数学公式、分块计算策略、流水线编排等内容，请参考：

- [通用 Flash Attention 文档](../../common/flash_atten/README_zh.md)

该文档包含：
1. 计算流程（FlashAttention 2.0）
2. 张量形状（按阶段）
3. 分阶段实现与调参（`compute_qk`、`compute_p`、`compute_pv`、`compute_gu`）
4. 流水线编排（Cube/Vector 并行）
5. 多核切分与负载均衡

## A5 平台特定优化

相比 A2/A3 平台，A5 版本的 Flash Attention kernel 针对 A5 架构特点进行了以下优化：

- 针对 A5 的内存层次结构调整了缓冲区分配策略
- 优化了 Cube 和 Vector 核心之间的同步机制
- 调整了流水线深度参数以适配 A5 的硬件特性
