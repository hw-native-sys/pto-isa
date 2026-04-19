# tests/

PTO Tile Library 的测试与示例，覆盖 CPU 仿真与 NPU（`sim` 和板上 `npu` 两种模式）。

## 测试入口

| 场景 | 命令 |
|------|------|
| CPU Simulator 全量运行 | `python3 tests/run_cpu.py --clean --verbose` |
| GEMM demo | `python3 tests/run_cpu.py --demo gemm --verbose` |
| Flash Attention demo | `python3 tests/run_cpu.py --demo flash_attn --verbose` |
| 单个 ST 用例（NPU） | `python3 tests/script/run_st.py -r [sim\|npu] -v [a3\|a5] -t [TEST_CASE] -g [GTEST_FILTER_CASE]` |
| 一键脚本 | `./build.sh --run_all --a3 --sim` |

## 目录结构

```
tests/
├── script/                     # 测试入口脚本（推荐从此入口）
│   ├── run_st.py              # 构建并运行 NPU ST
│   ├── build_st.py            # 仅构建 NPU ST
│   ├── all_cpu_tests.py       # 批量构建并运行 CPU ST 套件
│   └── README.md              # 脚本使用说明
│
├── cpu/                        # CPU 侧 ST 测试（gtest + CMake）
│   └── st/                    # CPU ST 工程与 testcase 数据生成脚本
│
├── npu/                        # 按 SoC 拆分的 NPU 侧 ST 测试
│   ├── a2a3/
│   │   ├── src/st/           # A2/A3 计算 ST
│   │   └── comm/st/           # A2/A3 通信 ST
│   └── a5/
│       ├── src/st/            # A5 计算 ST
│       └── comm/st/           # A5 通信 ST
│
├── run_st.sh                   # NPU ST 一键运行脚本
└── run_comm_test.sh           # 通信 ST 一键运行脚本
```

## 同步与异步通信指令测试

通信测试验证多卡间的 PTO 通信原语（Put / Get / Broadcast / Gather / Scatter / Reduce / Notify / Wait / Test），基于 MPI + HCCL 实现。

通信测试分为**同步指令**和**异步指令**两类：

| 类型 | 测试用例示例 | CANN 版本要求 |
|------|-------------|--------------|
| 同步指令 | `tput`、`tget`、`treduce`、`tbroadcast` 等 | CANN 8.x 及以上 |
| 异步指令 | `tput_async`、`tget_async` | **CANN 9.0 及以上** |

> 异步指令依赖 CANN 9.0 引入的 SDMA opapi 接口，在低版本 CANN 上会因符号缺失而运行失败。`run_comm_test.sh` **默认不包含异步指令测试**，需通过 `-a` 参数显式启用。

### 快速开始

```bash
# 8 卡全量测试（默认 A2/A3，不含异步指令）
./run_comm_test.sh

# 包含异步指令测试（需 CANN 9.0+）
./run_comm_test.sh -a

# 指定 A5 SoC，2 卡
./run_comm_test.sh -v a5 -n 2

# 开启 debug 日志
./run_comm_test.sh -d -t tput
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-n` | 可用 NPU 数量：2、4 或 8 | 8 |
| `-v` | SoC 版本：`a3`（Ascend910B）或 `a5`（Ascend910_9599） | a3 |
| `-t` | 指定测试用例（可多次使用） | 全部 |
| `-a` | 包含异步指令测试（需 CANN 9.0+） | 关闭 |
| `-d` | 开启调试模式 | 关闭 |

## 建议阅读顺序

| 顺序 | 文档 |
|------|------|
| 1（推荐先学） | [docs/getting-started_zh.md](../docs/getting-started_zh.md) |
| 2 | [docs/coding/tutorial_zh.md](../docs/coding/tutorial_zh.md) |
| 3 | 本页 |

更完整的环境配置与依赖说明请参见：
- [docs/getting-started_zh.md](../docs/getting-started_zh.md)
- [tests/README_zh.md](./README_zh.md) — 中文版
