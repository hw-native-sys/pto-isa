# include/pto/npu/

NPU 侧 PTO 指令实现。不同 SoC 代际的指令实现与流水线细节有所不同。

## 按平台选择

| SoC 代际 | 目录 | 说明 |
|----------|------|------|
| Ascend A2/A3 | `a2a3/` | Ascend 910B / 910C 系列实现 |
| Ascend A5 | `a5/` | Ascend 950 系列实现 |
| Kirin 9030 | `kirin9030/` | Kirin 9030 专用实现 |
| Kirin X90 | `kirinX90/` | Kirin X90 专用实现 |

## 目录结构

```
include/pto/npu/
├── a2a3/                      # Ascend A2/A3（910B/910C）系列
│   ├── TAdd.hpp              # TADD 实现
│   ├── TSub.hpp              # TSUB 实现
│   ├── TMul.hpp              # TMUL 实现
│   ├── TMatmul.hpp           # TMATMUL 实现
│   ├── TLoad.hpp             # TLOAD 实现
│   ├── TStore.hpp            # TSTORE 实现
│   └── ...                    # 其他指令实现
│
├── a5/                        # Ascend A5（950）系列
│   ├── TAdd.hpp
│   ├── TSub.hpp
│   ├── TMul.hpp
│   ├── TMatmul.hpp
│   ├── TLoad.hpp
│   ├── TStore.hpp
│   └── ...                    # 其他指令实现（A5 特有指令如 TMATMUL_MX 等）
│
├── kirin9030/                 # Kirin 9030
└── kirinX90/                  # Kirin X90
```

## 选择 SoC 版本

SoC 版本选择由构建系统和测试脚本控制：

- `tests/script/run_st.py` / `tests/script/build_st.py`：通过 `-v a3|a5` 选择
- `tests/npu/<soc>/src/st/CMakeLists.txt`：为每个 SoC 构建对应的 ST targets 和依赖

## A2/A3 与 A5 的主要差异

| 特性 | A2/A3 | A5 |
|------|:------:|:--:|
| 矩阵乘法单元 | CUBE | CUBE（增强） |
| MXFP4/MXFP8 支持 | — | 支持 |
| Vector 指令 | 仿真 | 硬件完整支持 |
| 分形布局 | 仿真 | 完整支持 |
| FP8 类型 | — | 支持 |

## 相关文档

| 文档 | 内容 |
|------|------|
| [include/pto/README_zh.md](../README_zh.md) | PTO 头文件总入口 |
| [docs/getting-started_zh.md](../../../docs/getting-started_zh.md) | 完整上手指南 |
| [include/pto/npu/README_zh.md](./README_zh.md) | 中文版 |
