# include/pto/npu/a5/

Ascend A5（Ascend 950）系列 PTO 指令实现头文件。

## 主要文件

```
include/pto/npu/a5/
├── TAdd.hpp            # TADD 逐元素加法
├── TSub.hpp            # TSUB 逐元素减法
├── TMul.hpp            # TMUL 逐元素乘法
├── TDiv.hpp            # TDIV 逐元素除法
├── TMatmul.hpp         # TMATMUL 矩阵乘法（增强版 CUBE）
├── TMatmulMx.hpp       # TMATMUL_MX 带 MX 格式的矩阵乘法
├── TLoad.hpp           # TLOAD GM → tile buffer
├── TStore.hpp          # TSTORE tile buffer → GM
├── TAssign.hpp         # TASSIGN 资源绑定
├── TSync.hpp           # TSYNC 同步
├── ...                  # 其他指令实现（Reduce、Expand、Layout 等）
```

## A5 特有功能

| 功能 | 说明 |
|------|------|
| MXFP8 / MXFP4 | A5 硬件支持的混合精度矩阵乘法 |
| 分形布局 | NZ / ZN / FR / RN 分形布局完整支持 |
| Vector 硬件 | `pto.v*` 指令在 A5 上硬件完整支持（非仿真） |
| FP8 类型 | `f8e4m3`、`f8e5m2` 数据类型支持 |

## 与 A2/A3 的主要差异

| 特性 | A2/A3 | A5 |
|------|:------:|:--:|
| 矩阵乘法 | 硬件 CUBE | 增强版 CUBE |
| MXFP4 / MXFP8 | 不支持 | 支持 |
| Vector 指令 | 仿真 | 硬件完整支持 |
| 分形布局 | 仿真 | 完整支持 |
| FP8 类型 | 不支持 | 支持 |

## 相关内容

| 文档 | 内容 |
|------|------|
| [docs/isa/](../../docs/isa/) | ISA 语义与示例 |
| [tests/npu/a5/src/st/](../../tests/npu/a5/src/st/) | A5 NPU ST 测试 |
| [include/pto/npu/](../README_zh.md) | NPU 实现总入口 |
