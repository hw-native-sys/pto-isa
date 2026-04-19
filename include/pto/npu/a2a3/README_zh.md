# include/pto/npu/a2a3/

Ascend A2/A3（Ascend 910B / 910C）系列 PTO 指令实现头文件。

## 主要文件

```
include/pto/npu/a2a3/
├── TAdd.hpp            # TADD 逐元素加法
├── TSub.hpp            # TSUB 逐元素减法
├── TMul.hpp            # TMUL 逐元素乘法
├── TDiv.hpp            # TDIV 逐元素除法
├── TMatmul.hpp         # TMATMUL 矩阵乘法（硬件 CUBE）
├── TLoad.hpp           # TLOAD GM → tile buffer
├── TStore.hpp          # TSTORE tile buffer → GM
├── TAssign.hpp         # TASSIGN 资源绑定
├── TSync.hpp           # TSYNC 同步
├── ...                  # 其他指令实现（Reduce、Expand、Layout 等）
```

## 与 A5 的主要差异

| 特性 | A2/A3 | A5 |
|------|:------:|:--:|
| 矩阵乘法 | 硬件 CUBE | 增强版 CUBE |
| MXFP4 / MXFP8 | 不支持 | 支持 |
| Vector 指令 | 仿真 | 硬件完整支持 |
| 分形布局 | 仿真 | 完整支持 |
| FP8 类型 | 不支持 | 支持 |
| Tile 指令 | 完整 | 完整 |

## 相关内容

| 文档 | 内容 |
|------|------|
| [docs/isa/](../../docs/isa/) | ISA 语义与示例 |
| [tests/npu/a2a3/src/st/](../../tests/npu/a2a3/src/st/) | A2/A3 NPU ST 测试 |
| [include/pto/npu/](../README_zh.md) | NPU 实现总入口 |
