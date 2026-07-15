# 手工调优 kernels（A5）

本目录包含面向 Ascend A5 的手工调优、偏性能的 kernel 示例。

## 示例

- Flash-Attention kernel：[flash_atten](flash_atten/README_zh.md)
- MXFP4 矩阵乘法性能 kernel：[matmul_mxfp4_performance](matmul_mxfp4_performance/README_zh.md)
- MXFP8 矩阵乘法性能 kernel：[matmul_mxfp8_performance](matmul_mxfp8_performance/README_zh.md)

## 通用环境准备

这些示例通常需要先 source CANN 环境后再构建/运行，例如：

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

然后按各示例目录里的 `run.sh`/README 说明执行即可。
