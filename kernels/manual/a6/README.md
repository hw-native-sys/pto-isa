# 手工调优 kernels（A6）

本目录包含面向 Ascend A6（dav-9201）的手工调优（手写、面向性能）kernel 示例。

## 示例

- HiF4（HiFloat4）矩阵乘高性能 kernel：[matmul_hif4_performance](matmul_hif4_performance/README.md)

## 通用环境准备

这些示例在构建/运行前需要启用带 dav-9201（bisheng）编译器的 CANN 环境。例如：

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

然后按各示例目录中的 `run.sh` 说明执行。
