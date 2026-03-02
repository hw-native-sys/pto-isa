# 手工调优 kernels（A2/A3）

本目录包含面向 Ascend A2/A3 的手工调优、偏性能的 kernel 示例。

## 示例

- GEMM 性能 kernel：[gemm_performance/README.md](gemm_performance/README.md)
- Flash-Attention kernel：[flash_atten/README.md](flash_atten/README.md)
- TOPK 性能 kernel：[topk/README.md](topk/README.md)

## 通用环境准备

这些示例通常需要先 source CANN 环境后再构建/运行，例如：

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

然后按各示例目录里的 `run.sh`/README 说明执行即可。
