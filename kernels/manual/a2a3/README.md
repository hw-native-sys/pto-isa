# 手工调优 kernels（A2/A3）

本目录包含面向 Ascend A2/A3 的手工调优、偏性能的 kernel 示例。

## 示例

- GEMM 性能 kernel：[gemm_performance/README.md](gemm_performance/README_en.md)
- AllGather + GEMM 融合：[allgather_gemm/README.md](allgather_gemm/README_en.md)
- Flash-Attention kernel：[../common/flash_atten/README.md](../common/flash_atten/README_en.md)
- TOPK 性能 kernel：[topk/README.md](topk/README_en.md)
- TGET 带宽对比 kernel：[tget_bandwidth/README.md](tget_bandwidth/README.md)

## 通用环境准备

这些示例通常需要先 source CANN 环境后再构建/运行，例如：

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

然后按各示例目录里的 `run.sh`/README 说明执行即可。
