# Manual kernels (A2/A3)

This directory contains manual, performance-oriented kernel examples targeting Ascend A2/A3.

## Examples

- GEMM performance kernel: [gemm_performance/README.md](gemm_performance/README.md)
- AllGather + GEMM fusion: [allgather_gemm/README.md](allgather_gemm/README.md)
- Flash-Attention kernel: [../common/flash_atten/README.md](../common/flash_atten/README.md)
- TOPK performance kernel: [topk/README.md](topk/README.md)
- TGET bandwidth comparison kernel: [tget_bandwidth/README.md](tget_bandwidth/README.md)

## Common setup

These examples typically require a CANN environment to be sourced before building/running. For example:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

Then follow the `run.sh` usage documented in each example directory.
