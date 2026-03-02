# Manual kernels (A2/A3)

This directory contains manual, performance-oriented kernel examples targeting Ascend A2/A3.

## Examples

- GEMM performance kernel: [gemm_performance/README.md](gemm_performance/README.md)
- Flash-Attention kernel: [flash_atten/README.md](flash_atten/README.md)
- TOPK performance kernel: [topk/README.md](topk/README.md)

## Common setup

These examples typically require a CANN environment to be sourced before building/running. For example:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

Then follow the `run.sh` usage documented in each example directory.

