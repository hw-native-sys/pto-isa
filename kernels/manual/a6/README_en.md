# Manual kernels (A6)

This directory contains manual, performance-oriented kernel examples targeting Ascend A6 (dav-9201).

## Examples

- HiF4 (HiFloat4) matrix multiplication performance kernel: [matmul_hif4_performance](matmul_hif4_performance/README_en.md)

## Common setup

These examples require a CANN environment with the dav-9201 (bisheng) compiler to be sourced before building/running. For example:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

Then follow the `run.sh` usage documented in each example directory.
