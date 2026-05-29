# Manual kernels (A5)

This directory contains manual, performance-oriented kernel examples targeting Ascend A5.

## Examples

- Flash-Attention kernel: [flash_atten](flash_atten/README.md)
- MXFP4 matrix multiplication performance kernel: [matmul_mxfp4_performance](matmul_mxfp4_performance/README.md)
- MXFP8 matrix multiplication performance kernel: [matmul_mxfp8_performance](matmul_mxfp8_performance/README.md)
- MHC (Multi-Head Computation) kernels: [mhc](mhc/README.md)

## Common setup

These examples typically require a CANN environment to be sourced before building/running. For example:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

Then follow the `run.sh` usage documented in each example directory.





