# Manual kernels (A5)

This directory contains manual, performance-oriented kernel examples targeting Ascend A5.

## Examples

- Flash-Attention kernel: [flash_atten](flash_atten/README_en.md)
- Sparse FlashMLA (BF16 compute, FP4 cache): [sparseFlashMLA](sparseFlashMLA/README.md)
- MXFP4 matrix multiplication performance kernel: [matmul_mxfp4_performance](matmul_mxfp4_performance/README_en.md)
- MXFP8 matrix multiplication performance kernel: [matmul_mxfp8_performance](matmul_mxfp8_performance/README_en.md)

## Common setup

These examples typically require a CANN environment to be sourced before building/running. For example:

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

Then follow the `run.sh` usage documented in each example directory.
