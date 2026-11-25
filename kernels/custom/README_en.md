# Custom Operators

This directory contains **PTO custom operator development examples**, demonstrating how to implement custom operators from scratch.

If you are new to PTO programming, start from the basics first:

- Getting Started: [docs/en/getting-started.md](../../docs/en/getting-started.md)
- Programming tutorials: [docs/en/coding/tutorial.md](../../docs/en/coding/tutorial.md)
- Add operator example: [demos/baseline/add/README.md](../../demos/baseline/add/README_en.md)

## Examples

- `fused_add_relu_mul/`: Operator fusion example, fusing Add + ReLU + Mul into one kernel, achieving 2-3× speedup.

## How to run

Each subdirectory is a standalone example with its own build/run instructions. See:

- [fused_add_relu_mul/README.md](fused_add_relu_mul/README_en.md)

## Developing Custom Operators

Refer to the `fused_add_relu_mul/` example and follow these steps:

1. Create directory: `mkdir -p kernels/custom/my_operator`
2. Implement kernel: `my_operator_kernel.cpp`
3. Write tests: `main.cpp`
4. Configure build: `CMakeLists.txt`
5. Run and verify: `./run.sh --sim`

For detailed development guides, see:

- [Operator Fusion](../../docs/zh/coding/operator-fusion.md)
- [Performance Optimization](../../docs/zh/coding/opt.md)
