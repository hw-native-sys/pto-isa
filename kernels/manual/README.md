# Manual kernels

This folder contains **manual (hand-tuned) kernel examples** that use explicit buffering, synchronization, and pipeline control for maximum performance on supported NPUs.

If you are new to PTO programming, start from the ISA and tutorials first:

- Programming tutorials: [docs/coding/tutorial.md](../../docs/coding/tutorial.md)
- Optimization notes: [docs/coding/opt.md](../../docs/coding/opt.md)
- PTO ISA reference: [docs/PTOISA.md](../../docs/PTOISA.md)

## Platforms

- `a2a3/`: Manual kernels for Ascend A2/A3 platforms.

## How to run

Each subdirectory is a standalone example with its own build/run instructions. See:

- [a2a3/README.md](a2a3/README.md)

