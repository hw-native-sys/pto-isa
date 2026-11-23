# Manual kernels

This folder contains **manual (hand-tuned) kernel examples** that use explicit buffering, synchronization, and pipeline control for maximum performance on supported NPUs.

If you are new to PTO programming, start from the ISA and tutorials first:

- Programming tutorials: [docs/en/coding/tutorial.md](../../docs/en/coding/tutorial.md)
- Optimization notes: [docs/en/coding/opt.md](../../docs/en/coding/opt.md)
- PTO ISA reference: [docs/en/PTOISA.md](../../docs/en/PTOISA.md)

## Platforms

- `a2a3/`: Manual kernels for Ascend A2/A3 platforms.
- `a5/`: Manual kernels for Ascend A5 platforms.
- `a6/`: Manual kernels for Ascend A6 platforms.
- `common/`: Cross-platform manual kernels (shared examples).

## How to run

Each subdirectory is a standalone example with its own build/run instructions. See:

- [a2a3/README.md](a2a3/README_en.md)
- [a5/README.md](a5/README_en.md)
- [a6/README.md](a6/README_en.md)
- [common/flash_atten/README.md](common/flash_atten/README_en.md)
