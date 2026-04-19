# docs/coding/

This directory describes the **PTO Tile Library programming model as seen from C++** (Tiles, GlobalTensor, events, scalar parameters) and provides guidance for extending the library.

If you are looking for the **ISA reference**, start from [docs/isa/README.md](../isa/README.md).

## Choose by Task

| Your goal | Start here |
|-----------|-----------|
| First time learning PTO | [Hands-on tutorial](tutorial.md) |
| Understanding Tile abstraction and valid regions | [Tile programming model](Tile.md) |
| Understanding global memory tensors | [GlobalTensor](GlobalTensor.md) |
| Understanding events and synchronization | [Events and synchronization](Event.md) |
| Understanding Auto Mode | [Auto Mode overview](../auto_mode/Auto_Mode_Overview.md) |
| Understanding the compilation pipeline | [Compilation process](compilation-process.md) |
| Finding performance bottlenecks | [Performance optimization](opt.md) |
| Understanding operator fusion | [Operator fusion](operator-fusion.md) |
| Debugging and error handling | [Debugging guide](debug.md), [Error codes](error-codes.md) |
| Multi-core programming | [Multi-core programming](multi-core-programming.md) |
| Memory optimization | [Memory optimization](memory-optimization.md) |
| PTO vs other frameworks | [PTO comparison](pto-comparison.md) |

## Document Index

### Foundations

- [Hands-on tutorial (write your first kernel)](tutorial.md) — Step-by-step guide to your first PTO kernel
- [More tutorial examples](tutorials/README.md) — Additional getting-started examples
- [Tile abstraction and layout/valid-region rules](Tile.md) — Tile model in depth
- [Global memory tensors (shape/stride/layout)](GlobalTensor.md) — GM tensor types
- [Events and synchronization model](Event.md) — Event recording, waiting, and synchronization
- [Scalar values, type mnemonics, and enums](Scalar.md) — Scalar parameters and type aliases
- [Auto Mode overview](../auto_mode/Auto_Mode_Overview.md) — Compiler-driven resource and sync management

### Build and Compilation

- [Compilation process](compilation-process.md) — Full pipeline from source to binary
- [CPU Simulator](cpu_sim.md) — Running PTO code on CPU

### Debugging and Error Handling

- [Debugging and assertion lookup](debug.md) — Debugging strategies
- [Error codes](error-codes.md) — Error code reference

### Advanced Topics

- [Performance optimization](opt.md) — Performance analysis and tuning guidance
- [Performance best practices](performance-best-practices.md) — Best practices and performance tips
- [Operator fusion](operator-fusion.md) — Tensor fusion techniques
- [Memory optimization](memory-optimization.md) — Memory optimization strategies
- [Pipeline parallelism](pipeline-parallel.md) — Pipeline-parallel programming
- [Multi-core programming](multi-core-programming.md) — Multi-core programming model
- [Version compatibility](version-compatibility.md) — Compatibility and migration
- [Framework integration](framework-integration.md) — Integration with PyTorch and other frameworks

### Reference

- [PTO comparison with other frameworks](pto-comparison.md) — PTO vs TVM, CUTLASS, etc.
- [References](references.md) — Bibliography and further reading
- [ConvTile](ConvTile.md) — Conv2D tile optimization

## Related Docs

| Document | Content |
|----------|---------|
| [PTO abstract machine model](../machine/README.md) | Abstract execution model |
| [docs/README.md](../README.md) | Documentation hub |
| [docs/isa/README.md](../isa/README.md) | ISA instruction reference |
