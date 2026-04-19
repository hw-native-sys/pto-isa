# PTO AUTO Mode

This directory contains detailed documentation for PTO AUTO Mode, helping developers understand and use Auto Mode for PTO programming.

## Choose by Task

| Your need | Start here |
|-----------|-----------|
| What is Auto Mode | [Auto Mode Overview](Auto_Mode_Overview.md) |
| Kernel development rules and limitations | [Kernel Developer Rules](Kernel_Developer_Rules_And_Limitations.md) |
| Library development rules and limitations | [Library Developer Rules](Library_Developer_Rules_And_Limitations.md) |
| Code examples | [Examples](Examples.md) |

## What is PTO AUTO

PTO AUTO is a programming mode that provides two major benefits:

1. **Simplifies development** while enabling necessary optimizations.
2. **Ensures cross-generation compatibility** across Ascend hardware.

In AUTO mode, kernel developers **do not need to** manually assign tile memory (`TASSIGN`) or manage synchronization between pipes (`set_flag`/`wait_flag`). The compiler handles these automatically while maintaining good performance.

## Auto vs Manual Mode Comparison

| Aspect | Auto Mode | Manual Mode |
|--------|-----------|-------------|
| Tile address allocation | Compiler automatic | Author explicit `TASSIGN` |
| Synchronization management | Compiler automatic | Author explicit `set_flag`/`wait_flag` |
| Data movement | Compiler automatic `TLOAD`/`TSTORE` | Author explicit `TLOAD`/`TSTORE` |
| Performance | Near hand-tuned | Highest performance |
| Development difficulty | Low | High |
| Cross-generation compatibility | Best | Requires per-generation tuning |

> Note: auto mode currently only supports the compiler `-O2` option.

## Document Index

| Document | Content |
|----------|---------|
| [Auto Mode Overview](Auto_Mode_Overview.md) | Core concepts, compiler features, comparison with Manual mode |
| [Kernel Developer Rules](Kernel_Developer_Rules_And_Limitations.md) | Programming rules and limitations for kernel developers in Auto Mode |
| [Library Developer Rules](Library_Developer_Rules_And_Limitations.md) | Programming rules and limitations for library developers in Auto Mode |
| [Examples](Examples.md) | Auto Mode code examples |

## Related Docs

| Document | Content |
|----------|---------|
| [docs/README.md](../README.md) | Documentation hub |
| [docs/coding/README.md](../coding/README.md) | Programming model docs entry |
| [docs/isa/README.md](../isa/README.md) | ISA instruction reference |
