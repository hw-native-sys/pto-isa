# include/

Public C/C++ headers for PTO Tile Lib (primarily header-only, template-based). Upper-layer frameworks or operator code can include these headers to emit PTO ISA Tile-level operations.

## Quick Start

Include the unified entry header:

```cpp
#include <pto/pto-inst.hpp>
```

`pto/pto-inst.hpp` selects the appropriate backend (CPU simulation/stub or NPU implementation) based on build configuration. See [include/pto/README.md](pto/README.md) for details.

## Layout

- `include/pto/`: Public PTO ISA API and backend implementations (common / cpu / npu)

## Related Docs

- [ISA guide](../docs/README.md)
- [Getting started](../docs/getting-started.md)

## PTO Instruction Implementation Status (CPU / A2 / A3 / A5)

This table tracks per-instruction backend availability:

- **CPU**: `__CPU_SIM` (CPU simulation backend).
- **A2 (Ascend 910B) / A3 (Ascend 910C)**: share the `include/pto/npu/a2a3/` implementation today (so the status is identical for both columns).
- **A5 (Ascend 950)**: uses the `include/pto/npu/a5/` implementation.
- **TODO** means the instruction is part of the public API but the backend implementation is not available yet.

| Instruction | CPU | A2 | A3 | A5 |
|---|---:|---:|---:|---:|
| [`MGATHER`](../docs/isa/MGATHER.md) | Yes | TODO | TODO | TODO |
| [`MSCATTER`](../docs/isa/MSCATTER.md) | Yes | TODO | TODO | TODO |
| [`TABS`](../docs/isa/TABS.md) | Yes | Yes | Yes | Yes |
| [`TADD`](../docs/isa/TADD.md) | Yes | Yes | Yes | Yes |
| [`TADDC`](../docs/isa/TADDC.md) | Yes | TODO | TODO | TODO |
| [`TADDS`](../docs/isa/TADDS.md) | Yes | Yes | Yes | Yes |
| [`TADDSC`](../docs/isa/TADDSC.md) | Yes | TODO | TODO | TODO |
| [`TAND`](../docs/isa/TAND.md) | Yes | Yes | Yes | Yes |
| [`TANDS`](../docs/isa/TANDS.md) | Yes | Yes | Yes | Yes |
| [`TASSIGN`](../docs/isa/TASSIGN.md) | Yes | Yes | Yes | Yes |
| [`TCI`](../docs/isa/TCI.md) | Yes | Yes | Yes | Yes |
| [`TCMP`](../docs/isa/TCMP.md) | Yes | Yes | Yes | Yes |
| [`TCMPS`](../docs/isa/TCMPS.md) | Yes | Yes | Yes | Yes |
| [`TCOLEXPAND`](../docs/isa/TCOLEXPAND.md) | Yes | TODO | TODO | TODO |
| [`TCOLEXPANDADD`]() | TODO | Yes | Yes | Yes |
| [`TCOLEXPANDDIV`]() | TODO | Yes | Yes | Yes |
| [`TCOLEXPANDEXPDIF`]() | TODO | Yes | Yes | Yes |
| [`TCOLEXPANDMAX`]() | TODO | Yes | Yes | Yes |
| [`TCOLEXPANDMIN`]() | TODO | Yes | Yes | Yes |
| [`TCOLEXPANDMUL`]() | TODO | Yes | Yes | Yes |
| [`TCOLEXPANDSUB`]() | TODO | Yes | Yes | Yes |
| [`TCOLMAX`](../docs/isa/TCOLMAX.md) | Yes | Yes | Yes | Yes |
| [`TCOLMIN`](../docs/isa/TCOLMIN.md) | Yes | Yes | Yes | Yes |
| [`TCOLSUM`](../docs/isa/TCOLSUM.md) | Yes | Yes | Yes | Yes |
| [`TCOLPROD`](../docs/isa/TCOLPROD.md) | TODO | Yes | Yes | Yes |
| [`TCVT`](../docs/isa/TCVT.md) | Yes | Yes | Yes | Yes |
| [`TDIV`](../docs/isa/TDIV.md) | Yes | Yes | Yes | Yes |
| [`TDIVS`](../docs/isa/TDIVS.md) | Yes | Yes | Yes | Yes |
| [`TEXP`](../docs/isa/TEXP.md) | Yes | Yes | Yes | Yes |
| [`TEXPANDS`](../docs/isa/TEXPANDS.md) | Yes | Yes | Yes | Yes |
| [`TEXTRACT`](../docs/isa/TEXTRACT.md) | Yes | Yes | Yes | Yes |
| [`TFILLPAD`](../docs/isa/TFILLPAD.md) | Yes | Yes | Yes | Yes |
| [`TGATHER`](../docs/isa/TGATHER.md) | Yes | Yes | Yes | Yes |
| [`TGATHERB`](../docs/isa/TGATHERB.md) | Yes | Yes | Yes | Yes |
| [`TLOAD`](../docs/isa/TLOAD.md) | Yes | Yes | Yes | Yes |
| [`TLOG`](../docs/isa/TLOG.md) | Yes | Yes | Yes | Yes |
| [`TLRELU`](../docs/isa/TLRELU.md) | Yes | Yes | Yes | Yes |
| [`TMATMUL`](../docs/isa/TMATMUL.md) | Yes | Yes | Yes | Yes |
| [`TMATMUL_ACC`](../docs/isa/TMATMUL_ACC.md) | Yes | Yes | Yes | Yes |
| [`TMATMUL_BIAS`](../docs/isa/TMATMUL_BIAS.md) | Yes | Yes | Yes | Yes |
| [`TMATMUL_MX`](../docs/isa/TMATMUL_MX.md) | Yes | Yes | Yes | Yes |
| [`TMAX`](../docs/isa/TMAX.md) | Yes | Yes | Yes | Yes |
| [`TMAXS`](../docs/isa/TMAXS.md) | Yes | Yes | Yes | Yes |
| [`TMIN`](../docs/isa/TMIN.md) | Yes | Yes | Yes | Yes |
| [`TMINS`](../docs/isa/TMINS.md) | Yes | Yes | Yes | Yes |
| [`TMOV`](../docs/isa/TMOV.md) | Yes | Yes | Yes | Yes |
| [`TMOV_FP`](../docs/isa/TMOV_FP.md) | TODO | TODO | TODO | TODO |
| [`TMRGSORT`](../docs/isa/TMRGSORT.md) | Yes | Yes | Yes | Yes |
| [`TMUL`](../docs/isa/TMUL.md) | Yes | Yes | Yes | Yes |
| [`TMULS`](../docs/isa/TMULS.md) | Yes | Yes | Yes | Yes |
| [`TNEG`](../docs/isa/TNEG.md) | Yes | Yes | Yes | Yes |
| [`TNOT`](../docs/isa/TNOT.md) | Yes | Yes | Yes | Yes |
| [`TOR`](../docs/isa/TOR.md) | Yes | Yes | Yes | Yes |
| [`TORS`](../docs/isa/TORS.md) | Yes | Yes | Yes | Yes |
| [`TPARTADD`](../docs/isa/TPARTADD.md) | Yes | Yes | Yes | Yes |
| [`TPARTMAX`](../docs/isa/TPARTMAX.md) | Yes | Yes | Yes | Yes |
| [`TPARTMIN`](../docs/isa/TPARTMIN.md) | Yes | Yes | Yes | Yes |
| [`TPARTMUL`]() | TODO | Yes | Yes | Yes |
| [`TPRELU`](../docs/isa/TPRELU.md) | Yes | Yes | Yes | Yes |
| [`TPREFETCH`]() | TODO | Yes | Yes | Yes |
| [`TPRINT`]() | TODO | Yes | Yes | Yes |
| [`TRECIP`](../docs/isa/TRECIP.md) | Yes | Yes | Yes | Yes |
| [`TRELU`](../docs/isa/TRELU.md) | Yes | Yes | Yes | Yes |
| [`TREM`](../docs/isa/TREM.md) | Yes | Yes | Yes | Yes |
| [`TREMS`](../docs/isa/TREMS.md) | Yes | Yes | Yes | Yes |
| [`TFMOD`](../docs/isa/TFMOD.md) | TODO | Yes | Yes | Yes |
| [`TFMODS`](../docs/isa/TFMODS.md) | TODO | Yes | Yes | Yes |
| [`TRESHAPE`](../docs/isa/TRESHAPE.md) | Yes | Yes | Yes | Yes |
| [`TROWEXPAND`](../docs/isa/TROWEXPAND.md) | Yes | Yes | Yes | Yes |
| [`TROWEXPANDADD `]() | TODO | Yes | Yes | Yes |
| [`TROWEXPANDDIV`](../docs/isa/TROWEXPANDDIV.md) | Yes | Yes | Yes | Yes |
| [`TROWEXPANDEXPDIF`]() | TODO | Yes | Yes | Yes |
| [`TROWEXPANDMAX`](../docs/isa/TROWEXPANDMUL.md) | Yes | Yes | Yes | Yes |
| [`TROWEXPANDMIN`]() | TODO | Yes | Yes | Yes |
| [`TROWEXPANDMUL`]() | TODO | Yes | Yes | Yes |
| [`TROWEXPANDSUB`](../docs/isa/TROWEXPANDSUB.md) | Yes | Yes | Yes | Yes |
| [`TROWMAX`](../docs/isa/TROWMAX.md) | Yes | Yes | Yes | Yes |
| [`TROWMIN`](../docs/isa/TROWMIN.md) | Yes | Yes | Yes | Yes |
| [`TROWSUM`](../docs/isa/TROWSUM.md) | Yes | Yes | Yes | Yes |
| [`TROWPROD`](../docs/isa/TROWPROD.md) | TODO | Yes | Yes | TODO |
| [`TRSQRT`](../docs/isa/TRSQRT.md) | Yes | Yes | Yes | Yes |
| [`TSCATTER`](../docs/isa/TSCATTER.md) | Yes | Yes | Yes | Yes |
| [`TSEL`](../docs/isa/TSEL.md) | Yes | Yes | Yes | Yes |
| [`TSELS`](../docs/isa/TSELS.md) | Yes | Yes | Yes | Yes |
| [`TSHL`](../docs/isa/TSHL.md) | Yes | Yes | Yes | Yes |
| [`TSHLS`](../docs/isa/TSHLS.md) | Yes | Yes | Yes | Yes |
| [`TSHR`](../docs/isa/TSHR.md) | Yes | Yes | Yes | Yes |
| [`TSHRS`](../docs/isa/TSHRS.md) | Yes | Yes | Yes | Yes |
| [`TSORT32`](../docs/isa/TSORT32.md) | Yes | Yes | Yes | Yes |
| [`TSQRT`](../docs/isa/TSQRT.md) | Yes | Yes | Yes | Yes |
| [`TSTORE`](../docs/isa/TSTORE.md) | Yes | Yes | Yes | Yes |
| [`TSTORE_FP`](../docs/isa/TSTORE_FP.md) | TODO | TODO | TODO | TODO |
| [`TSUB`](../docs/isa/TSUB.md) | Yes | Yes | Yes | Yes |
| [`TSUBC`](../docs/isa/TSUBC.md) | Yes | TODO | TODO | TODO |
| [`TSUBS`](../docs/isa/TSUBS.md) | Yes | Yes | Yes | Yes |
| [`TSUBSC`](../docs/isa/TSUBSC.md) | Yes | TODO | TODO | TODO |
| [`TSYNC`](../docs/isa/TSYNC.md) | TODO | Yes | Yes | Yes |
| [`TTRANS`](../docs/isa/TTRANS.md) | Yes | Yes | Yes | Yes |
| [`TTRI`]() | TODO | Yes | Yes | Yes |
| [`TXOR`](../docs/isa/TXOR.md) | Yes | Yes | Yes | Yes |
| [`TXORS`](../docs/isa/TXORS.md) | Yes | Yes | Yes | Yes |
