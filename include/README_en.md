# include/

Public C/C++ headers for PTO Tile Lib (primarily header-only, template-based). Upper-layer frameworks or operator code can include these headers to emit PTO ISA Tile-level operations.

## Quick Start

Include the unified entry header:

```cpp
#include <pto/pto-inst.hpp>
```

`pto/pto-inst.hpp` selects the appropriate backend (CPU simulation/stub or NPU implementation) based on build configuration. See [include/pto/README_en.md](pto/README_en.md) for details.

## Layout

- `include/pto/`: Public PTO ISA API and backend implementations (common / cpu / npu / comm)

## Related Docs

- [ISA guide](../docs/en/README.md)
- [Getting started](../docs/en/getting-started.md)

## PTO Instruction Implementation Status (CPU / Costmodel / A2 / A3 / A5 / Kirin)

This table tracks per-instruction backend availability:

- **CPU**: `__CPU_SIM` (CPU simulation backend). More information about this backend can be found in [docs/en/coding/cpu_sim.md](../docs/en/coding/cpu_sim.md)
- **Costmodel**: `__COSTMODEL` (A2 / A3 cost model backend, including `stub` and `fit` paths; if either path supports an instruction, it is marked as supported).
- **A2 (Ascend 910B) / A3 (Ascend 910C)**: share the `include/pto/npu/a2a3/` implementation today (so the status is identical for both columns).
- **A5 (Ascend 950)**: uses the `include/pto/npu/a5/` implementation.
- **Kirin**: uses the `include/pto/npu/kirin9030/` implementation.

| Instruction | CPU | Costmodel | A2 | A3 | A5 | Kirin |
|---|---:|---:|---:|---:|---:|---:|
| [`MGATHER`](../docs/en/isa/MGATHER.md) | Yes | TODO | TODO | TODO | Yes | TODO |
| [`MSCATTER`](../docs/en/isa/MSCATTER.md) | Yes | TODO | TODO | TODO | Yes | TODO |
| [`TABS`](../docs/en/isa/TABS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TADD`](../docs/en/isa/TADD.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TADDS`](../docs/en/isa/TADDS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TAND`](../docs/en/isa/TAND.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TANDS`](../docs/en/isa/TANDS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TASSIGN`](../docs/en/isa/TASSIGN.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TAXPY`](../docs/en/isa/TAXPY.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TBROADCAST`](../docs/en/isa/comm/TBROADCAST.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TCI`](../docs/en/isa/TCI.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TCMP`](../docs/en/isa/TCMP.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TCMPS`](../docs/en/isa/TCMPS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TCOLARGMAX`](../docs/en/isa/TCOLARGMAX.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TCOLARGMIN`](../docs/en/isa/TCOLARGMIN.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TCOLEXPAND`](../docs/en/isa/TCOLEXPAND.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TCOLEXPANDADD`](../docs/en/isa/TCOLEXPANDADD.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLEXPANDDIV`](../docs/en/isa/TCOLEXPANDDIV.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLEXPANDEXPDIF`](../docs/en/isa/TCOLEXPANDEXPDIF.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLEXPANDMAX`](../docs/en/isa/TCOLEXPANDMAX.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLEXPANDMIN`](../docs/en/isa/TCOLEXPANDMIN.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLEXPANDMUL`](../docs/en/isa/TCOLEXPANDMUL.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLEXPANDSUB`](../docs/en/isa/TCOLEXPANDSUB.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLMAX`](../docs/en/isa/TCOLMAX.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TCOLMIN`](../docs/en/isa/TCOLMIN.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TCOLPROD`](../docs/en/isa/TCOLPROD.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TCOLSUM`](../docs/en/isa/TCOLSUM.md) | Yes | Yes | Yes | Yes | Yes | TODO |
| [`TConcat`](../docs/en/isa/TCONCAT.md) | TODO | TODO | TODO | TODO | Yes | TODO |
| [`TCVT`](../docs/en/isa/TCVT.md) | Yes | Yes | Yes | Yes | Yes | TODO |
| [`TDEQUANT`](../docs/en/isa/TDEQUANT.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TDIV`](../docs/en/isa/TDIV.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TDIVS`](../docs/en/isa/TDIVS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TEXP`](../docs/en/isa/TEXP.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TEXPANDS`](../docs/en/isa/TEXPANDS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TEXTRACT`](../docs/en/isa/TEXTRACT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TEXTRACT_FP`](../docs/en/isa/TEXTRACT_FP.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TFILLPAD`](../docs/en/isa/TFILLPAD.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TFILLPAD_EXPAND`](../docs/en/isa/TFILLPAD_EXPAND.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TFILLPAD_INPLACE`](../docs/en/isa/TFILLPAD_INPLACE.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TFMOD`](../docs/en/isa/TFMOD.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TFMODS`](../docs/en/isa/TFMODS.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TGATHER`](../docs/en/isa/TGATHER.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TGATHERB`](../docs/en/isa/TGATHERB.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TGEMV`](../docs/en/isa/TGEMV.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TGEMV_ACC`](../docs/en/isa/TGEMV_ACC.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TGEMV_BIAS`](../docs/en/isa/TGEMV_BIAS.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TGEMV_MX`](../docs/en/isa/TGEMV_MX.md) | TODO | TODO | TODO | TODO | Yes | TODO |
| [`TGET`](../docs/en/isa/comm/TGET.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TGET_ASYNC`](../docs/en/isa/comm/TGET_ASYNC.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`THISTOGRAM`](../docs/en/isa/THISTOGRAM.md) | Yes | TODO | No | No | Yes | Yes |
| [`TIMG2COL`](../docs/en/isa/TIMG2COL.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TINSERT`](../docs/en/isa/TINSERT.md) | TODO | TODO | TODO | TODO | Yes | TODO |
| [`TINSERT_FP`](../docs/en/isa/TINSERT_FP.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TLOAD`](../docs/en/isa/TLOAD.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TLOG`](../docs/en/isa/TLOG.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TLRELU`](../docs/en/isa/TLRELU.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TMADD`](../docs/en/isa/TMADD.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TMATMUL`](../docs/en/isa/TMATMUL.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMATMUL_ACC`](../docs/en/isa/TMATMUL_ACC.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TMATMUL_BIAS`](../docs/en/isa/TMATMUL_BIAS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TMATMUL_MX`](../docs/en/isa/TMATMUL_MX.md) | Yes | TODO | TODO | TODO | Yes | Yes |
| [`TMAX`](../docs/en/isa/TMAX.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMAXS`](../docs/en/isa/TMAXS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMIN`](../docs/en/isa/TMIN.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMINS`](../docs/en/isa/TMINS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMOV`](../docs/en/isa/TMOV.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMOV_FP`](../docs/en/isa/TMOV_FP.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMRGSORT`](../docs/en/isa/TMRGSORT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMUL`](../docs/en/isa/TMUL.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMULS`](../docs/en/isa/TMULS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TNEG`](../docs/en/isa/TNEG.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TNOT`](../docs/en/isa/TNOT.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TNOTIFY`](../docs/en/isa/comm/TNOTIFY.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TOR`](../docs/en/isa/TOR.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TORS`](../docs/en/isa/TORS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TPARTADD`](../docs/en/isa/TPARTADD.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TPARTARGMAX`](../docs/en/isa/TPARTARGMAX.md) | TODO | TODO | TODO | TODO | Yes | TODO |
| [`TPARTARGMIN`](../docs/en/isa/TPARTARGMIN.md) | TODO | TODO | TODO | TODO | Yes | TODO |
| [`TPARTMAX`](../docs/en/isa/TPARTMAX.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TPARTMIN`](../docs/en/isa/TPARTMIN.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TPARTMUL`](../docs/en/isa/TPARTMUL.md) | No | TODO | Yes | Yes | Yes | Yes |
| [`TPREFETCH`](../docs/en/isa/TPREFETCH.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TPREFETCH_ASYNC`](../docs/en/isa/TPREFETCH_ASYNC.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TPRELU`](../docs/en/isa/TPRELU.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TPRINT`](../docs/en/isa/TPRINT.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TPOW`](../docs/en/isa/TPOW.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TPUT`](../docs/en/isa/comm/TPUT.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TPUT_ASYNC`](../docs/en/isa/comm/TPUT_ASYNC.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TPOWS`](../docs/en/isa/TPOWS.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TQUANT`](../docs/en/isa/TQUANT.md) | TODO | TODO | TODO | TODO | Yes | TODO |
| [`TRANDOM`](../docs/en/isa/TRANDOM.md) | No | TODO | TODO | TODO | Yes | TODO |
| [`TRECIP`](../docs/en/isa/TRECIP.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TREDUCE`](../docs/en/isa/comm/TREDUCE.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TRELU`](../docs/en/isa/TRELU.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TREM`](../docs/en/isa/TREM.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TREMS`](../docs/en/isa/TREMS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TRESHAPE`](../docs/en/isa/TRESHAPE.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TROWARGMAX`](../docs/en/isa/TROWARGMAX.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TROWARGMIN`](../docs/en/isa/TROWARGMIN.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TROWEXPAND`](../docs/en/isa/TROWEXPAND.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TROWEXPANDADD`](../docs/en/isa/TROWEXPANDADD.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TROWEXPANDDIV`](../docs/en/isa/TROWEXPANDDIV.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TROWEXPANDEXPDIF`](../docs/en/isa/TROWEXPANDEXPDIF.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TROWEXPANDMAX`](../docs/en/isa/TROWEXPANDMAX.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TROWEXPANDMIN`](../docs/en/isa/TROWEXPANDMIN.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TROWEXPANDMUL`](../docs/en/isa/TROWEXPANDMUL.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TROWEXPANDSUB`](../docs/en/isa/TROWEXPANDSUB.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TROWMAX`](../docs/en/isa/TROWMAX.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TROWMIN`](../docs/en/isa/TROWMIN.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TROWPROD`](../docs/en/isa/TROWPROD.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TROWSUM`](../docs/en/isa/TROWSUM.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TRSQRT`](../docs/en/isa/TRSQRT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSCATTER`](../docs/en/isa/TSCATTER.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSEL`](../docs/en/isa/TSEL.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSELS`](../docs/en/isa/TSELS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`SETFMATRIX`](../docs/en/isa/SETFMATRIX.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`SET_IMG2COL_PADDING`](../docs/en/isa/SET_IMG2COL_PADDING.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`SET_IMG2COL_RPT`](../docs/en/isa/SET_IMG2COL_RPT.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TSHL`](../docs/en/isa/TSHL.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TSHLS`](../docs/en/isa/TSHLS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TSHR`](../docs/en/isa/TSHR.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TSHRS`](../docs/en/isa/TSHRS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TSORT32`](../docs/en/isa/TSORT32.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSQRT`](../docs/en/isa/TSQRT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSTORE`](../docs/en/isa/TSTORE.md) | Yes | Yes | Yes | Yes | Yes | TODO |
| [`TSTORE_FP`](../docs/en/isa/TSTORE_FP.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSUB`](../docs/en/isa/TSUB.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSUBS`](../docs/en/isa/TSUBS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TTEST`](../docs/en/isa/comm/TTEST.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TTRANS`](../docs/en/isa/TTRANS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TTRI`](../docs/en/isa/TTRI.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TWAIT`](../docs/en/isa/comm/TWAIT.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TXOR`](../docs/en/isa/TXOR.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TXORS`](../docs/en/isa/TXORS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TMULA`](../docs/zh/isa/TMULA.md) | TODO | TODO | Yes | Yes | Yes | TODO |

Notes:

- `Yes`: the backend has an available implementation.
- `TODO`: the instruction is already part of the public API or documented ISA surface, but the backend implementation is not available yet or has not been integrated for that backend.
- `No`: explicitly unsupported, or not planned for that backend at the moment.
- Blank: the status has not been finalized yet, or the entry is still being reviewed.
