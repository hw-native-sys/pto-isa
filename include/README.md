# include/

Public C/C++ headers for PTO Tile Lib (primarily header-only, template-based). Upper-layer frameworks or operator code can include these headers to emit PTO ISA Tile-level operations.

## Quick Start

Include the unified entry header:

```cpp
#include <pto/pto-inst.hpp>
```

`pto/pto-inst.hpp` selects the appropriate backend (CPU simulation/stub or NPU implementation) based on build configuration. See [include/pto/README.md](pto/README.md) for details.

## Layout

- `include/pto/`: Public PTO ISA API and backend implementations (common / cpu / npu / comm)

## Related Docs

- [ISA guide](../docs/README.md)
- [Getting started](../docs/getting-started.md)

## PTO Instruction Implementation Status (CPU / Costmodel / A2 / A3 / A5 / Kirin)

This table tracks per-instruction backend availability:

- **CPU**: `__CPU_SIM` (CPU simulation backend). More information about this backend can be found in [docs/coding/cpu_sim.md](../docs/coding/cpu_sim.md)
- **Costmodel**: `__COSTMODEL` (A2 / A3 cost model backend, including `stub` and `fit` paths; if either path supports an instruction, it is marked as supported).
- **A2 (Ascend 910B) / A3 (Ascend 910C)**: share the `include/pto/npu/a2a3/` implementation today (so the status is identical for both columns).
- **A5 (Ascend 950)**: uses the `include/pto/npu/a5/` implementation.
- **Kirin**: uses the `include/pto/npu/kirin9030/` implementation.

| Instruction | CPU | Costmodel | A2 | A3 | A5 | Kirin |
|---|---:|---:|---:|---:|---:|---:|
| [`MGATHER`](../docs/isa/MGATHER.md) | Yes | TODO | TODO | TODO | Yes | TODO |
| [`MSCATTER`](../docs/isa/MSCATTER.md) | Yes | TODO | TODO | TODO | Yes | TODO |
| [`TABS`](../docs/isa/TABS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TADD`](../docs/isa/TADD.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TADDC`](../docs/isa/TADDC.md) | Yes | TODO | TODO | TODO | TODO | TODO |
| [`TADDS`](../docs/isa/TADDS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TADDSC`](../docs/isa/TADDSC.md) | Yes | TODO | TODO | TODO | TODO | TODO |
| [`TAND`](../docs/isa/TAND.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TANDS`](../docs/isa/TANDS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TASSIGN`](../docs/isa/TASSIGN.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TAXPY`]() | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TBROADCAST`](../docs/isa/comm/TBROADCAST.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TCI`](../docs/isa/TCI.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TCMP`](../docs/isa/TCMP.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TCMPS`](../docs/isa/TCMPS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TCOLARGMAX`](../docs/isa/TCOLARGMAX.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TCOLARGMIN`](../docs/isa/TCOLARGMIN.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TCOLEXPAND`](../docs/isa/TCOLEXPAND.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TCOLEXPANDADD`](../docs/isa/TCOLEXPANDADD.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLEXPANDDIV`](../docs/isa/TCOLEXPANDDIV.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLEXPANDEXPDIF`](../docs/isa/TCOLEXPANDEXPDIF.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLEXPANDMAX`](../docs/isa/TCOLEXPANDMAX.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLEXPANDMIN`](../docs/isa/TCOLEXPANDMIN.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLEXPANDMUL`](../docs/isa/TCOLEXPANDMUL.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLEXPANDSUB`](../docs/isa/TCOLEXPANDSUB.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLMAX`](../docs/isa/TCOLMAX.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TCOLMIN`](../docs/isa/TCOLMIN.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TCOLPROD`](../docs/isa/TCOLPROD.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TCOLSUM`](../docs/isa/TCOLSUM.md) | Yes | Yes | Yes | Yes | Yes | TODO |
| [`TCVT`](../docs/isa/TCVT.md) | Yes | Yes | Yes | Yes | Yes | TODO |
| [`TDIV`](../docs/isa/TDIV.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TDIVS`](../docs/isa/TDIVS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TEXP`](../docs/isa/TEXP.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TEXPANDS`](../docs/isa/TEXPANDS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TEXTRACT`](../docs/isa/TEXTRACT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TEXTRACT_FP`](../docs/isa/TEXTRACT_FP.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TFILLPAD`](../docs/isa/TFILLPAD.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TFILLPAD_EXPAND`](../docs/isa/TFILLPAD_EXPAND.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TFILLPAD_INPLACE`](../docs/isa/TFILLPAD_INPLACE.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TFMOD`](../docs/isa/TFMOD.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TFMODS`](../docs/isa/TFMODS.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TGATHER`](../docs/isa/TGATHER.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TGATHERB`](../docs/isa/TGATHERB.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TGEMV`](../docs/isa/TGEMV.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TGEMV_ACC`](../docs/isa/TGEMV_ACC.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TGEMV_BIAS`](../docs/isa/TGEMV_BIAS.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TGEMV_MX`](../docs/isa/TGEMV_MX.md) | TODO | TODO | TODO | TODO | Yes | TODO |
| [`TGET`](../docs/isa/comm/TGET.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TGET_ASYNC`](../docs/isa/comm/TGET_ASYNC.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TGET_SCALE_ADDR`](../docs/isa/TGET_SCALE_ADDR.md) | TODO | TODO | TODO | TODO | Yes | TODO |
| [`TIMG2COL`](../docs/isa/TIMG2COL.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TINSERT`](../docs/isa/TINSERT.md) | TODO | TODO | TODO | TODO | Yes | TODO |
| [`TINSERT_FP`](../docs/isa/TINSERT_FP.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TLOAD`](../docs/isa/TLOAD.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TLOG`](../docs/isa/TLOG.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TLRELU`](../docs/isa/TLRELU.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TMATMUL`](../docs/isa/TMATMUL.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMATMUL_ACC`](../docs/isa/TMATMUL_ACC.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TMATMUL_BIAS`](../docs/isa/TMATMUL_BIAS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TMATMUL_MX`](../docs/isa/TMATMUL_MX.md) | Yes | TODO | TODO | TODO | Yes | Yes |
| [`TMAX`](../docs/isa/TMAX.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMAXS`](../docs/isa/TMAXS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMIN`](../docs/isa/TMIN.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMINS`](../docs/isa/TMINS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMOV`](../docs/isa/TMOV.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMOV_FP`](../docs/isa/TMOV_FP.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TMRGSORT`](../docs/isa/TMRGSORT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMUL`](../docs/isa/TMUL.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMULS`](../docs/isa/TMULS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TNEG`](../docs/isa/TNEG.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TNOT`](../docs/isa/TNOT.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TNOTIFY`](../docs/isa/comm/TNOTIFY.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TOR`](../docs/isa/TOR.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TORS`](../docs/isa/TORS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TPARTADD`](../docs/isa/TPARTADD.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TPARTARGMAX`](../docs/isa/TPARTARGMAX.md) | TODO | TODO | TODO | TODO | Yes | TODO |
| [`TPARTARGMIN`](../docs/isa/TPARTARGMIN.md) | TODO | TODO | TODO | TODO | Yes | TODO |
| [`TPARTMAX`](../docs/isa/TPARTMAX.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TPARTMIN`](../docs/isa/TPARTMIN.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TPARTMUL`](../docs/isa/TPARTMUL.md) | No | TODO | Yes | Yes | Yes | Yes |
| [`TPREFETCH`](../docs/isa/TPREFETCH.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TPRELU`](../docs/isa/TPRELU.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TPRINT`](../docs/isa/TPRINT.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TPOW`](../docs/isa/TPOW.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TPUT`](../docs/isa/comm/TPUT.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TPUT_ASYNC`](../docs/isa/comm/TPUT_ASYNC.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TPOWS`](../docs/isa/TPOWS.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TQUANT`](../docs/isa/TQUANT.md) | TODO | TODO | TODO | TODO | Yes | TODO |
| [`TRANDOM`](../docs/isa/TRANDOM.md) | No | TODO | TODO | TODO | Yes | TODO |
| [`TRECIP`](../docs/isa/TRECIP.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TREDUCE`](../docs/isa/comm/TREDUCE.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TRELU`](../docs/isa/TRELU.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TREM`](../docs/isa/TREM.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TREMS`](../docs/isa/TREMS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TRESHAPE`](../docs/isa/TRESHAPE.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TROWARGMAX`](../docs/isa/TROWARGMAX.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TROWARGMIN`](../docs/isa/TROWARGMIN.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TROWEXPAND`](../docs/isa/TROWEXPAND.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TROWEXPANDADD`](../docs/isa/TROWEXPANDADD.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TROWEXPANDDIV`](../docs/isa/TROWEXPANDDIV.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TROWEXPANDEXPDIF`](../docs/isa/TROWEXPANDEXPDIF.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TROWEXPANDMAX`](../docs/isa/TROWEXPANDMAX.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TROWEXPANDMIN`](../docs/isa/TROWEXPANDMIN.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TROWEXPANDMUL`](../docs/isa/TROWEXPANDMUL.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TROWEXPANDSUB`](../docs/isa/TROWEXPANDSUB.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TROWMAX`](../docs/isa/TROWMAX.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TROWMIN`](../docs/isa/TROWMIN.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TROWPROD`](../docs/isa/TROWPROD.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TROWSUM`](../docs/isa/TROWSUM.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TRSQRT`](../docs/isa/TRSQRT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSCATTER`](../docs/isa/TSCATTER.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSEL`](../docs/isa/TSEL.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSELS`](../docs/isa/TSELS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`SETFMATRIX`](../docs/isa/SETFMATRIX.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`SET_IMG2COL_PADDING`](../docs/isa/SET_IMG2COL_PADDING.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`SET_IMG2COL_RPT`](../docs/isa/SET_IMG2COL_RPT.md) | TODO | TODO | Yes | Yes | Yes | TODO |
| [`TSHL`](../docs/isa/TSHL.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TSHLS`](../docs/isa/TSHLS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TSHR`](../docs/isa/TSHR.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TSHRS`](../docs/isa/TSHRS.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TSORT32`](../docs/isa/TSORT32.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSQRT`](../docs/isa/TSQRT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSTORE`](../docs/isa/TSTORE.md) | Yes | Yes | Yes | Yes | Yes | TODO |
| [`TSTORE_FP`](../docs/isa/TSTORE_FP.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TSUB`](../docs/isa/TSUB.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSUBC`](../docs/isa/TSUBC.md) | Yes | TODO | TODO | TODO | TODO | TODO |
| [`TSUBS`](../docs/isa/TSUBS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSUBSC`](../docs/isa/TSUBSC.md) | Yes | TODO | TODO | TODO | TODO | TODO |
| [`TSUBVIEW`](../docs/isa/TSUBVIEW.md) | TODO | TODO | Yes | Yes | TODO | TODO |
| [`TSYNC`](../docs/isa/TSYNC.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TTEST`](../docs/isa/comm/TTEST.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TTRANS`](../docs/isa/TTRANS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TTRI`](../docs/isa/TTRI.md) | TODO | TODO | Yes | Yes | Yes | Yes |
| [`TWAIT`](../docs/isa/comm/TWAIT.md) | Yes | TODO | Yes | Yes | Yes | TODO |
| [`TXOR`](../docs/isa/TXOR.md) | Yes | TODO | Yes | Yes | Yes | Yes |
| [`TXORS`](../docs/isa/TXORS.md) | Yes | TODO | Yes | Yes | Yes | Yes |

Notes:

- `Yes`: the backend has an available implementation.
- `TODO`: the instruction is already part of the public API or documented ISA surface, but the backend implementation is not available yet or has not been integrated for that backend.
- `No`: explicitly unsupported, or not planned for that backend at the moment.
- Blank: the status has not been finalized yet, or the entry is still being reviewed.
