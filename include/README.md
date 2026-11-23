# include/

PTO Tile Lib 对外的 C/C++ 头文件（以模板化、基本 header-only 为主）。上层框架/算子可以通过这些头文件生成 PTO ISA 的 Tile 指令序列。

## 快速开始

推荐直接 include 统一入口头：

```cpp
#include <pto/pto-inst.hpp>
```

`pto/pto-inst.hpp` 会根据构建配置选择合适的后端（CPU 仿真/stub 或 NPU 实现）。详情见 [include/pto/README.md](pto/README.md)。

## 目录结构

- `include/pto/`：公共 PTO ISA API 与后端实现（common / cpu / npu / comm）

## 相关文档

- [ISA 指南](../docs/zh/README.md)
- [入门指南](../docs/zh/getting-started.md)

## PTO 指令实现状态（CPU / Costmodel / A2 / A3 / A5 / Kirin）

下表用于跟踪每条指令在不同后端的可用性：

- **CPU**：`__CPU_SIM`（CPU 仿真后端）。
- **Costmodel**：`__COSTMODEL`（A2 / A3 性能仿真后端，包含 `stub` 与 `fit` 两条路径；任一路径支持即记为支持）。
- **A2（Ascend 910B）/ A3（Ascend 910C）**：当前共享 `include/pto/npu/a2a3/` 的实现（因此两列状态相同）。
- **A5（Ascend 950）**：使用 `include/pto/npu/a5/` 的实现。
- **Kirin**：使用 `include/pto/npu/kirin9030/` 的实现。

| 指令 | CPU | Costmodel | A2 | A3 | A5 | Kirin |
|---|---:|---:|---:|---:|---:|---:|
| [`MGATHER`](../docs/zh/isa/MGATHER.md) | 是 | TODO | TODO | TODO | 是 | TODO |
| [`MSCATTER`](../docs/zh/isa/MSCATTER.md) | 是 | TODO | TODO | TODO | 是 | TODO |
| [`TABS`](../docs/zh/isa/TABS.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TADD`](../docs/zh/isa/TADD.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TADDS`](../docs/zh/isa/TADDS.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TAND`](../docs/zh/isa/TAND.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TANDS`](../docs/zh/isa/TANDS.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TASSIGN`](../docs/zh/isa/TASSIGN.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TAXPY`](../docs/zh/isa/TAXPY.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TBROADCAST`](../docs/zh/isa/comm/TBROADCAST.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TCI`](../docs/zh/isa/TCI.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TCMP`](../docs/zh/isa/TCMP.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TCMPS`](../docs/zh/isa/TCMPS.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TCOLARGMAX`](../docs/zh/isa/TCOLARGMAX.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TCOLARGMIN`](../docs/zh/isa/TCOLARGMIN.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TCOLEXPAND`](../docs/zh/isa/TCOLEXPAND.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TCOLEXPANDADD`](../docs/zh/isa/TCOLEXPANDADD.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLEXPANDDIV`](../docs/zh/isa/TCOLEXPANDDIV.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLEXPANDEXPDIF`](../docs/zh/isa/TCOLEXPANDEXPDIF.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLEXPANDMAX`](../docs/zh/isa/TCOLEXPANDMAX.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLEXPANDMIN`](../docs/zh/isa/TCOLEXPANDMIN.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLEXPANDMUL`](../docs/zh/isa/TCOLEXPANDMUL.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLEXPANDSUB`](../docs/zh/isa/TCOLEXPANDSUB.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLMAX`](../docs/zh/isa/TCOLMAX.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TCOLMIN`](../docs/zh/isa/TCOLMIN.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TCOLPROD`](../docs/zh/isa/TCOLPROD.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLSUM`](../docs/zh/isa/TCOLSUM.md) | 是 | 是 | 是 | 是 | 是 | TODO |
| [`TConcat`](../docs/zh/isa/TCONCAT.md) | TODO | TODO | TODO | TODO | 是 | TODO |
| [`TCVT`](../docs/zh/isa/TCVT.md) | 是 | 是 | 是 | 是 | 是 | TODO |
| [`TDEQUANT`](../docs/zh/isa/TDEQUANT.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TDIV`](../docs/zh/isa/TDIV.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TDIVS`](../docs/zh/isa/TDIVS.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TEXP`](../docs/zh/isa/TEXP.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TEXPANDS`](../docs/zh/isa/TEXPANDS.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TEXTRACT`](../docs/zh/isa/TEXTRACT.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TEXTRACT_FP`](../docs/zh/isa/TEXTRACT_FP.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TFILLPAD`](../docs/zh/isa/TFILLPAD.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TFILLPAD_EXPAND`](../docs/zh/isa/TFILLPAD_EXPAND.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TFILLPAD_INPLACE`](../docs/zh/isa/TFILLPAD_INPLACE.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TFMOD`](../docs/zh/isa/TFMOD.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TFMODS`](../docs/zh/isa/TFMODS.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TGATHER`](../docs/zh/isa/TGATHER.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TGATHERB`](../docs/zh/isa/TGATHERB.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TGEMV`](../docs/zh/isa/TGEMV.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TGEMV_ACC`](../docs/zh/isa/TGEMV_ACC.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TGEMV_BIAS`](../docs/zh/isa/TGEMV_BIAS.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TGEMV_MX`](../docs/zh/isa/TGEMV_MX.md) | TODO | TODO | TODO | TODO | 是 | TODO |
| [`TGET`](../docs/zh/isa/comm/TGET.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TGET_ASYNC`](../docs/zh/isa/comm/TGET_ASYNC.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`THISTOGRAM`](../docs/zh/isa/THISTOGRAM.md) | 是 | TODO | 否 | 否 | 是 | 是 |
| [`TIMG2COL`](../docs/zh/isa/TIMG2COL.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TINSERT`](../docs/zh/isa/TINSERT.md) | TODO | TODO | TODO | TODO | 是 | TODO |
| [`TINSERT_FP`](../docs/zh/isa/TINSERT_FP.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TLOAD`](../docs/zh/isa/TLOAD.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TLOG`](../docs/zh/isa/TLOG.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TLRELU`](../docs/zh/isa/TLRELU.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TMADD`](../docs/zh/isa/TMADD.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TMATMUL`](../docs/zh/isa/TMATMUL.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMATMUL_ACC`](../docs/zh/isa/TMATMUL_ACC.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TMATMUL_BIAS`](../docs/zh/isa/TMATMUL_BIAS.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TMATMUL_MX`](../docs/zh/isa/TMATMUL_MX.md) | 是 | TODO | TODO | TODO | 是 | 是 |
| [`TMAX`](../docs/zh/isa/TMAX.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMAXS`](../docs/zh/isa/TMAXS.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMIN`](../docs/zh/isa/TMIN.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMINS`](../docs/zh/isa/TMINS.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMOV`](../docs/zh/isa/TMOV.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMOV_FP`](../docs/zh/isa/TMOV_FP.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMRGSORT`](../docs/zh/isa/TMRGSORT.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMUL`](../docs/zh/isa/TMUL.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMULS`](../docs/zh/isa/TMULS.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TNEG`](../docs/zh/isa/TNEG.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TNOT`](../docs/zh/isa/TNOT.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TNOTIFY`](../docs/zh/isa/comm/TNOTIFY.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TOR`](../docs/zh/isa/TOR.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TORS`](../docs/zh/isa/TORS.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TPARTADD`](../docs/zh/isa/TPARTADD.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TPARTARGMAX`](../docs/zh/isa/TPARTARGMAX.md) | TODO | TODO | TODO | TODO | 是 | TODO |
| [`TPARTARGMIN`](../docs/zh/isa/TPARTARGMIN.md) | TODO | TODO | TODO | TODO | 是 | TODO |
| [`TPARTMAX`](../docs/zh/isa/TPARTMAX.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TPARTMIN`](../docs/zh/isa/TPARTMIN.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TPARTMUL`](../docs/zh/isa/TPARTMUL.md) | 否 | TODO | 是 | 是 | 是 | 是 |
| [`TPREFETCH`](../docs/zh/isa/TPREFETCH.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TPRELU`](../docs/zh/isa/TPRELU.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TPRINT`](../docs/zh/isa/TPRINT.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TPOW`](../docs/zh/isa/TPOW.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TPUT`](../docs/zh/isa/comm/TPUT.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TPUT_ASYNC`](../docs/zh/isa/comm/TPUT_ASYNC.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TPOWS`](../docs/zh/isa/TPOWS.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TQUANT`](../docs/zh/isa/TQUANT.md) | TODO | TODO | TODO | TODO | 是 | TODO |
| [`TRANDOM`](../docs/zh/isa/TRANDOM.md) | 否 | TODO | TODO | TODO | 是 | TODO |
| [`TRECIP`](../docs/zh/isa/TRECIP.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TREDUCE`](../docs/zh/isa/comm/TREDUCE.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TRELU`](../docs/zh/isa/TRELU.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TREM`](../docs/zh/isa/TREM.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TREMS`](../docs/zh/isa/TREMS.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TRESHAPE`](../docs/zh/isa/TRESHAPE.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TROWARGMAX`](../docs/zh/isa/TROWARGMAX.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TROWARGMIN`](../docs/zh/isa/TROWARGMIN.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TROWEXPAND`](../docs/zh/isa/TROWEXPAND.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TROWEXPANDADD`](../docs/zh/isa/TROWEXPANDADD.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TROWEXPANDDIV`](../docs/zh/isa/TROWEXPANDDIV.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TROWEXPANDEXPDIF`](../docs/zh/isa/TROWEXPANDEXPDIF.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TROWEXPANDMAX`](../docs/zh/isa/TROWEXPANDMAX.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TROWEXPANDMIN`](../docs/zh/isa/TROWEXPANDMIN.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TROWEXPANDMUL`](../docs/zh/isa/TROWEXPANDMUL.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TROWEXPANDSUB`](../docs/zh/isa/TROWEXPANDSUB.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TROWMAX`](../docs/zh/isa/TROWMAX.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TROWMIN`](../docs/zh/isa/TROWMIN.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TROWPROD`](../docs/zh/isa/TROWPROD.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TROWSUM`](../docs/zh/isa/TROWSUM.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TRSQRT`](../docs/zh/isa/TRSQRT.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSCATTER`](../docs/zh/isa/TSCATTER.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSEL`](../docs/zh/isa/TSEL.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSELS`](../docs/zh/isa/TSELS.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`SETFMATRIX`](../docs/zh/isa/SETFMATRIX.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`SET_IMG2COL_PADDING`](../docs/zh/isa/SET_IMG2COL_PADDING.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`SET_IMG2COL_RPT`](../docs/zh/isa/SET_IMG2COL_RPT.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TSHL`](../docs/zh/isa/TSHL.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TSHLS`](../docs/zh/isa/TSHLS.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TSHR`](../docs/zh/isa/TSHR.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TSHRS`](../docs/zh/isa/TSHRS.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TSORT32`](../docs/zh/isa/TSORT32.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSQRT`](../docs/zh/isa/TSQRT.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSTORE`](../docs/zh/isa/TSTORE.md) | 是 | 是 | 是 | 是 | 是 | TODO |
| [`TSTORE_FP`](../docs/zh/isa/TSTORE_FP.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSUB`](../docs/zh/isa/TSUB.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSUBS`](../docs/zh/isa/TSUBS.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TTEST`](../docs/zh/isa/comm/TTEST.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TTRANS`](../docs/zh/isa/TTRANS.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TTRI`](../docs/zh/isa/TTRI.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TWAIT`](../docs/zh/isa/comm/TWAIT.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TXOR`](../docs/zh/isa/TXOR.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TXORS`](../docs/zh/isa/TXORS.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TMULA`](../docs/zh/isa/TMULA.md) | TODO | TODO | 是 | 是 | 是 | TODO |

说明：

- `是`：该后端已提供可用实现。
- `TODO`：该指令已进入公共 API 或文档范围，但对应后端实现暂未提供，或尚未在该后端接入。
- `否`：明确不支持或当前不计划在该后端提供实现。
- 空白：状态尚未最终确认，或该项仍在整理中。
