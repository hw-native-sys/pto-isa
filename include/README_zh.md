# include/

PTO Tile Lib 对外的 C/C++ 头文件（以模板化、基本 header-only 为主）。上层框架/算子可以通过这些头文件生成 PTO ISA 的 Tile 指令序列。

## 快速开始

推荐直接 include 统一入口头：

```cpp
#include <pto/pto-inst.hpp>
```

`pto/pto-inst.hpp` 会根据构建配置选择合适的后端（CPU 仿真/stub 或 NPU 实现）。详情见 [include/pto/README_zh.md](pto/README_zh.md)。

## 目录结构

- `include/pto/`：公共 PTO ISA API 与后端实现（common / cpu / npu / comm）

## 相关文档

- [ISA 指南](../docs/README_zh.md)
- [入门指南](../docs/getting-started_zh.md)

## PTO 指令实现状态（CPU / Costmodel / A2 / A3 / A5 / Kirin）

下表用于跟踪每条指令在不同后端的可用性：

- **CPU**：`__CPU_SIM`（CPU 仿真后端）。
- **Costmodel**：`__COSTMODEL`（A2 / A3 性能仿真后端，包含 `stub` 与 `fit` 两条路径；任一路径支持即记为支持）。
- **A2（Ascend 910B）/ A3（Ascend 910C）**：当前共享 `include/pto/npu/a2a3/` 的实现（因此两列状态相同）。
- **A5（Ascend 950）**：使用 `include/pto/npu/a5/` 的实现。
- **Kirin**：使用 `include/pto/npu/kirin9030/` 的实现。

| 指令 | CPU | Costmodel | A2 | A3 | A5 | Kirin |
|---|---:|---:|---:|---:|---:|---:|
| [`MGATHER`](../docs/isa/MGATHER_zh.md) | 是 | TODO | TODO | TODO | 是 | TODO |
| [`MSCATTER`](../docs/isa/MSCATTER_zh.md) | 是 | TODO | TODO | TODO | 是 | TODO |
| [`TABS`](../docs/isa/TABS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TADD`](../docs/isa/TADD_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TADDC`](../docs/isa/TADDC_zh.md) | 是 | TODO | TODO | TODO | TODO | TODO |
| [`TADDS`](../docs/isa/TADDS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TADDSC`](../docs/isa/TADDSC_zh.md) | 是 | TODO | TODO | TODO | TODO | TODO |
| [`TAND`](../docs/isa/TAND_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TANDS`](../docs/isa/TANDS_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TASSIGN`](../docs/isa/TASSIGN_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TAXPY`]() | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TBROADCAST`](../docs/isa/comm/TBROADCAST_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TCI`](../docs/isa/TCI_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TCMP`](../docs/isa/TCMP_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TCMPS`](../docs/isa/TCMPS_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TCOLARGMAX`](../docs/isa/TCOLARGMAX_zh.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TCOLARGMIN`](../docs/isa/TCOLARGMIN_zh.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TCOLEXPAND`](../docs/isa/TCOLEXPAND_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TCOLEXPANDADD`](../docs/isa/TCOLEXPANDADD_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLEXPANDDIV`](../docs/isa/TCOLEXPANDDIV_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLEXPANDEXPDIF`](../docs/isa/TCOLEXPANDEXPDIF_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLEXPANDMAX`](../docs/isa/TCOLEXPANDMAX_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLEXPANDMIN`](../docs/isa/TCOLEXPANDMIN_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLEXPANDMUL`](../docs/isa/TCOLEXPANDMUL_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLEXPANDSUB`](../docs/isa/TCOLEXPANDSUB_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLMAX`](../docs/isa/TCOLMAX_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TCOLMIN`](../docs/isa/TCOLMIN_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TCOLPROD`](../docs/isa/TCOLPROD_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TCOLSUM`](../docs/isa/TCOLSUM_zh.md) | 是 | 是 | 是 | 是 | 是 | TODO |
| [`TCVT`](../docs/isa/TCVT_zh.md) | 是 | 是 | 是 | 是 | 是 | TODO |
| [`TDIV`](../docs/isa/TDIV_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TDIVS`](../docs/isa/TDIVS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TEXP`](../docs/isa/TEXP_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TEXPANDS`](../docs/isa/TEXPANDS_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TEXTRACT`](../docs/isa/TEXTRACT_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TEXTRACT_FP`](../docs/isa/TEXTRACT_FP_zh.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TFILLPAD`](../docs/isa/TFILLPAD_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TFILLPAD_EXPAND`](../docs/isa/TFILLPAD_EXPAND_zh.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TFILLPAD_INPLACE`](../docs/isa/TFILLPAD_INPLACE_zh.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TFMOD`](../docs/isa/TFMOD_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TFMODS`](../docs/isa/TFMODS_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TGATHER`](../docs/isa/TGATHER_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TGATHERB`](../docs/isa/TGATHERB_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TGEMV`](../docs/isa/TGEMV_zh.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TGEMV_ACC`](../docs/isa/TGEMV_ACC_zh.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TGEMV_BIAS`](../docs/isa/TGEMV_BIAS_zh.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TGEMV_MX`](../docs/isa/TGEMV_MX_zh.md) | TODO | TODO | TODO | TODO | 是 | TODO |
| [`TGET`](../docs/isa/comm/TGET_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TGET_ASYNC`](../docs/isa/comm/TGET_ASYNC_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TGET_SCALE_ADDR`](../docs/isa/TGET_SCALE_ADDR_zh.md) | TODO | TODO | TODO | TODO | 是 | TODO |
| [`TIMG2COL`](../docs/isa/TIMG2COL_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TINSERT`](../docs/isa/TINSERT_zh.md) | TODO | TODO | TODO | TODO | 是 | TODO |
| [`TINSERT_FP`](../docs/isa/TINSERT_FP_zh.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TLOAD`](../docs/isa/TLOAD_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TLOG`](../docs/isa/TLOG_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TLRELU`](../docs/isa/TLRELU_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TMATMUL`](../docs/isa/TMATMUL_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMATMUL_ACC`](../docs/isa/TMATMUL_ACC_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TMATMUL_BIAS`](../docs/isa/TMATMUL_BIAS_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TMATMUL_MX`](../docs/isa/TMATMUL_MX_zh.md) | 是 | TODO | TODO | TODO | 是 | 是 |
| [`TMAX`](../docs/isa/TMAX_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMAXS`](../docs/isa/TMAXS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMIN`](../docs/isa/TMIN_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMINS`](../docs/isa/TMINS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMOV`](../docs/isa/TMOV_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMOV_FP`](../docs/isa/TMOV_FP_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TMRGSORT`](../docs/isa/TMRGSORT_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMUL`](../docs/isa/TMUL_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMULS`](../docs/isa/TMULS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TNEG`](../docs/isa/TNEG_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TNOT`](../docs/isa/TNOT_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TNOTIFY`](../docs/isa/comm/TNOTIFY_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TOR`](../docs/isa/TOR_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TORS`](../docs/isa/TORS_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TPARTADD`](../docs/isa/TPARTADD_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TPARTARGMAX`](../docs/isa/TPARTARGMAX_zh.md) | TODO | TODO | TODO | TODO | 是 | TODO |
| [`TPARTARGMIN`](../docs/isa/TPARTARGMIN_zh.md) | TODO | TODO | TODO | TODO | 是 | TODO |
| [`TPARTMAX`](../docs/isa/TPARTMAX_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TPARTMIN`](../docs/isa/TPARTMIN_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TPARTMUL`](../docs/isa/TPARTMUL_zh.md) | 否 | TODO | 是 | 是 | 是 | 是 |
| [`TPREFETCH`](../docs/isa/TPREFETCH_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TPRELU`](../docs/isa/TPRELU_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TPRINT`](../docs/isa/TPRINT_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TPOW`](../docs/isa/TPOW_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TPUT`](../docs/isa/comm/TPUT_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TPUT_ASYNC`](../docs/isa/comm/TPUT_ASYNC_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TPOWS`](../docs/isa/TPOWS_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TQUANT`](../docs/isa/TQUANT_zh.md) | TODO | TODO | TODO | TODO | 是 | TODO |
| [`TRANDOM`](../docs/isa/TRANDOM_zh.md) | 否 | TODO | TODO | TODO | 是 | TODO |
| [`TRECIP`](../docs/isa/TRECIP_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TREDUCE`](../docs/isa/comm/TREDUCE_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TRELU`](../docs/isa/TRELU_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TREM`](../docs/isa/TREM_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TREMS`](../docs/isa/TREMS_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TRESHAPE`](../docs/isa/TRESHAPE_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TROWARGMAX`](../docs/isa/TROWARGMAX_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TROWARGMIN`](../docs/isa/TROWARGMIN_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TROWEXPAND`](../docs/isa/TROWEXPAND_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TROWEXPANDADD`](../docs/isa/TROWEXPANDADD_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TROWEXPANDDIV`](../docs/isa/TROWEXPANDDIV_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TROWEXPANDEXPDIF`](../docs/isa/TROWEXPANDEXPDIF_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TROWEXPANDMAX`](../docs/isa/TROWEXPANDMAX_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TROWEXPANDMIN`](../docs/isa/TROWEXPANDMIN_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TROWEXPANDMUL`](../docs/isa/TROWEXPANDMUL_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TROWEXPANDSUB`](../docs/isa/TROWEXPANDSUB_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TROWMAX`](../docs/isa/TROWMAX_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TROWMIN`](../docs/isa/TROWMIN_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TROWPROD`](../docs/isa/TROWPROD_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TROWSUM`](../docs/isa/TROWSUM_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TRSQRT`](../docs/isa/TRSQRT_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSCATTER`](../docs/isa/TSCATTER_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSEL`](../docs/isa/TSEL_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSELS`](../docs/isa/TSELS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`SETFMATRIX`](../docs/isa/SETFMATRIX_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`SET_IMG2COL_PADDING`](../docs/isa/SET_IMG2COL_PADDING_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`SET_IMG2COL_RPT`](../docs/isa/SET_IMG2COL_RPT_zh.md) | TODO | TODO | 是 | 是 | 是 | TODO |
| [`TSHL`](../docs/isa/TSHL_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TSHLS`](../docs/isa/TSHLS_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TSHR`](../docs/isa/TSHR_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TSHRS`](../docs/isa/TSHRS_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TSORT32`](../docs/isa/TSORT32_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSQRT`](../docs/isa/TSQRT_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSTORE`](../docs/isa/TSTORE_zh.md) | 是 | 是 | 是 | 是 | 是 | TODO |
| [`TSTORE_FP`](../docs/isa/TSTORE_FP_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TSUB`](../docs/isa/TSUB_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSUBC`](../docs/isa/TSUBC_zh.md) | 是 | TODO | TODO | TODO | TODO | TODO |
| [`TSUBS`](../docs/isa/TSUBS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSUBSC`](../docs/isa/TSUBSC_zh.md) | 是 | TODO | TODO | TODO | TODO | TODO |
| [`TSUBVIEW`](../docs/isa/TSUBVIEW_zh.md) | TODO | TODO | 是 | 是 | TODO | TODO |
| [`TSYNC`](../docs/isa/TSYNC_zh.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TTEST`](../docs/isa/comm/TTEST_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TTRANS`](../docs/isa/TTRANS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TTRI`](../docs/isa/TTRI_zh.md) | TODO | TODO | 是 | 是 | 是 | 是 |
| [`TWAIT`](../docs/isa/comm/TWAIT_zh.md) | 是 | TODO | 是 | 是 | 是 | TODO |
| [`TXOR`](../docs/isa/TXOR_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |
| [`TXORS`](../docs/isa/TXORS_zh.md) | 是 | TODO | 是 | 是 | 是 | 是 |

说明：

- `是`：该后端已提供可用实现。
- `TODO`：该指令已进入公共 API 或文档范围，但对应后端实现暂未提供，或尚未在该后端接入。
- `否`：明确不支持或当前不计划在该后端提供实现。
- 空白：状态尚未最终确认，或该项仍在整理中。
