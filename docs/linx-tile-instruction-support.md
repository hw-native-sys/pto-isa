# Linx instruction support and restricted experimental integration

[中文](linx-tile-instruction-support_zh.md)

The native Linx NPU backend is a **restricted experimental integration**. Imported interfaces and implementations are retained, while known unsupported paths produce explicit compilation errors. CPU-SIM retains the same calling interface. Native execution here means local `gfrun` functional-model execution, not physical NPU testing.

## Instruction-based acceptance

The primary denominator is **151 ordinary instruction API names**. Eleven `_U` and 64 `_ASS` forms are audited separately. Coverage totals 226 active instruction API names, not 226 distinct architectural opcodes. Four view/construction helpers, 32 retired declarations and lowercase scalar/Predicate utilities are excluded from that denominator.

A name passes only when every profile assigned to it in this matrix passes. Overloads may be partially available; rejection is never numerical success. Coverage means these profiles were constructed and actually attempted, not exhaustive coverage of formats, layouts, shapes, control words, aliases or every overload.

| Group | CPU: all listed profiles pass | NPU model: all listed profiles pass |
| --- | ---: | ---: |
| Ordinary instruction API names | 151/151 | 109/151 |
| Explicit dimensions (`_U`) | 11/11 | 11/11 |
| Associated (`_ASS`) | 64/64 | 1/64 |
| All active instruction API names | **226/226** | **121/226** |

The underlying 229 profiles all pass CPU execution. Both the unguarded import and pinned upstream implementation pass 122 and fail 107 profiles. The restricted route executes 122 successfully and explicitly rejects 107, with no other unresolved profile failure. Thus 105 names are wholly or partially restricted; they are not numerical passes.

## Pinned environment and criteria

| Component | Revision |
| --- | --- |
| Linx-TileOP-API (`linx`) | `74d7326c521aeb1b21c52602b67c55b42b3563a3` |
| SuperScalarModel / gfrun | `637bde0ec63955c4796828b297e16a057efad3cc` |
| LinxV5 Clang 15.0.4 | `b19e8c0e184cd7e3a1fa5ab57b368cbf978c3aa8` |
| PTO-spec reference | `42067d0f2890b8c013a868bd1247ebc6437b3179` |
| Host | aarch64 Linux / GCC 15.2.1 |

The [catalog](../tests/cpu/linx/instruction_catalog.json) is checked against both preprocessed public entry points. Missing or newly unclassified names fail the audit. [Profile construction](../tests/cpu/linx_instruction_cases.py) assigns actual kernels and independent goldens to every active name.

A pass requires successful compilation/linking, process exit, guest completion, absence of model error diagnostics, exact golden bytes and all 16 trailing canaries. Successful results agree byte for byte across the three routes. Exit code zero or matching upstream failure does not establish success.

The 107 unguarded failures comprise nine compile/assembly failures, 26 normal-completion value/memory mismatches, 11 other execution errors and 61 `_ASS` profiles rejected in their preceding TLOAD INIT. Those 61 targets were not reached; do not claim 61 independent arithmetic defects. Their end-to-end profiles remain unaccepted. The 11 execution errors also include two reloads that print ERROR and return zero with wrong output. In total, 28 profiles exit normally with wrong output, 26 without an error diagnostic.

## Admission boundaries

The [restriction definitions](../include/pto/npu/linx/experimental.hpp) and [mechanical importer](../tests/script/sync_linx_headers.py) insert diagnostics before emitters while preserving signatures, original validation and assembly text. Messages start with `PTO Linx experimental:` and identify the reason and support document. Template paths reject during instantiation, before known compiler backend crashes. A few auxiliary forms with non-template aliases use error attributes and terminate before their retained assembly. `-fsyntax-only` compilation is insufficient for those forms; generate code.

There is no bypass macro. Upstream execution diagnostics use a separate unguarded import or pinned upstream checkout. CPU generation removes only marked integration guards and retains upstream legality checks. The separate admission test runs 428 checks with `NDEBUG`, `-O0/-O2` and public/direct implementation includes. All 107 restricted profiles emit an explicit diagnostic without a compiler internal crash.

| Restricted path | Measured cause and boundary | Upstream record/evidence |
| --- | --- | --- |
| Bounded `MGATHER_EXCH/MAX/MIN/ADD/INC/DEC/AND/OR/XOR` and `MSCATTER_MAX/MIN/ADD/INC/DEC/AND/OR/XOR/POPC` | Eighteen profiles return wrong values, potentially with normal guest completion. Range encoding conflicts with unused B.IOR schema fields. Validated unbounded forms remain available. | [TileOP #317](https://github.com/LinxISA/Linx-TileOP-API/issues/317), [Model #1252](https://github.com/LinxISA/SuperScalarModel/issues/1252) |
| Eight `TSTORE_ATOMIC_*` modes | Normal completion with overwrite instead of reduction: ADD produces 5 where 3+5=8 is required. All are rejected. This is not floating-point tolerance. | [Model #1119](https://github.com/LinxISA/SuperScalarModel/issues/1119#issuecomment-6813968508) |
| TileArray/SubTileView operations, generic `range::Assemble` TLOAD, 61 TEPL `_ASS` forms | The `regions` case rejects its TADD INIT writer. `_ASS` profiles reject their preceding TLOAD INIT before reaching the target arithmetic. Restrict the path; do not infer 61 independent arithmetic defects. | [Model #1253](https://github.com/LinxISA/SuperScalarModel/issues/1253); TLOAD INIT setup diagnostics |
| `TGPR2T` | Model validation rejects attributes, shape and the 3+1 B.IOR schema after successful compilation. | [TileOP #318](https://github.com/LinxISA/Linx-TileOP-API/issues/318) |
| Local `TIMG2COL` | Compilation succeeds; a callee spill TSTORE on an empty-shard PE triggers a model assertion. | [LLVM #164](https://github.com/LinxISA/llvm-project/issues/164) |
| `TMATMUL_MX_ACC` | Basic ACC has an incompatible source binding order. Isolated plain MX and BIAS pass and remain available. | [TileOP #319](https://github.com/LinxISA/Linx-TileOP-API/issues/319) |
| Predicate `atomic_add`, `subview`, `concatenate` | Mask/range contracts or full-capacity TMOV checks reject execution. Unreached later calls are not independently judged. | [Model #1131](https://github.com/LinxISA/SuperScalarModel/issues/1131#issuecomment-6813972601), [Model #1254](https://github.com/LinxISA/SuperScalarModel/issues/1254) |
| `linx_cvt`/packed conversions and BF16 arithmetic | Scalar legalization crashes Clang; packed CVT/ADD receive assembler errors. Conservatively disable this auxiliary family; exclude it from the Tile instruction denominator. | [LLVM #165](https://github.com/LinxISA/llvm-project/issues/165), [LLVM #166](https://github.com/LinxISA/llvm-project/issues/166) |
| `TLOAD2_*`, `TLOAD4_*`, `TSTORE_PART` | Six multi-binder loads fail assembly; `TSTORE.SPART` is unsupported by the assembler. | Independent assembly tests |
| `TSTORE2_DN2DN` | The model requires one source Tile for ordinary Local TSTORE; the two-source form asserts. | Independent execution test |
| `TMOV_L2S_INSERT/PUBLISH`, Shared ND/CUBE conversion | Shared writer range or TMOV descriptors are rejected. Later calls after the first failure remain unreached; the path is restricted. | Independent collective tests |
| Ordinary allocated-Tile `TLOAD_CUBE_ASS`/`TLOAD_ASS` reload | The model logs `ERROR: TLOAD with empty dstTile!` yet completes normally, preserving 128 old values. Dedicated `LocalCubeAssembly` `TLOAD_ASS` remains available. | Distinct initial/replacement input test |

Ordinary `MGATHER_MASK/MSCATTER_MASK`, including bounded overloads with U8 byte masks, remain available. They are distinct from bounded atomics. Packed RowMajor `TCMP/TCMPS` comparison bits are not indexed U8 0/1 masks.

Enabling restricted paths requires pinning a validated API/compiler/model combination, rerunning the same inputs, independent goldens and failure-side-effect checks, then removing the relevant guard and updating both language versions. Closing an upstream issue or compiling successfully is insufficient.

## Numerical contract and profile scope

- Same-type `TEXPDIF` rounds SUB in its operation type before EXP; widening forms subtract in FP32. Each FP16/BF16 test covers 128 inputs that distinguish intermediate rounding behavior.
- Experimental CPU-SIM `TRSQRT` explicitly follows the pinned model: typed SQRT followed by DIV, including FP16/BF16. The pinned PTO-spec high-precision reciprocal-square-root reference differs. The two contracts have a numerical discrepancy. Each narrow format has 128 distinguishing inputs.
- CPU `TPACK_ASS` computes its width from the current writer's source raw-word count and confines reads to that writer. An independent two-fragment M32 test passes on all three routes.
- FP32 vector, reduction, selection, indexed and bitwise profiles use nonconstant coordinate data. Atomics check both returned old values and GM updates; Shared/slot profiles use four PEs. MX ACC and BIAS execute independently. TPREFETCH checks completion and subsequent data, not cache behavior.
- These profiles do not prove all-input transcendental equivalence, physical resource/timing/cache behavior, every atomic race order or undefined data behavior. See the [backend contract](../include/pto/npu/linx/README.md).

## Per-instruction matrix

“Partial / blocked” in the NPU model column means at least one admitted profile and one rejected profile. Profile IDs work with `--case`; source and generated evidence retain exact types, layouts, inputs, calls and goldens.

### Ordinary instruction names

| Instruction | CPU | NPU model | Profile IDs |
| --- | --- | --- | --- |
| `GMOV` | pass | pass | `core_gmov` |
| `MGATHER` | pass | pass | `indexed_mask` |
| `MGATHER_ADD` | pass | partial / blocked | `gather_atomic_3`, `gather_atomic_3_unbounded` |
| `MGATHER_AND` | pass | partial / blocked | `gather_atomic_6`, `gather_atomic_6_unbounded` |
| `MGATHER_CAS` | pass | pass | `compare_exchange` |
| `MGATHER_DEC` | pass | partial / blocked | `gather_atomic_5`, `gather_atomic_5_unbounded` |
| `MGATHER_EXCH` | pass | partial / blocked | `gather_atomic_0`, `gather_atomic_0_unbounded` |
| `MGATHER_INC` | pass | partial / blocked | `gather_atomic_4`, `gather_atomic_4_unbounded` |
| `MGATHER_MASK` | pass | pass | `indexed_mask` |
| `MGATHER_MAX` | pass | partial / blocked | `gather_atomic_1`, `gather_atomic_1_unbounded` |
| `MGATHER_MIN` | pass | partial / blocked | `gather_atomic_2`, `gather_atomic_2_unbounded` |
| `MGATHER_OR` | pass | partial / blocked | `gather_atomic_7`, `gather_atomic_7_unbounded` |
| `MGATHER_XOR` | pass | partial / blocked | `gather_atomic_8`, `gather_atomic_8_unbounded` |
| `MSCATTER` | pass | pass | `MSCATTER` |
| `MSCATTER_ADD` | pass | partial / blocked | `scatter_atomic_2`, `scatter_atomic_2_unbounded` |
| `MSCATTER_AND` | pass | partial / blocked | `scatter_atomic_5`, `scatter_atomic_5_unbounded` |
| `MSCATTER_DEC` | pass | partial / blocked | `scatter_atomic_4`, `scatter_atomic_4_unbounded` |
| `MSCATTER_INC` | pass | partial / blocked | `scatter_atomic_3`, `scatter_atomic_3_unbounded` |
| `MSCATTER_MASK` | pass | pass | `indexed_mask` |
| `MSCATTER_MAX` | pass | partial / blocked | `scatter_atomic_0`, `scatter_atomic_0_unbounded` |
| `MSCATTER_MIN` | pass | partial / blocked | `scatter_atomic_1`, `scatter_atomic_1_unbounded` |
| `MSCATTER_OR` | pass | partial / blocked | `scatter_atomic_6`, `scatter_atomic_6_unbounded` |
| `MSCATTER_POPC` | pass | partial / blocked | `scatter_atomic_8`, `scatter_atomic_8_unbounded` |
| `MSCATTER_XOR` | pass | partial / blocked | `scatter_atomic_7`, `scatter_atomic_7_unbounded` |
| `TABS` | pass | pass | `TABS` |
| `TADD` | pass | partial / blocked | `TADD`, `regions` |
| `TADDS` | pass | pass | `TADDS`, `dynamic_shape` |
| `TAND` | pass | pass | `TAND` |
| `TANDS` | pass | pass | `select_bits` |
| `TCI` | pass | pass | `TCI` |
| `TCI_2D` | pass | pass | `TCI_2D` |
| `TCI_COL` | pass | pass | `TCI_COL` |
| `TCI_ROW` | pass | pass | `TCI_ROW` |
| `TCMP` | pass | pass | `compare_move` |
| `TCMPS` | pass | pass | `select_bits` |
| `TCOLARGMAX` | pass | pass | `TCOLARGMAX` |
| `TCOLARGMIN` | pass | pass | `TCOLARGMIN` |
| `TCOLEXPAND` | pass | pass | `broadcast_col_0` |
| `TCOLEXPANDADD` | pass | pass | `broadcast_col_1` |
| `TCOLEXPANDDIV` | pass | pass | `broadcast_col_4` |
| `TCOLEXPANDEXPDIF` | pass | pass | `broadcast_col_7` |
| `TCOLEXPANDMAX` | pass | pass | `broadcast_col_5` |
| `TCOLEXPANDMIN` | pass | pass | `broadcast_col_6` |
| `TCOLEXPANDMUL` | pass | pass | `broadcast_col_3` |
| `TCOLEXPANDSUB` | pass | pass | `broadcast_col_2` |
| `TCOLMAX` | pass | pass | `TCOLMAX` |
| `TCOLMIN` | pass | pass | `TCOLMIN` |
| `TCOLPROD` | pass | pass | `TCOLPROD` |
| `TCOLSUM` | pass | pass | `TCOLSUM` |
| `TCVT` | pass | pass | `fp16_convert` |
| `TCVT_T` | pass | pass | `TCVT_T` |
| `TDIV` | pass | pass | `TDIV` |
| `TDIVS` | pass | pass | `TDIVS` |
| `TEXP` | pass | pass | `TEXP` |
| `TEXPANDS` | pass | pass | `rearrange` |
| `TEXPDIF` | pass | pass | `TEXPDIF`, `TEXPDIF_fp16`, `TEXPDIF_bf16` |
| `TFMA` | pass | pass | `TFMA` |
| `TGATHER` | pass | pass | `tile_indexed` |
| `TGEMV` | pass | pass | `gemv` |
| `TGEMV_ACC` | pass | pass | `gemv` |
| `TGEMV_BIAS` | pass | pass | `gemv` |
| `TGEMV_MX` | pass | pass | `gemv` |
| `TGEMV_MX_ACC` | pass | pass | `gemv` |
| `TGEMV_MX_BIAS` | pass | pass | `gemv` |
| `TGPR2T` | pass | blocked | `gpr_bits` |
| `TIMG2COL` | pass | blocked | `image_nhwc` |
| `TIMG2COL_SPART` | pass | pass | `image_nchw` |
| `TIMG2COL_SPART_SLOT` | pass | pass | `matrix_slots` |
| `TLOAD` | pass | pass | `shared_load`, `shared_single_producer`, `weights`, `dynamic_shape` |
| `TLOAD2_DN2ZN` | pass | blocked | `TLOAD2_DN2ZN` |
| `TLOAD2_ND2NZ` | pass | blocked | `TLOAD2_ND2NZ` |
| `TLOAD2_ND2ZN` | pass | blocked | `TLOAD2_ND2ZN` |
| `TLOAD4_DN2ZN` | pass | blocked | `TLOAD4_DN2ZN` |
| `TLOAD4_ND2NZ` | pass | blocked | `TLOAD4_ND2NZ` |
| `TLOAD4_ND2ZN` | pass | blocked | `TLOAD4_ND2ZN` |
| `TLOAD_CUBE` | pass | pass | `cube_transport` |
| `TLOAD_SLOT` | pass | pass | `matrix_slots` |
| `TLOG` | pass | pass | `TLOG` |
| `TMATMUL` | pass | pass | `matrix_shared`, `matrix_aux` |
| `TMATMUL_ACC` | pass | pass | `matrix_acc_scale` |
| `TMATMUL_BIAS` | pass | pass | `TMATMUL_BIAS` |
| `TMATMUL_MX` | pass | pass | `matrix_hif4`, `TMATMUL_MX_isolated` |
| `TMATMUL_MX_ACC` | pass | blocked | `TMATMUL_MX_ACC_isolated` |
| `TMATMUL_MX_BIAS` | pass | pass | `TMATMUL_MX_BIAS_isolated` |
| `TMATMUL_SLOT` | pass | pass | `matrix_slots` |
| `TMAX` | pass | pass | `TMAX` |
| `TMAXS` | pass | pass | `TMAXS` |
| `TMIN` | pass | pass | `TMIN` |
| `TMINS` | pass | pass | `TMINS` |
| `TMOV` | pass | pass | `compare_move` |
| `TMOV_CUBE_TO_SHARED_ND` | pass | blocked | `shared_nd` |
| `TMOV_L2S_INSERT` | pass | blocked | `local_publish` |
| `TMOV_L2S_PUBLISH` | pass | blocked | `local_publish` |
| `TMOV_S2L_BROADCAST` | pass | pass | `shared_load` |
| `TMOV_S2L_EXTRACT` | pass | pass | `image_nchw` |
| `TMOV_SHARED_ND_TO_CUBE` | pass | blocked | `shared_nd` |
| `TMUL` | pass | pass | `TMUL` |
| `TMULS` | pass | pass | `TMULS` |
| `TNEG` | pass | pass | `TNEG` |
| `TNOT` | pass | pass | `select_bits` |
| `TOR` | pass | pass | `TOR` |
| `TORS` | pass | pass | `select_bits` |
| `TPACK` | pass | pass | `rearrange` |
| `TPERMUTE` | pass | pass | `rearrange` |
| `TPREFETCH` | pass | pass | `compare_move` |
| `TRECIP` | pass | pass | `TRECIP` |
| `TRELU` | pass | pass | `TRELU` |
| `TREM` | pass | pass | `TREM` |
| `TREMS` | pass | pass | `TREMS` |
| `TROWARGMAX` | pass | pass | `TROWARGMAX` |
| `TROWARGMIN` | pass | pass | `TROWARGMIN` |
| `TROWEXPAND` | pass | pass | `broadcast_row_0` |
| `TROWEXPANDADD` | pass | pass | `broadcast_row_1` |
| `TROWEXPANDDIV` | pass | pass | `broadcast_row_4` |
| `TROWEXPANDEXPDIF` | pass | pass | `broadcast_row_7` |
| `TROWEXPANDMAX` | pass | pass | `broadcast_row_5` |
| `TROWEXPANDMIN` | pass | pass | `broadcast_row_6` |
| `TROWEXPANDMUL` | pass | pass | `broadcast_row_3` |
| `TROWEXPANDSUB` | pass | pass | `broadcast_row_2` |
| `TROWMAX` | pass | pass | `TROWMAX` |
| `TROWMIN` | pass | pass | `TROWMIN` |
| `TROWPROD` | pass | pass | `TROWPROD` |
| `TROWSUM` | pass | pass | `TROWSUM` |
| `TRSQRT` | pass | pass | `TRSQRT`, `TRSQRT_fp16`, `TRSQRT_bf16` |
| `TSCATTER` | pass | pass | `tile_indexed` |
| `TSEL` | pass | pass | `compare_move` |
| `TSELS` | pass | pass | `select_bits` |
| `TSHL` | pass | pass | `TSHL` |
| `TSHLS` | pass | pass | `select_bits` |
| `TSHR` | pass | pass | `TSHR` |
| `TSHRS` | pass | pass | `select_bits` |
| `TSHUF` | pass | pass | `rearrange` |
| `TSQRT` | pass | pass | `TSQRT` |
| `TSTORE` | pass | pass | `dynamic_shape` |
| `TSTORE2_DN2DN` | pass | blocked | `TSTORE2_DN2DN` |
| `TSTORE_ATOMIC_ADD` | pass | blocked | `store_atomic_2` |
| `TSTORE_ATOMIC_AND` | pass | blocked | `store_atomic_5` |
| `TSTORE_ATOMIC_DEC` | pass | blocked | `store_atomic_4` |
| `TSTORE_ATOMIC_INC` | pass | blocked | `store_atomic_3` |
| `TSTORE_ATOMIC_MAX` | pass | blocked | `store_atomic_0` |
| `TSTORE_ATOMIC_MIN` | pass | blocked | `store_atomic_1` |
| `TSTORE_ATOMIC_OR` | pass | blocked | `store_atomic_6` |
| `TSTORE_ATOMIC_XOR` | pass | blocked | `store_atomic_7` |
| `TSTORE_CUBE` | pass | pass | `cube_transport` |
| `TSTORE_PART` | pass | blocked | `TSTORE_PART` |
| `TSUB` | pass | pass | `TSUB` |
| `TSUBS` | pass | pass | `TSUBS` |
| `TTRI` | pass | pass | `TTRI` |
| `TUNPACK` | pass | pass | `rearrange` |
| `TXOR` | pass | pass | `TXOR` |
| `TXORS` | pass | pass | `select_bits` |

### Explicit-dimension forms

| Instruction | CPU | NPU model | Profile IDs |
| --- | --- | --- | --- |
| `TADD_U` | pass | pass | `TADD_U` |
| `TAND_U` | pass | pass | `TAND_U` |
| `TDIV_U` | pass | pass | `TDIV_U` |
| `TMAX_U` | pass | pass | `TMAX_U` |
| `TMIN_U` | pass | pass | `TMIN_U` |
| `TMUL_U` | pass | pass | `TMUL_U` |
| `TOR_U` | pass | pass | `TOR_U` |
| `TSHL_U` | pass | pass | `TSHL_U` |
| `TSHR_U` | pass | pass | `TSHR_U` |
| `TSUB_U` | pass | pass | `TSUB_U` |
| `TXOR_U` | pass | pass | `TXOR_U` |

### Associated forms

| Instruction | CPU | NPU model | Profile IDs |
| --- | --- | --- | --- |
| `TABS_ASS` | pass | blocked | `TABS_ASS` |
| `TADDS_ASS` | pass | blocked | `TADDS_ASS` |
| `TADD_ASS` | pass | blocked | `TADD_ASS` |
| `TANDS_ASS` | pass | blocked | `TANDS_ASS` |
| `TAND_ASS` | pass | blocked | `TAND_ASS` |
| `TCMPS_ASS` | pass | blocked | `TCMPS_ASS` |
| `TCMP_ASS` | pass | blocked | `TCMP_ASS` |
| `TCOLEXPANDADD_ASS` | pass | blocked | `TCOLEXPANDADD_ASS` |
| `TCOLEXPANDDIV_ASS` | pass | blocked | `TCOLEXPANDDIV_ASS` |
| `TCOLEXPANDEXPDIF_ASS` | pass | blocked | `TCOLEXPANDEXPDIF_ASS` |
| `TCOLEXPANDMAX_ASS` | pass | blocked | `TCOLEXPANDMAX_ASS` |
| `TCOLEXPANDMIN_ASS` | pass | blocked | `TCOLEXPANDMIN_ASS` |
| `TCOLEXPANDMUL_ASS` | pass | blocked | `TCOLEXPANDMUL_ASS` |
| `TCOLEXPANDSUB_ASS` | pass | blocked | `TCOLEXPANDSUB_ASS` |
| `TCOLEXPAND_ASS` | pass | blocked | `TCOLEXPAND_ASS` |
| `TCOLMAX_ASS` | pass | blocked | `TCOLMAX_ASS` |
| `TCOLMIN_ASS` | pass | blocked | `TCOLMIN_ASS` |
| `TCOLPROD_ASS` | pass | blocked | `TCOLPROD_ASS` |
| `TCOLSUM_ASS` | pass | blocked | `TCOLSUM_ASS` |
| `TCVT_ASS` | pass | blocked | `TCVT_ASS` |
| `TDIVS_ASS` | pass | blocked | `TDIVS_ASS` |
| `TDIV_ASS` | pass | blocked | `TDIV_ASS` |
| `TEXP_ASS` | pass | blocked | `TEXP_ASS` |
| `TFMA_ASS` | pass | blocked | `TFMA_ASS` |
| `TLOAD_ASS` | pass | partial / blocked | `local_assembly`, `TLOAD_ASS_reload` |
| `TLOAD_CUBE_ASS` | pass | blocked | `TLOAD_CUBE_ASS` |
| `TLOG_ASS` | pass | blocked | `TLOG_ASS` |
| `TMAXS_ASS` | pass | blocked | `TMAXS_ASS` |
| `TMAX_ASS` | pass | blocked | `TMAX_ASS` |
| `TMINS_ASS` | pass | blocked | `TMINS_ASS` |
| `TMIN_ASS` | pass | blocked | `TMIN_ASS` |
| `TMULS_ASS` | pass | blocked | `TMULS_ASS` |
| `TMUL_ASS` | pass | blocked | `TMUL_ASS` |
| `TNEG_ASS` | pass | blocked | `TNEG_ASS` |
| `TNOT_ASS` | pass | blocked | `TNOT_ASS` |
| `TORS_ASS` | pass | blocked | `TORS_ASS` |
| `TOR_ASS` | pass | blocked | `TOR_ASS` |
| `TPACK_ASS` | pass | pass | `TPACK_ASS` |
| `TRECIP_ASS` | pass | blocked | `TRECIP_ASS` |
| `TRELU_ASS` | pass | blocked | `TRELU_ASS` |
| `TREMS_ASS` | pass | blocked | `TREMS_ASS` |
| `TREM_ASS` | pass | blocked | `TREM_ASS` |
| `TROWEXPANDADD_ASS` | pass | blocked | `TROWEXPANDADD_ASS` |
| `TROWEXPANDDIV_ASS` | pass | blocked | `TROWEXPANDDIV_ASS` |
| `TROWEXPANDEXPDIF_ASS` | pass | blocked | `TROWEXPANDEXPDIF_ASS` |
| `TROWEXPANDMAX_ASS` | pass | blocked | `TROWEXPANDMAX_ASS` |
| `TROWEXPANDMIN_ASS` | pass | blocked | `TROWEXPANDMIN_ASS` |
| `TROWEXPANDMUL_ASS` | pass | blocked | `TROWEXPANDMUL_ASS` |
| `TROWEXPANDSUB_ASS` | pass | blocked | `TROWEXPANDSUB_ASS` |
| `TROWEXPAND_ASS` | pass | blocked | `TROWEXPAND_ASS` |
| `TROWMAX_ASS` | pass | blocked | `TROWMAX_ASS` |
| `TROWMIN_ASS` | pass | blocked | `TROWMIN_ASS` |
| `TROWPROD_ASS` | pass | blocked | `TROWPROD_ASS` |
| `TROWSUM_ASS` | pass | blocked | `TROWSUM_ASS` |
| `TRSQRT_ASS` | pass | blocked | `TRSQRT_ASS` |
| `TSHLS_ASS` | pass | blocked | `TSHLS_ASS` |
| `TSHL_ASS` | pass | blocked | `TSHL_ASS` |
| `TSHRS_ASS` | pass | blocked | `TSHRS_ASS` |
| `TSHR_ASS` | pass | blocked | `TSHR_ASS` |
| `TSQRT_ASS` | pass | blocked | `TSQRT_ASS` |
| `TSUBS_ASS` | pass | blocked | `TSUBS_ASS` |
| `TSUB_ASS` | pass | blocked | `TSUB_ASS` |
| `TXORS_ASS` | pass | blocked | `TXORS_ASS` |
| `TXOR_ASS` | pass | blocked | `TXOR_ASS` |

## Non-instruction entries

`TPARTVIEW/TASSEMBLY` execute in `regions`: CPU passes and native admission is restricted. `TPARTELEMENT/TREDUCEPREFIXVIEW` have no numerical acceptance in the test suite. These view helpers are outside the 226-name instruction coverage claim. Lowercase utilities have separate `scalar_rounding`, `packed_bf16`, `predicate_atomic` and `predicate_regions` profiles; utility counts are not Tile opcode counts.

Retired declarations retain their rejection contract and are excluded from executable instruction counts:

`ACCCVT`, `ACCCVT_RMAX_SCALE_NZ2DN`, `ACCSCALE_NZ2DN`, `ACCSCALE_T`, `TADDC`, `TADDSC`, `TAXPY`, `TCONCAT`, `TCONCAT_ASS`, `TDEQUANT`, `TEXTRACT`, `TFILLPAD`, `TFMOD`, `TFMODS`, `TGATHERB`, `THISTOGRAM`, `TINSERT`, `TLRELU`, `TMRGSORT`, `TPARTADD`, `TPARTMAX`, `TPARTMIN`, `TPARTMUL`, `TPRELU`, `TQUANT`, `TRANDOM`, `TSORT`, `TSORT32`, `TSUBC`, `TSUBSC`, `TTRANS`, `TTRANS_ASS`

## Reproduction

```bash
# Run the instruction suite through CPU-SIM and the integrated NPU backend.
python3 tests/cpu/test_linx_models.py --compiler /path/to/g++-15 \
  --linx-compiler /path/to/linx-clang++ \
  --model-root /path/to/SuperScalarModel --runtime-dir /path/to/target-libs \
  --upstream /path/to/Linx-TileOP-API --output-dir /tmp/linx-instructions \
  --routes cpu device --jobs 8 --timeout 30
# --case TPACK_ASS (repeatable) selects an independent profile.
# --routes upstream runs the pinned upstream implementation for comparison.
# Admission requires matching profiles from an unguarded diagnostic import.
# --device-include /path/to/unguarded-import/include selects that import.
python3 tests/cpu/test_linx_admission.py \
  --baseline /tmp/unguarded-instructions/result.json \
  --restricted /tmp/linx-instructions/result.json \
  --output-dir /tmp/linx-admission --jobs 8
python3 tests/cpu/test_linx_native.py --compiler /path/to/g++-15 \
  --linx-compiler /path/to/linx-clang++ --upstream /path/to/Linx-TileOP-API \
  --survey --sanitize --release --output-dir /tmp/linx-regression
python3 tests/script/sync_linx_headers.py --upstream /path/to/Linx-TileOP-API
```

The full numerical audit returns nonzero for rejected/failing native profiles. The admission script separately verifies that admitted profiles execute correctly and known failures are explicitly rejected. Do not combine these into one pass rate. Test output directories retain ELF files, dumps, traces and full commands for inspection.
