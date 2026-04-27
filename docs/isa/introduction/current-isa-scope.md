# Current PTO ISA Scope

The current PTO ISA manual defines the instruction inventory and architecture surfaces documented below. This page is the current scope statement for the manual.

## Instruction Sets

PTO defines five named instruction sets with explicit per-op reference pages:

- **Tile instructions**: `pto.t*` operations together with `pto.mgather` and `pto.mscatter`
- **Vector micro instructions**: `pto.v*` operations
- **Scalar and control instructions**: `pto.*` operations used for synchronization, DMA control, predicate construction, and machine-visible control
- **Communication instructions**: inter-NPU collective, point-to-point, and notification operations under `docs/isa/comm/`
- **System scheduling instructions**: TPipe/TMPipe producer-consumer and resource-lifetime operations under `docs/isa/system/`

## Inventory Summary

The current manual documents:

- **127** tile instructions
- **100** vector micro instructions
- **48** scalar and control instructions
- **11** communication instructions
- **3** system scheduling instructions

That yields **289 named instructions** in the current reference set.

## Tile Instruction Inventory

### Sync And Config

`tsync`, `tassign`, `talias`, `setfmatrix`, `settf32mode`, `set_img2col_rpt`, `set_img2col_padding`, `subview`, `get_scale_addr`

### Elementwise Tile-Tile

`tabs`, `tadd`, `taddc`, `tand`, `tcmp`, `tcvt`, `tdiv`, `texp`, `tpow`, `tfmod`, `tlog`, `tmax`, `tmin`, `tmul`, `tneg`, `tnot`, `tor`, `tprelu`, `trecip`, `trelu`, `trem`, `trsqrt`, `tsel`, `tshl`, `tshr`, `tsqrt`, `tsub`, `tsubc`, `txor`

### Tile-Scalar And Immediate

`tadds`, `taddsc`, `taxpy`, `tands`, `tcmps`, `tdivs`, `texpands`, `tfmods`, `tlrelu`, `tmaxs`, `tmins`, `tmuls`, `tpows`, `tors`, `trems`, `tsels`, `tshls`, `tshrs`, `tsubs`, `tsubsc`, `txors`

### Reduce And Expand

`tcolexpand`, `tcolexpandadd`, `tcolexpanddiv`, `tcolexpandexpdif`, `tcolexpandmax`, `tcolexpandmin`, `tcolexpandmul`, `tcolexpandsub`, `tcolargmax`, `tcolargmin`, `tcolmax`, `tcolmin`, `tcolprod`, `tcolsum`, `trowargmax`, `trowargmin`, `trowexpand`, `trowexpandadd`, `trowexpanddiv`, `trowexpandexpdif`, `trowexpandmax`, `trowexpandmin`, `trowexpandmul`, `trowexpandsub`, `trowmax`, `trowmin`, `trowprod`, `trowsum`

### Memory And Data Movement

`tload`, `tprefetch`, `tstore`, `mgather`, `mscatter`

### Matrix And Matrix-Vector

`tgemv`, `tgemv_acc`, `tgemv_bias`, `tgemv_mx`, `tmatmul`, `tmatmul_acc`, `tmatmul_bias`, `tmatmul_mx`

### Layout And Rearrangement

`tconcat`, `textract`, `tfillpad`, `tfillpad_expand`, `tfillpad_inplace`, `timg2col`, `tinsert`, `tmov`, `tpack`, `treshape`, `ttrans`

### Irregular And Complex

`tci`, `tdequant`, `tgather`, `tgatherb`, `thistogram`, `tmrgsort`, `tpartadd`, `tpartmax`, `tpartmin`, `tpartmul`, `tprint`, `tquant`, `trandom`, `tscatter`, `tsort32`, `ttri`

## Vector Micro-Instruction Inventory

### Vector Load-Store

`vgather2`, `vgather2_bc`, `vgatherb`, `vldas`, `vlds`, `vldus`, `vldx2`, `vscatter`, `vsld`, `vsldb`, `vsst`, `vsstb`, `vsta`, `vstar`, `vstas`, `vsts`, `vstu`, `vstur`, `vstus`, `vstx2`

### Predicate And Materialization

`vbr`, `vdup`

### Unary Vector Operations

`vabs`, `vbcnt`, `vcls`, `vexp`, `vln`, `vmov`, `vneg`, `vnot`, `vrec`, `vrelu`, `vrsqrt`, `vsqrt`

### Binary Vector Operations

`vadd`, `vaddc`, `vand`, `vdiv`, `vmax`, `vmin`, `vmul`, `vor`, `vshl`, `vshr`, `vsub`, `vsubc`, `vxor`

### Vector-Scalar Operations

`vaddcs`, `vadds`, `vands`, `vlrelu`, `vmaxs`, `vmins`, `vmuls`, `vors`, `vshls`, `vshrs`, `vsubcs`, `vsubs`, `vxors`

### Conversion Operations

`vci`, `vcvt`, `vtrc`

### Reduction Operations

`vcadd`, `vcgadd`, `vcgmax`, `vcgmin`, `vcmax`, `vcmin`, `vcpadd`

### Compare And Select

`vcmp`, `vcmps`, `vsel`, `vselr`, `vselrv2`

### Data Rearrangement

`vdintlv`, `vdintlvv2`, `vintlv`, `vintlvv2`, `vpack`, `vperm`, `vshift`, `vslide`, `vsqz`, `vsunpack`, `vusqz`, `vzunpack`

### SFU And DSA Operations

`vaddrelu`, `vaddreluconv`, `vaxpy`, `vexpdiff`, `vmrgsort`, `vmula`, `vmulconv`, `vmull`, `vprelu`, `vsort32`, `vsubrelu`, `vtranspose`

## Scalar And Control Instruction Inventory

### Pipeline Sync

`get_buf`, `mem_bar`, `pipe_barrier`, `rls_buf`, `set_cross_core`, `set_flag`, `set_intra_block`, `wait_flag`, `wait_flag_dev`, `wait_intra_core`

### DMA Copy

`copy_gm_to_ubuf`, `copy_ubuf_to_gm`, `copy_ubuf_to_ubuf`, `set_loop_size_outtoub`, `set_loop_size_ubtoout`, `set_loop1_stride_outtoub`, `set_loop1_stride_ubtoout`, `set_loop2_stride_outtoub`, `set_loop2_stride_ubtoout`

### Predicate Load-Store

`pld`, `pldi`, `plds`, `pst`, `psti`, `psts`, `pstu`

### Predicate Generation And Algebra

`pand`, `pdintlv_b8`, `pge_b16`, `pge_b32`, `pge_b8`, `pintlv_b16`, `plt_b16`, `plt_b32`, `plt_b8`, `pnot`, `por`, `ppack`, `psel`, `pset_b16`, `pset_b32`, `pset_b8`, `punpack`, `pxor`

## Communication Instruction Inventory

`tbroadcast`, `tget`, `tget_async`, `tgather`, `tnotify`, `tput`, `tput_async`, `treduce`, `tscatter`, `ttest`, `twait`

## System Scheduling Instruction Inventory

`tfree`, `tpop`, `tpush`

## Related Pages

- [Instruction Set Overview](../instruction-families/README.md)
- [Instruction Family Overview](../instruction-families/README.md)
