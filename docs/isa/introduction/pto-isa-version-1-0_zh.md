# PTO ISA 版本 1.0

PTO ISA 版本 1.0 的指令清单与架构指令集范围如下。该发布基线也为后续版本的 release note 和兼容性说明提供对照点。

## 版本 1.0 的范围

PTO ISA 版本 1.0 定义了三套具名指令集，并为每条指令提供明确的参考页：

- **Tile 指令**：全部 `pto.t*` 操作，以及 `pto.mgather` 与 `pto.mscatter`
- **向量微指令**：全部 `pto.v*` 操作
- **标量与控制指令**：用于同步、DMA 控制、谓词构造及其他机器可见控制的 `pto.*` 操作

版本 1.0 也包含 `other/` 树中的支撑性参考内容，用于说明通信/运行时和非 ISA 支撑操作。

## 版本 1.0 的清单摘要

PTO ISA 版本 1.0 当前文档化了：

- **120** 条 tile 指令
- **99** 条向量微指令
- **44** 条标量与控制指令

合计为 **263 条具名指令**。这一数字不包含 `other/` 树中的非 ISA / 支撑性参考页。

## Tile 指令清单

### 同步与配置

`tsync`、`tassign`、`tsettf32mode`、`tset_img2col_rpt`、`tset_img2col_padding`、`tsubview`、`tget_scale_addr`

### 逐元素 Tile-Tile

`tabs`、`tadd`、`taddc`、`tand`、`tcmp`、`tcvt`、`tdiv`、`texp`、`tfmod`、`tlog`、`tmax`、`tmin`、`tmul`、`tneg`、`tnot`、`tor`、`tprelu`、`trecip`、`trelu`、`trem`、`trsqrt`、`tsel`、`tshl`、`tshr`、`tsqrt`、`tsub`、`tsubc`、`txor`

### Tile-标量与立即数

`tadds`、`taddsc`、`tands`、`tcmps`、`tdivs`、`texpands`、`tfmods`、`tlrelu`、`tmaxs`、`tmins`、`tmuls`、`tors`、`trems`、`tsels`、`tshls`、`tshrs`、`tsubs`、`tsubsc`、`txors`

### 归约与扩展

`tcolexpand`、`tcolexpandadd`、`tcolexpanddiv`、`tcolexpandexpdif`、`tcolexpandmax`、`tcolexpandmin`、`tcolexpandmul`、`tcolexpandsub`、`tcolmax`、`tcolmin`、`tcolprod`、`tcolsum`、`trowargmax`、`trowargmin`、`trowexpand`、`trowexpandadd`、`trowexpanddiv`、`trowexpandexpdif`、`trowexpandmax`、`trowexpandmin`、`trowexpandmul`、`trowexpandsub`、`trowmax`、`trowmin`、`trowsum`

### 内存与数据搬运

`tload`、`tprefetch`、`tstore`、`tstore_fp`、`mgather`、`mscatter`

### 矩阵与矩阵-向量

`tgemv`、`tgemv_acc`、`tgemv_bias`、`tgemv_mx`、`tmatmul`、`tmatmul_acc`、`tmatmul_bias`、`tmatmul_mx`

### 布局与重排

`textract`、`textract_fp`、`tfillpad`、`tfillpad_expand`、`tfillpad_inplace`、`timg2col`、`tinsert`、`tinsert_fp`、`tmov`、`tmov_fp`、`treshape`、`ttrans`

### 不规则与复杂操作

`tci`、`tgather`、`tgatherb`、`tmrgsort`、`tpartadd`、`tpartmax`、`tpartmin`、`tpartmul`、`tprint`、`tquant`、`tscatter`、`tsort32`、`ttri`

## 向量微指令清单

### 向量加载存储

`vgather2`、`vgather2_bc`、`vgatherb`、`vldas`、`vlds`、`vldus`、`vldx2`、`vscatter`、`vsld`、`vsldb`、`vsst`、`vsstb`、`vsta`、`vstar`、`vstas`、`vsts`、`vstu`、`vstur`、`vstus`、`vstx2`

### 谓词与物化

`vbr`、`vdup`

### 一元向量操作

`vabs`、`vbcnt`、`vcls`、`vexp`、`vln`、`vmov`、`vneg`、`vnot`、`vrec`、`vrelu`、`vrsqrt`、`vsqrt`

### 二元向量操作

`vadd`、`vaddc`、`vand`、`vdiv`、`vmax`、`vmin`、`vmul`、`vor`、`vshl`、`vshr`、`vsub`、`vsubc`、`vxor`

### 向量-标量操作

`vaddcs`、`vadds`、`vands`、`vlrelu`、`vmaxs`、`vmins`、`vmuls`、`vors`、`vshls`、`vshrs`、`vsubcs`、`vsubs`、`vxors`

### 转换操作

`vci`、`vcvt`、`vtrc`

### 归约操作

`vcadd`、`vcgadd`、`vcgmax`、`vcgmin`、`vcmax`、`vcmin`、`vcpadd`

### 比较与选择

`vcmp`、`vcmps`、`vsel`、`vselr`、`vselrv2`

### 数据重排

`vdintlv`、`vdintlvv2`、`vintlv`、`vintlvv2`、`vpack`、`vperm`、`vshift`、`vslide`、`vsqz`、`vsunpack`、`vusqz`、`vzunpack`

### SFU 与 DSA 操作

`vaddrelu`、`vaddreluconv`、`vaxpy`、`vexpdiff`、`vmrgsort`、`vmula`、`vmulconv`、`vmull`、`vprelu`、`vsort32`、`vsubrelu`、`vtranspose`

## 标量与控制指令清单

### 流水线同步

`get_buf`、`mem_bar`、`pipe_barrier`、`rls_buf`、`set_cross_core`、`set_flag`、`set_intra_block`、`wait_flag`、`wait_flag_dev`、`wait_intra_core`

### DMA 拷贝

`copy_gm_to_ubuf`、`copy_ubuf_to_gm`、`copy_ubuf_to_ubuf`、`set_loop_size_outtoub`、`set_loop_size_ubtoout`、`set_loop1_stride_outtoub`、`set_loop1_stride_ubtoout`、`set_loop2_stride_outtoub`、`set_loop2_stride_ubtoout`

### 谓词加载存储

`pld`、`pldi`、`plds`、`pst`、`psti`、`psts`、`pstu`

### 谓词生成与代数

`pand`、`pdintlv_b8`、`pge_b16`、`pge_b32`、`pge_b8`、`pintlv_b16`、`plt_b16`、`plt_b32`、`plt_b8`、`pnot`、`por`、`ppack`、`psel`、`pset_b16`、`pset_b32`、`pset_b8`、`punpack`、`pxor`

## 支撑性参考分组

版本 1.0 的手册也包含 `docs/isa/other/` 下的以下支撑分组：

- `communication-and-runtime`
- `non-isa-and-supporting-ops`

## 相关页面

- [指令集总览](../instruction-surfaces/README_zh.md)
- [指令族总览](../instruction-families/README_zh.md)
