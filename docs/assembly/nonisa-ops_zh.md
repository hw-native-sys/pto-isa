# 非 ISA 运算

## 1. 范围

本页给出 `~/pto-isa.txt` 中非 ISA 指令条目的 PTO AS 运算规范。

- Level-1：SSA 形态，由编译器管理分配与同步。
- Level-2：DPS 形态，支持显式缓冲复用与同步原语。

## 2. View 运算

### 2.1 `make_tensor_view`

```text
// L1
%dst = pto.make_tensor_view %ptr, shape = [sh1,sh2,sh3,sh4,sh5] strides = [st1,st2,st3,st4,st5] : !pto.tensor_view<sh1xsh2xsh3xsh4xsh5xdtype>
```

### 2.2 `partition_view`

```text
// L1
%dst = pto.partition_view %src, offsets = [of1,of2,of3,of4,of5], sizes = [sh1,sh2,sh3,sh4,sh5] : !pto.tensor_view<sh1xsh2xsh3xsh4xsh5xdtype> -> !pto.partition_tensor_view<sh1xsh2xsh3xsh4xsh5xdtype>
```

## 3. Tile 分配

### 3.1 `alloc_tile`（静态参数）

```text
// L2
%dst = pto.alloc_tile : !pto.tile_buf<loc, dtype, rows, cols, v_row, v_col, blayout, slayou, fractal, pad>
```

### 3.2 `alloc_tile`（动态有效域）

```text
// L2
%dst = pto.alloc_tile valid_row = %vr valid_col = %vc : !pto.tile_buf<loc, dtype, rows, cols, v_row=?, v_col=?, blayout, slayou, fractal, pad>
```

## 4. 核参数查询

### 4.1 `get_block_idx`

```text
// L1 / L2
%idx = pto.get_block_idx
```

### 4.2 `get_subblock_idx`

```text
// L1 / L2
%idx = pto.get_subblock_idx
```

### 4.3 `get_block_num`

```text
// L1 / L2
%num = pto.get_block_num
```

### 4.4 `get_subblock_num`

```text
// L1 / L2
%num = pto.get_subblock_num
```

## 5. 指针与标量访问

### 5.1 `addptr`

```text
// L2
%ptr_new = pto.addptr %ptr, %offset
```

### 5.2 `tgetval`

```text
// L2
pto.tgetval ins(%src, %index : !pto.tile_buf<...>, dtype) outs(%val : dtype)
```

### 5.3 `tsetval`

```text
// L2
pto.tsetval ins(%index, %val : dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

## 6. 同步原语（Level-2）

### 6.1 `record_event`

```text
pto.record_event[src_op, dst_op, eventID]
```

说明：这是低层同步原语。对当前的 PTO-DSL 前端流程，优先保持源码不显式写
事件，并让 `ptoas --enable-insert-sync` 插入对应同步。

当前表格支持 op：`TLOAD`、`TSTORE_ACC`、`TSTORE_VEC`、`TMOV_M2L`、`TMOV_M2S`、`TMOV_M2B`、`TMOV_M2V`、`TMOV_V2M`、`TMATMUL`、`TVEC`。

### 6.2 `wait_event`

```text
pto.wait_event[src_op, dst_op, eventID]
```

说明：这是低层同步原语。对当前的 PTO-DSL 前端流程，优先保持源码不显式写
事件，并让 `ptoas --enable-insert-sync` 插入对应同步。

当前表格支持 op：`TLOAD`、`TSTORE_ACC`、`TSTORE_VEC`、`TMOV_M2L`、`TMOV_M2S`、`TMOV_M2B`、`TMOV_M2V`、`TMOV_V2M`、`TMATMUL`、`TVEC`。

### 6.3 `barrier`

```text
pto.barrier(op)
```

当前表格支持 op：`TVEC`、`TMATMUL`。

## 7. 一致性说明

- 这些非 ISA PTO AS 运算统一收敛到本节文档，不进入 `docs/isa/` 的 manifest 驱动指令索引。
- `docs/isa/TSYNC.md` / `docs/isa/TSYNC_zh.md` 仍是 ISA 层同步语义权威来源。
- 当 `~/pto-isa.txt` 变更时，本页应在同一变更集中同步更新。
