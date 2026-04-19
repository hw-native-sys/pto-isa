# 标量与控制参考

`pto.*` 中的标量与控制部分负责同步、DMA、谓词、控制外壳和共享标量支持逻辑。它们为 tile 与 vector 有效载荷提供执行外壳。

## 组织方式

标量与控制参考按指令族组织，具体 per-op 页面位于 `scalar/ops/` 下。

## 指令族

| 族 | 说明 | 典型操作 |
|----|------|----------|
| [控制与配置](./control-and-configuration_zh.md) | NOP、barrier、yield；tsetf32mode、tsethf32mode、tsetfmatrix | `nop`、`barrier`、`yield` 等 |
| [PTO 微指令参考](./ops/micro-instruction/README_zh.md) | 标量微指令：BlockDim、指针操作、向量作用域、对齐状态 | `pto.get_block_idx`、`pto.castptr`、`pto.vecscope` 等 |
| [流水线同步](./pipeline-sync_zh.md) | 基于事件的跨 pipe 同步 | `set_flag`、`wait_flag`、`pipe_barrier`、`mem_bar`、`get_buf`、`rls_buf` 等 |
| [DMA 拷贝](./dma-copy_zh.md) | GM↔UB 和 UB↔UB 数据搬运 | `copy_gm_to_ubuf`、`copy_ubuf_to_gm`、`copy_ubuf_to_ubuf`、loop size/stride setters |
| [谓词加载存储](./predicate-load-store_zh.md) | 谓词感知的标量加载/存储 | `pld`、`plds`、`pldi`、`psts`、`pst`、`psti`、`pstu` |
| [谓词生成与代数](./predicate-generation-and-algebra_zh.md) | 谓词构造与逻辑运算 | `pset_b8/b16/b32`、`pge_b8/b16/b32`、`plt_b8/b16/b32`、`pand`、`por`、`pxor`、`pnot`、`psel` 等 |
| [共享算术](./shared-arith_zh.md) | 跨指令集共享的标量算术运算 | 标量算术操作 |
| [共享 SCF](./shared-scf_zh.md) | 标量结构化控制流 | `scf.for`、`scf.if`、`scf.while` |

## 共享约束

- pipe / event 空间受目标 profile 约束。
- DMA 参数必须自洽。
- 谓词宽度和控制参数必须与目标操作匹配。
- 顺序边必须与后续 tile / vector 有效载荷对齐。

## 关键架构概念

### Pipe 类型

| Pipe | 角色 |
|------|------|
| `PIPE_V` | 向量流水线 |
| `PIPE_MTE1` | 内存传输引擎 1（GM↔UB 入方向） |
| `PIPE_MTE2` | 内存传输引擎 2（UB↔tile buffer 入方向） |
| `PIPE_MTE3` | 内存传输引擎 3（tile buffer↔UB↔GM 出方向） |
| `PIPE_CUBE` | CUBE / 矩阵乘法单元 |

### 事件同步

事件（`event_t`）协调跨 pipe 的异步操作。程序从一个 pipe 设置标志（`set_flag`），从另一个 pipe 等待（`wait_flag`）。

## 相关页面

- [标量与控制指令集](../instruction-surfaces/scalar-and-control-instructions_zh.md) — 高层描述
- [标量与控制指令族](../instruction-families/scalar-and-control-families_zh.md) — 规范契约
- [指令描述格式](../reference/format-of-instruction-descriptions_zh.md) — per-op 页面格式标准
