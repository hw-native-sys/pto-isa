# 向量 ISA 参考

`pto.v*` 是 PTO ISA 的向量微指令集。它直接暴露向量流水线、向量寄存器、谓词和向量可见 UB 搬运。

## 组织方式

向量参考按指令族组织，具体 per-op 页面位于 `vector/ops/` 下。

## 指令族

| 族 | 说明 | 典型操作 |
|----|------|----------|
| [向量加载存储](./vector-load-store_zh.md) | GM↔UB 和 UB↔vreg 数据搬运 | `vlds`、`vldas`、`vgather2`、`vsld`、`vsst`、`vscatter` 等 |
| [谓词与物化](./predicate-and-materialization_zh.md) | 谓词广播与复制 | `vbr`、`vdup` |
| [一元向量运算](./unary-vector-ops_zh.md) | 单操作数向量运算 | `vabs`、`vneg`、`vexp`、`vsqrt`、`vrec`、`vrelu` 等 |
| [二元向量运算](./binary-vector-ops_zh.md) | 双操作数向量运算 | `vadd`、`vsub`、`vmul`、`vmax`、`vmin`、`vand`、`vor` 等 |
| [向量-标量运算](./vec-scalar-ops_zh.md) | 向量与标量组合运算 | `vadds`、`vsubs`、`vmuls`、`vshls`、`vlrelu` 等 |
| [类型转换](./conversion-ops_zh.md) | 类型转换 | `vci`、`vcvt`、`vtrc` |
| [归约指令](./reduction-ops_zh.md) | 跨 lane 归约 | `vcadd`、`vcmax`、`vcmin`、`vcgadd`、`vcgmax` 等 |
| [比较与选择](./compare-select_zh.md) | 谓词生成与条件选择 | `vcmp`、`vcmps`、`vsel`、`vselr`、`vselrv2` |
| [数据重排](./data-rearrangement_zh.md) | Lane 置换与打包 | `vintlv`、`vslide`、`vshift`、`vpack`、`vzunpack` 等 |
| [SFU 与 DSA](./sfu-and-dsa-ops_zh.md) | 特殊函数单元与 DSA 操作 | `vprelu`、`vexpdiff`、`vaxpy`、`vtranspose`、`vsort32` 等 |

## 共享约束

- 向量宽度由元素类型决定。
- 谓词宽度必须匹配向量宽度。
- 对齐、分布和部分高级形式依赖目标 profile。
- 向量层没有 tile 级 valid region 语义。

## 快速参考

### 常见向量类型

| 类型 | 单元素宽度 | 每个 vreg 的总元素数 |
|------|----------|-------------------|
| f32 / i32 | 4 B | 64 |
| f16 / bf16 / i16 | 2 B | 128 |
| i8 / si8 / ui8 | 1 B | 256 |

### Mask 类型

| Mask 类型 | 每个元素槽字节数 | 总 lane 数 |
|-----------|----------------|----------|
| `mask<b32>` | 4 | 64 |
| `mask<b16>` | 2 | 128 |
| `mask<b8>` | 1 | 256 |

## 相关页面

- [向量指令集](../instruction-surfaces/vector-instructions_zh.md) — 高层描述
- [向量指令族](../instruction-families/vector-families_zh.md) — 规范契约
- [指令描述格式](../reference/format-of-instruction-descriptions_zh.md) — per-op 页面格式标准
- [微指令汇总](./micro-instruction-summary.md) — 向量作用域的标量微指令

## 来源与时序披露

当前向量微指令参考页以现有公开 VPTO 语义和时序材料为准，并据此统一生成各 per-op 页的时序披露：

- 公开来源给出了数字时延或吞吐时，页面直接写出该数字。
- 公开来源只给出流级描述时，页面只写出该更窄的公开契约。
- 公开来源没有给出数字时，页面会明确写成"公开来源未给出"，而不是推测一个常数。
