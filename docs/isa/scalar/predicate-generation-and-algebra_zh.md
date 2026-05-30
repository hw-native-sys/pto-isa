# 谓词生成与代数

谓词生成与代数操作在标量与控制指令集中创建、组合、打包、解包和交错 `!pto.mask<G>`。`!pto.mask<G>` 是 `pto.v*` 向量操作消费的 lane mask 机制。

## `!pto.mask<G>` 类型

`!pto.mask<G>` 的宽度与当前元素类型绑定：

| 元素类型 | 向量宽度 N | 谓词宽度 |
| --- | :---: | :---: |
| f32 | 64 | 64 bits |
| f16 / bf16 | 128 | 128 bits |
| i8 / u8 | 256 | 256 bits |

位值 `1` 表示 lane 激活，`0` 表示 lane 非激活。

## 子类概览

| 子类 | 操作 |
| --- | --- |
| Pattern-based construction | `pset_b8`, `pset_b16`, `pset_b32` |
| Comparison generation | `pge_*`, `plt_*` |
| Predicate pack / unpack | `ppack`, `punpack` |
| Boolean algebra | `pand`, `por`, `pxor`, `pnot`, `psel` |
| Interleave / deinterleave | `pdintlv_b8`、`pdintlv_b16`、`pdintlv_b32`、`pintlv_b8`、`pintlv_b16`、`pintlv_b32` |

## 共享约束

- 所有谓词操作数必须是 `!pto.mask<G>`
- 同一操作中的谓词宽度必须一致
- pattern token 必须被当前 profile 支持
- `pge_*` / `plt_*` 的标量类型必须与后缀匹配

## 相关页面

- [谓词加载存储](./predicate-load-store_zh.md)
- [向量谓词与物化](../vector/predicate-and-materialization_zh.md)
