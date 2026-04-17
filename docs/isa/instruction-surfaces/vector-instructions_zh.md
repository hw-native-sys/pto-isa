# 向量指令集

`pto.v*` 指令集直接暴露向量流水线，用于对 lane 级寄存器、谓词和向量 tile buffer（硬件实现为 UB）做细粒度控制。

## 指令集概览

向量指令是位于 tile 指令之下的细粒度计算层。它们不处理 tile 的 valid region，而是直接处理：

- 向量寄存器 `!pto.vreg<NxT>`
- 谓词 `!pto.mask`
- 标量操作数
- 向量 tile buffer 指针 `!pto.ptr<T, ub>`

向量宽度 `N` 由元素类型决定，而不是用户随意指定：

| 元素类型 | 向量宽度 N | 寄存器总大小 |
|----------|:---------:|:-----------:|
| `f32` | 64 | 256 B |
| `f16` / `bf16` | 128 | 256 B |
| `i16` / `u16` | 128 | 256 B |
| `i8` / `u8` | 256 | 256 B |
| `f8e4m3` / `f8e5m2` | 256 | 256 B |

## 数据流

```text
向量 tile buffer（硬件实现为 UB）
    │
    │  vlds / vsld / vgather2
    ▼
Vector Registers (!pto.vreg<NxT>) ──► Vector Compute (pto.v*) ──► Vector Registers
    │                                                            │
    │  vsts / vsst / vscatter                                    │
    └────────────────────────────────────────────────────────────┘
                    │
                    ▼
          向量 tile buffer（硬件实现为 UB） ──► copy_ubuf_to_gm ──► GM
```

## 指令分类

| 类别 | 说明 | 示例 |
|------|------|------|
| 向量加载存储 | 向量 tile buffer 与向量寄存器之间搬运 | `vlds`、`vldas`、`vldus`、`vgather2`、`vsts`、`vscatter` |
| 谓词与物化 | 广播、复制、谓词相关物化 | `vbr`、`vdup` |
| 一元向量操作 | 单输入 lane 级操作 | `vabs`、`vneg`、`vexp`、`vsqrt`、`vrec`、`vrelu` |
| 二元向量操作 | 双输入 lane 级操作 | `vadd`、`vsub`、`vmul`、`vdiv`、`vmax`、`vmin` |
| 向量-标量操作 | 向量与标量混合 | `vadds`、`vmuls`、`vshls`、`vlrelu` |
| 转换操作 | 数值类型转换和索引生成 | `vci`、`vcvt`、`vtrc` |
| 归约操作 | 跨 lane 归约 | `vcadd`、`vcmax`、`vcmin`、`vcgadd`、`vcgmax` |
| 比较与选择 | 比较、谓词生成、按 mask 选择 | `vcmp`、`vcmps`、`vsel`、`vselr` |
| 数据重排 | pack、permute、interleave、slide 等 | `vintlv`、`vdintlv`、`vpack`、`vslide` |
| SFU 与 DSA | 特殊函数与融合操作 | `vprelu`、`vexpdiff`、`vaxpy`、`vtranspose`、`vsort32` |

## 输入

向量指令常见输入包括：

- 向量寄存器 `!pto.vreg<NxT>`
- 标量寄存器或立即数
- 谓词 `!pto.mask`
- 向量 tile buffer 指针 `!pto.ptr<T, ub>`
- distribution / rounding / alignment 相关属性

## 预期输出

向量指令会产生：

- 向量寄存器结果
- 标量结果（如某些归约）
- 谓词结果
- 通过向量 store 写回向量 tile buffer 的数据

## 副作用

绝大多数向量指令是纯计算操作，没有额外架构副作用。真正有副作用的主要是：

| 类别 | 架构副作用 |
|------|------------|
| 向量加载存储 | 读写向量 tile buffer 可见内存 |
| 比较与选择 | 产生后续操作要消费的谓词结果 |

## Mask 行为

向量操作可以由谓词 mask 控制。mask 宽度必须与目标向量宽度匹配：

- mask 位为 `1` 的 lane 正常参与运算
- mask 位为 `0` 的 lane 是否保留原值、写 0、还是采用其他结果，必须以具体指令页为准

程序不能把 masked lane 的行为当作“默认固定契约”，除非该指令页明确说明。

## 对齐状态

向量非对齐 store（`vstu`、`vstus`、`vstur`）需要维护显式对齐状态。该状态会随着每次 store 更新：

```mlir
%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base
    : !pto.align, index, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, index
```

对于这类有状态 store，还需要尾部 flush 形式（例如 `vstar` / `vstas`）来提交缓存下来的尾部字节。这些形式是 **A5 专属**。

## 约束

- `N` 由元素类型决定，不是自由参数。
- 谓词宽度必须和目标向量宽度一致。
- 对齐要求因操作和 profile 而异。
- 某些非对齐 store 形式只在 A5 可用。
- 不存在隐式类型提升；所有操作数类型必须显式兼容。

## 不允许的情形

- 使用与向量宽度不匹配的谓词
- 在目标 profile 不支持的情况下使用 A5 专属向量形式
- 把 masked lane 的未文档化行为当成稳定契约
- 没有经过显式同步就跨过 DMA / 向量计算之间的顺序边

## 语法

### PTO-AS 形式

```asm
vadd %vdst, %vsrc0, %vsrc1 : !pto.vreg<f32, 64>
vlds %vreg, %ub_ptr[%offset] {dist = "NORM"} : !pto.ptr<f32, ub>
```

### SSA 形式（AS Level 1）

```mlir
%vdst = pto.vadd %vsrc0, %vsrc1, %mask
    : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask) -> !pto.vreg<64xf32>
```

### DPS 形式（AS Level 2）

```mlir
pto.vadd ins(%vsrc0, %vsrc1, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask)
          outs(%vdst : !pto.vreg<64xf32>)
```

完整语法见 [汇编拼写与操作数](../syntax-and-operands/assembly-model_zh.md)。

## C++ 内建接口

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

PTO_INST void VADD(VecDst& dst, VecSrc0& src0, VecSrc1& src1);
PTO_INST void VLDS(VecData& dst, PtrType addr);
PTO_INST void VLDS(VecData& dst, PtrType addr, MaskType pred);
```

## 相关页面

- [向量指令族](../instruction-families/vector-families_zh.md)
- [向量参考入口](../vector/README_zh.md)
- [指令描述格式](../reference/format-of-instruction-descriptions_zh.md)
