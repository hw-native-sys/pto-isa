# 指令族

本章描述 PTO ISA 的**指令族**（Instruction Set）——共享约束和行为的指令分组。每个族定义了该族所有指令共同遵循的规则，是 per-op 页面之上的抽象层。

## 四类指令集

| 指令集 | 前缀 | 指令族数 | 说明 |
|--------|------|----------|------|
| [Tile 指令族](./tile-families_zh.md) | `pto.t*` | 8 | 逐元素、归约、布局、矩阵乘、数据搬运等 |
| [Vector 指令族](./vector-families_zh.md) | `pto.v*` | 9 | 向量加载存储、一元/二元向量、归约、SFU 等 |
| [标量与控制指令族](./scalar-and-control-families_zh.md) | `pto.*` | 6 | 同步、DMA、谓词、标量算术等 |
| [其他指令族](./other-families_zh.md) | `pto.*` | 2 | 通信与支撑操作 |

## 指令集与指令族的关系

| 层级 | 定义 |
|------|------|
| **指令集（Instruction Set）** | 按功能角色（Tile / Vector / Scalar&Control / Other）分类指令 |
| **族（Family）** | 同一族内共享约束、行为模式和规范语言 |

## 导航地图

```
Tile 指令族
├── 同步与配置             → tassign、tsync、tsetf32mode、tsetfmatrix、tset_img2col_*、tsubview
├── 逐元素 Tile-Tile       → tadd、tsub、tmul、tdiv、tmin、tmax、tcmp、tcvt、tsel 等
├── Tile-标量与立即数      → tadds、tsubs、tmuls、tdi等等vs、tcmps、tsels 等
├── 归约与扩展            → trowsum、tcolmax、trowexpand、tcolexpand 等
├── 内存与数据搬运         → tload、tprefetch、tstore、tstore_fp、mgather、mscatter
├── 矩阵与矩阵-向量        → tgemv、tgemv_mx、tmatmul、tmatmul_acc、tmatmul_bias 等
├── 布局与重排            → tmov、ttrans、textract、tinsert、timg2col、tfillpad 等
└── 非常规与复杂操作       → tprint、tsort32、tgather、tscatter、tquant 等

Vector 指令族
├── 向量加载存储             → vlds、vldas、vgather2、vsld、vsst、vscatter 等
├── 谓词与物化              → vbr、vdup
├── 一元向量运算            → vabs、vneg、vexp、vln、vsqrt、vrec、vrelu 等
├── 二元向量运算            → vadd、vsub、vmul、vdiv、vmax、vmin、vand、vor 等
├── 向量-标量运算           → vadds、vsubs、vmuls、vshls、vlrelu 等
├── 类型转换                → vci、vcvt、vtrc
├── 归约指令                → vcadd、vcmax、vcmin、vcgadd、vcgmax 等
├── 比较与选择              → vcmp、vcmps、vsel、vselr、vselrv2
├── 数据重排                → vintlv、vslide、vshift、vpack、vzunpack 等
└── SFU 与 DSA             → vprelu、vexpdiff、vaxpy、vtranspose、vsort32 等

标量与控制指令族
├── 控制与配置              → nop、barrier、yield；tsetf32mode、tsetfmatrix
├── 流水线同步             → set_flag、wait_flag、pipe_barrier、mem_bar、get_buf
├── DMA 拷贝               → copy_gm_to_ubuf、copy_ubuf_to_gm、copy_ubuf_to_ubuf
├── 谓词加载存储            → pld、plds、psts、pst、pstu
├── 谓词生成                → pset_b8/b16/b32、pge_b8/b16/b32、plt_b8/b16/b32
│                             → pand、por、pxor、pnot、psel、ppack、punpack
├── 共享标量算术            → 跨指令集共享的标量算术运算
└── 共享结构化控制流        → 标量结构化控制流

通信指令族
├── 集体操作                → tbroadcast、tget、tput、tgather、tscatter、treduce、tnotify、ttest、twait
└── 非 ISA 支撑操作          → talias、taxpy、tconcat、tdequant、tfree、thistogram、tpack、tpop、tpush、trandom
```

## 每个族必须定义的内容

1. **Mechanism** — 族的用途说明
2. **Shared Operand Model** — 共同的操作数模型和交互方式
3. **Common 副作用** — 所有族内操作共享的副作用
4. **Shared Constraints** — 适用于全族的合法性规则
5. **Cases That Is Not Allowed** — 全族禁止的条件
6. **Target-Profile Narrowing** — A2/A3 和 A5 的差异
7. **Operation List** — 指向各 per-op 页面的链接

## 章节定位

本章属于手册第 7 章（指令集）的一部分。族文档是 per-op 页面的上一层抽象，同一族的指令共享家族概览页中的共同约束。

## 相关页面

- [指令集总览](../instruction-surfaces/README_zh.md) — 四类指令集总览与数据流关系
- [Tile 参考](../tile/README_zh.md) — Tile 指令逐条参考
- [Vector 参考](../vector/README_zh.md) — 向量指令逐条参考
- [标量与控制参考](../scalar/README_zh.md) — 标量与控制指令逐条参考
- [其他与通信参考](../other/README_zh.md) — 通信与支撑操作逐条参考
