# Tile 指令集

`pto.t*` 指令集覆盖以 tile 为中心的计算、搬运、重排、归约和同步。tile 的 shape、layout、role 与 valid region 都属于架构可见状态。

## 指令集概览

Tile 指令作用在 tile 上。与向量寄存器或纯标量寄存器不同，tile 自带以下元数据：

- shape
- layout
- role / location intent
- valid region

因此，Tile 指令的输出不只是“一个数值结果”，还可能是：

- 新的 tile payload
- valid region 的解释变化
- 资源绑定状态
- 明确的同步边

## 数据流

```text
GlobalMemory
    │
    │  TLOAD: GM → local tile buffer
    ▼
Tile Buffer ──► Tile Compute ──► Tile Buffer ──► TSTORE: Tile Buffer → GM
(Vec/Mat/Acc)    (pto.tadd, TMATMUL, 等)          (Vec/Mat/Acc)
```

## 指令分类

| 类别 | 说明 | 示例 |
|------|------|------|
| 同步与配置 | 资源绑定、事件建立、tile 侧配置 | `tassign`、`tsync`、`tsettf32mode`、`tset_img2col_rpt` |
| 逐元素 Tile-Tile | tile 与 tile 之间的逐元素运算 | `tadd`、`tmul`、`tcmp`、`tcvt`、`tsel`、`trelu` |
| Tile-标量与立即数 | tile 与标量 / 立即数混合运算 | `tadds`、`tmuls`、`tlrelu`、`tcmps` |
| 归约与扩展 | 按行 / 列归约或广播扩展 | `trowsum`、`tcolmax`、`trowexpand`、`tcolexpand` |
| 内存与数据搬运 | GM↔tile 搬运、gather / scatter、fix-pipe store | `tload`、`tstore`、`tstore_fp`、`mgather`、`mscatter` |
| 矩阵与矩阵-向量 | GEMV、matmul 及变体 | `tgemv`、`tgemv_mx`、`tmatmul`、`tmatmul_acc`、`tmatmul_bias` |
| 布局与重排 | reshape、transpose、extract、insert、img2col | `tmov`、`ttrans`、`treshape`、`textract`、`tinsert`、`timg2col` |
| 不规则与复杂 | 排序、量化、打印、partial 类操作 | `tmrgsort`、`tsort32`、`tquant`、`thistogram`、`tprint` |

## 输入

Tile 指令常见输入包括：

- 源 tile（只读）
- 目标 tile / tile buffer（写入或读写）
- 标量修饰符和立即数
- GM 视图（`!pto.partition_tensor_view<...>`）
- 可选事件链（`RecordEvent` 或 `WaitEvents...`）

## 预期输出

Tile 指令会产生：

- 目标 tile payload
- valid region 或布局解释变化
- 显式状态更新（例如地址绑定）
- 同步边

## 副作用

| 类别 | 架构副作用 |
|------|------------|
| 内存与数据搬运 | 读写 GM 可见数据 |
| 同步与配置 | 建立同步边或绑定 tile 地址 |
| 不规则与复杂 | 可能产生调试输出或修改分配状态 |

## Valid Region 模型

所有逐元素 tile 操作都以 **目标 tile 的 valid region** 为迭代域。对目标 valid region 内的每个 `(r, c)`：

- 会读取每个源 tile 对应位置 `(r, c)` 的元素
- 即使源 tile 自己的 valid region 不覆盖该位置，也仍然会去读，只是结果属于实现定义
- 程序不能依赖源 tile 域外值的具体内容，除非该操作页明确说明了行为

完整模型见 [Tile 与有效区域](../programming-model/tiles-and-valid-regions_zh.md)。

## 约束

- tile 合法性从来不只取决于 dtype；shape、layout、role、valid region 都可能参与判断。
- 多输入 tile 指令必须明确说明 valid region 如何组合。
- 部分 tile 形式受 profile 限制：MX block-scale tile（`Left`、`Right`、`ScaleLeft`、`ScaleRight`）是 A5 专属；FP8 / FP4 家族元素类型也是 A5 专属。
- `tsethf32mode` 与 `tsetfmatrix` 虽然名字保留 `t` 前缀，但在手册中归类到标量 / 控制路径，因为它们修改的是标量可见模式状态，而不是 tile payload。
- Tile 指令不会自动继承向量寄存器语义。
- 不存在默认广播；所有源 tile 的 shape 都必须和目标 tile 的使用方式兼容。

## 不允许的情形

- 把 valid region 外的值当作稳定语义使用
- 假设 tile 指令天然继承 vector lane 语义
- 依赖未文档化的隐式广播、隐式 reshape、隐式 valid-region 修复
- 在 CPU 或 A2A3 上使用 MX block-scale tile
- 在 CPU 或 A2A3 上使用 FP8 / FP4 家族元素类型

## 语法

### PTO-AS 形式

```asm
tadd %dst, %src0, %src1 : !pto.tile<f32, 16, 16>
```

### SSA 形式（AS Level 1）

```mlir
%dst = pto.tadd %src0, %src1
    : (!pto.tile<f32, 16, 16>, !pto.tile<f32, 16, 16>)
    -> !pto.tile<f32, 16, 16>
```

### DPS 形式（AS Level 2）

```mlir
pto.tadd ins(%src0, %src1 : !pto.tile_buf<f32, 16, 16>, !pto.tile_buf<f32, 16, 16>)
          outs(%dst : !pto.tile_buf<f32, 16, 16>)
```

完整语法见 [汇编拼写与操作数](../syntax-and-operands/assembly-model_zh.md)。

## C++ 内建接口

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INST RecordEvent TADD(TileDst& dst, TileSrc0& src0, TileSrc1& src1);

template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData& dst, GlobalData& src, WaitEvents&... events);

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix,
                             WaitEvents&... events);
```

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [Tile 参考入口](../tile/README_zh.md)
- [指令描述格式](../reference/format-of-instruction-descriptions_zh.md)
