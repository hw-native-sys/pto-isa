# Tile-标量与立即数指令集

Tile-标量类操作把一个 tile 与一个标量或立即数结合。标量在语义上会广播到 tile 的有效区域。多数操作产生数值 tile，比较类变体产生谓词或可作为选择条件使用的结果。

## 指令一览

| 操作 | 说明 |
| --- | --- |
| `pto.tadds` / `tsubs` / `tmuls` / `tdivs` | 与标量做逐元素算术 |
| `pto.tfmods` / `trems` | 与标量做模 / 余数 |
| `pto.tmins` / `tmaxs` | 与标量做逐元素 min / max |
| `pto.tands` / `tors` / `txors` | 与标量做逐元素按位逻辑 |
| `pto.tshls` / `tshrs` | 用标量做逐元素位移 |
| `pto.tlrelu` | 标量斜率的 Leaky ReLU |
| `pto.taddsc` / `tsubsc` | 结合标量和第二个 tile 的融合逐元素运算 |
| `pto.texpands` | 用标量填充整个目标 tile |
| `pto.tcmps` | tile 与标量比较 |
| `pto.tsels` | 用 mask / 标量参与逐元素选择 |

## 机制

对目标 tile 的 valid region 中每个元素 `(r, c)`：

$$ \mathrm{dst}_{r,c} = f(\mathrm{src}_{r,c}, \mathrm{scalar}) $$

这里的 `scalar` 可以是：

- 标量寄存器值
- 编译期立即数
- 运行时传入的普通标量

PTO 不允许在 tile-标量操作里依赖隐式类型提升。标量如何广播、比较结果如何编码、饱和与否怎样处理，都属于具体操作自己的架构语义。

## 目标 Profile 支持

| 元素类型 | CPU | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `f32 / f16 / bf16` | Yes | Yes | Yes |
| `i8/u8, i16/u16, i32/u32, i64/u64` | Yes | Yes | Yes |

实际每条指令支持的子集，以各自 per-op 页面为准。

## 约束

- 标量类型必须与 tile 元素类型兼容。
- `TSHLS` / `TSHRS` 将标量解释为无符号 shift count。
- `TCMPS` 的结果不应被当成普通数值 tile 使用，除非对应 target/profile 明确约定其编码。
- 需要保持 valid region 语义的操作，迭代域都以 `dst` 的 valid row / col 为准。

## 不允许的情形

- 使用与 tile 元素类型不兼容的标量。
- 依赖隐式类型提升。
- 对位移类操作传入超出元素位宽的移位量。
- 把不同 target profile 的实现细节当成跨目标保证。

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [Tile 指令面](../instruction-surfaces/tile-instructions_zh.md)
