# 汇编拼写与操作数

PTO ISA 包含一套文本汇编拼写形式，即 PTO-AS，但架构契约本身仍定义在 PTO ISA manual 中。下文给出三种文本层级的 BNF 语法、操作数修饰规则和属性语法；具体指令的特殊拼写由各自的 per-op 页面补充。

## 三层语法系统

PTO 定义三种文本语法层级，它们保留同一套 ISA 契约：

| 层级 | 名称 | 形式 | 常见用途 |
| --- | --- | --- | --- |
| **Assembly Form (PTO-AS)** | 人类可读形式 | `tadd %dst, %src0, %src1` | 文档、伪代码 |
| **SSA Form (AS Level 1)** | MLIR SSA | `%dst = pto.tadd %src0, %src1` | IR、代码生成器 |
| **DPS Form (AS Level 2)** | Functional DPS | `pto.tadd ins(...) outs(...)` | C++ intrinsic 对应形态 |

三种形式在语义上等价。backend 或 verifier 不能因为语法层不同而赋予不同语义。

## BNF 语法

### Assembly Form（PTO-AS）

```text
assembly-program  ::= assembly-stmt*
assembly-stmt     ::= label? op-name operands? ":" type-ref ("#" attribute)*
label             ::= identifier ":"
op-name           ::= ("pto.")? identifier ("_" identifier)*
operands          ::= operand ("," operand)*
operand           ::= register | immediate | memory-operand | mask-operand
register          ::= "%" identifier
immediate         ::= integer | hex-integer | floating-point
hex-integer       ::= "0x" [0-9a-fA-F]+
floating-point    ::= [0-9]+ "." [0-9]+ ("e" [+-]? [0-9]+)?
memory-operand    ::= register "[" register ("," register)* "]"
mask-operand      ::= "%" identifier ":" "!" pto.mask
type-ref          ::= "!" pto "." type-key "<" type-params ">"
type-key          ::= "tile" | "tile_buf" | "vreg" | "ptr" | "partition_tensor_view" | "mask" | "event"
```

### SSA Form（AS Level 1）

```text
ssa-program       ::= ssa-stmt*
ssa-stmt          ::= ssa-result "=" op-name operands ":" ssa-type -> ssa-type
ssa-result        ::= "%" identifier
op-name           ::= "pto." identifier ("_" identifier)*
operands          ::= ssa-operand ("," ssa-operand)*
ssa-operand       ::= ssa-result | immediate | memory-operand
ssa-type          ::= ssa-type-key "<" type-params ">"
ssa-type-key      ::= "tile" | "tile_buf" | "vreg" | "ptr" | "partition_tensor_view" | "mask"
```

### DPS Form（AS Level 2）

```text
dps-program       ::= dps-stmt*
dps-stmt          ::= op-name "ins(" dps-ins ")" "outs(" dps-outs ")"
dps-ins           ::= dps-ins-item ("," dps-ins-item)*
dps-outs          ::= dps-out-item ("," dps-out-item)*
dps-ins-item      ::= ssa-result ":" ssa-type
dps-out-item      ::= ssa-result ":" ssa-type
```

## 操作数修饰规则

### Tile 操作数

PTO-AS 中的 tile 操作数可以带修饰：

```text
%tile
%tile[%r, %c]
%tile!loc=vec
```

在 SSA 形式中，location intent 和 valid-region 相关信息编码在 tile 类型里：

```text
!pto.tile<loc=vec, f32, 16, 16, RowMajor, NoneBox, None, Zero>
!pto.tile_buf<loc=vec, f32, 16, 16, RowMajor, NoneBox, None, Zero>
```

### GlobalTensor 操作数

在 PTO-AS 中，`GlobalTensor` 一般表现为 `memref` 或 `partition_tensor_view`：

```text
%tensor
%tensor[%r, %c]
%tensor[%b, %h, %w, %r, %c]
```

### 谓词操作数

```text
%mask : !pto.mask<G>
```

向量指令把 mask 作为显式操作数：

```text
%result = pto.vadd %src0, %src1, %mask : ... -> ...
```

### 立即数操作数

```text
tadds %dst, %src, 0x3F800000
tshrs %dst, %src, 16
```

## 指令后缀

| 后缀 | 含义 | 例子 |
| --- | --- | --- |
| *(none)* | 标准形式 | `tadd` |
| `s` | 标量变体 | `tadds` |
| `c` | carry / saturating 变体 | `taddc`, `tsubc` |
| `sc` | 标量 + carry | `taddsc`, `tsubsc` |
| `_fp` | 浮点专用变体 | `tstore_fp`, `tinsert_fp` |
| `_acc` | 累加变体 | `tmatmul_acc` |
| `_bias` | 带 bias 变体 | `tmatmul_bias` |
| `_mx` | MX format 变体 | `tgemv_mx` |

## 属性语法

在 PTO-AS 中，属性写在 `#` 后面：

```text
tstore %tile, %tensor #atomic=add
tcmps %dst, %src, 0 #cmp=gt
tmatmul %c, %a, %b #phase=relu
```

在 SSA 中，属性写成 `{key = value}`：

```text
%result = pto.tcmp %src0, %src1 {cmp = "lt"} : ... -> ...
```

## 完整示例

### Tile 逐元素加法

**Assembly Form**:

```text
tadd %dst, %src0, %src1 : !pto.tile<f32, 16, 16>
```

**SSA Form**:

```text
%dst = pto.tadd %src0, %src1 : (!pto.tile<f32, 16, 16>, !pto.tile<f32, 16, 16>) -> !pto.tile<f32, 16, 16>
```

**DPS Form**:

```text
pto.tadd ins(%src0, %src1 : !pto.tile_buf<f32, 16, 16>, !pto.tile_buf<f32, 16, 16>)
          outs(%dst : !pto.tile_buf<f32, 16, 16>)
```

### Tile 加载

```text
tload %tile, %tensor[%r, %c] : (!pto.tile<f32,16,16>, !pto.memref<f32,5>) -> !pto.tile<f32,16,16>
```

### 向量加法

```text
%result = pto.vadd %src0, %src1, %mask : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32>) -> !pto.vreg<64xf32>
```

### 谓词生成

```text
%pred = pto.pge_b32 %src0, %src1 : (!pto.vreg<64xi32>, !pto.vreg<64xi32>) -> !pto.mask<b32>
```

## 文本拼写不能替代的内容

文本拼写不能替代：

- PTO ISA 的机器模型
- PTO ISA 的内存模型
- CPU / A2/A3 / A5 的 target-profile 规则
- 独立于文本拼写之外的架构级合法性规则

## 契约说明

- 文本汇编形式必须与对应 intrinsic 形式保持同一可见语义
- 汇编语法规则必须留在 PTO ISA 的语法与操作数页面中，而不是落入 backend 私有注释
- 会改变语义的语法变体必须被明确文档化

## 相关页面

- [操作数与属性](./operands-and-attributes_zh.md)
- [类型系统](../state-and-types/type-system_zh.md)
- [什么是 PTO 虚拟 ISA](../introduction/what-is-pto-visa_zh.md)
- [指令族总览](../instruction-families/README_zh.md)
- [通用约定](../conventions_zh.md)
