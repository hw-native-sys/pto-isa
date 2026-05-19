# pto.psel

按条件选择谓词结果。

## 语法

### PTO 汇编形式

```text
psel %dst, %src0, %src1, %sel : !pto.mask<G>, !pto.mask<G>, !pto.mask<G>, !pto.mask<G>
```

### AS Level 1（SSA）

```mlir
%dst = pto.psel %src0, %src1, %sel, %mask : !pto.mask<G>, !pto.mask<G>, !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>
```

### AS Level 2（DPS）

```mlir
pto.psel ins(%src0, %src1, %sel, %mask : !pto.mask<G>, !pto.mask<G>, !pto.mask<G>, !pto.mask<G>) outs(%dst : !pto.mask<G>)
```

## 关键约束

- 参与操作的谓词宽度必须兼容。
- pattern / partition token 必须属于文档化取值域。

## 相关页面

- [谓词生成与代数](../../predicate-generation-and-algebra_zh.md)
