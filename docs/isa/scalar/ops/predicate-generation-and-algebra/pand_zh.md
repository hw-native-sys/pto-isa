# pto.pand

对两个谓词做按位与。

## 语法

### PTO 汇编形式

```text
pand %dst, %src0, %src1 : !pto.mask<G>, !pto.mask<G>, !pto.mask<G>
```

### AS Level 1（SSA）

```mlir
%dst = pto.pand %src0, %src1, %mask : !pto.mask<G>, !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>
```

### AS Level 2（DPS）

```mlir
pto.pand ins(%src0, %src1, %mask : !pto.mask<G>, !pto.mask<G>, !pto.mask<G>) outs(%dst : !pto.mask<G>)
```

## 关键约束

- 参与操作的谓词宽度必须兼容。
- pattern / partition token 必须属于文档化取值域。

## 相关页面

- [谓词生成与代数](../../predicate-generation-and-algebra_zh.md)
