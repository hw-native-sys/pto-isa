# pto.pge_b8

生成大于等于比较谓词。

## 语法

### PTO 汇编形式

```text
pge_b8 %dst, %scalar : !pto.mask<b8>, i8
```

### AS Level 1（SSA）

```mlir
%mask = pto.pge_b8 %scalar : i8 -> !pto.mask<b8>
```

### AS Level 2（DPS）

```mlir
pto.pge_b8 ins(%scalar : i8) outs(%mask : !pto.mask<b8>)
```

## 关键约束

- 参与操作的谓词宽度必须兼容。
- pattern / partition token 必须属于文档化取值域。

## 相关页面

- [谓词生成与代数](../../predicate-generation-and-algebra_zh.md)
