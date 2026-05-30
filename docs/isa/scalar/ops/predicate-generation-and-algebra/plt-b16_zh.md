# pto.plt_b16

生成小于比较谓词。

## 语法

### PTO 汇编形式

```text
plt_b16 %dst, %scalar_in : !pto.mask<b16>, i16 -> !pto.mask<b16>, i16
```

### AS Level 1（SSA）

```mlir
%mask, %scalar_out = pto.plt_b16 %scalar_in : i16 -> !pto.mask<b16>, i16
```

### AS Level 2（DPS）

```mlir
pto.plt_b16 ins(%scalar_in : i16) outs(%mask, %scalar_out : !pto.mask<b16>, i16)
```

## 关键约束

- 参与操作的谓词宽度必须兼容。
- pattern / partition token 必须属于文档化取值域。

## 相关页面

- [谓词生成与代数](../../predicate-generation-and-algebra_zh.md)
