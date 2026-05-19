# pto.pset_b16

按模式构造谓词。

## 语法

### PTO 汇编形式

```text
pset_b16 %dst, "PATTERN" : !pto.mask<b16>
```

### AS Level 1（SSA）

```mlir
%mask = pto.pset_b16 "PATTERN" : !pto.mask<b16>
```

### AS Level 2（DPS）

```mlir
pto.pset_b16 "PATTERN" outs(%mask : !pto.mask<b16>)
```

## 关键约束

- 参与操作的谓词宽度必须兼容。
- pattern / partition token 必须属于文档化取值域。

## 相关页面

- [谓词生成与代数](../../predicate-generation-and-algebra_zh.md)
